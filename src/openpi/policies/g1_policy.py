"""
G1 Policy Transforms

G1 robot configuration:
- Arms: 14 DOF (7 per arm)
- Hands: 14 DOF Dex3 (7 per hand)
- Locomotion: 3 DOF (vx, vy, vyaw)
- Camera: 1 (ego_cam / cam_head)

State Space (32 dims):
    [0:28]  qpos        - arm (14) + hand (14) joint positions
    [28:31] rpy         - roll, pitch, yaw from IMU (radians)
    [31]    yaw_rate    - yaw angular velocity from gyroscope

Action Space (32 dims):
    [0:28]  upper_body  - arm (14) + hand (14) joint targets
    [28]    vx          - forward/backward velocity command
    [29]    vy          - strafe left/right velocity command  
    [30]    vyaw        - turn (yaw angular velocity) command
    [31]    padding     - zero padding

HDF5 Data format (from episode_writer_hdf5.py):
/observations/
    qpos: [T, 28] - arm (14) + hand (14) joint positions
    loco_state: [T, 17] - locomotion state:
        [0]     mode_machine    - FSM state
        [1:4]   rpy             - roll, pitch, yaw (radians)
        [4:8]   quaternion      - orientation (w, x, y, z)
        [8:11]  accelerometer   - linear acceleration
        [11:14] gyroscope       - angular velocity (wx, wy, wz)
        [14:17] leg_joints      - knee positions (height proxy)
    images/
        ego_cam: [T, H, W, 3] - RGB images
/action: [T, 28] - arm + hand joint targets
/loco_action: [T, 20] - joysticks (4: Lx, Ly, Rx, Ry) + buttons (16)

Training transforms build 32-dim state/action from HDF5 data:
- State: qpos[28] + loco_state[1:4](rpy) + loco_state[13](gyro_z) = 32
- Action: action[28] + loco_action[0:3](vx,vy,vyaw) + padding = 32
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# G1 dimension constants
G1_QPOS_DIM = 28      # 14 arm + 14 Dex3 hand
G1_RPY_DIM = 3        # roll, pitch, yaw from IMU
G1_YAWRATE_DIM = 1    # yaw angular velocity from gyroscope
G1_STATE_DIM = 32     # qpos(28) + rpy(3) + yaw_rate(1)

G1_UPPER_BODY_DIM = 28  # 14 arm + 14 Dex3 hand
G1_LOCO_DIM = 3         # vx, vy, vyaw
G1_ACTION_DIM = 32      # upper_body(28) + loco(3) + padding(1)


def make_g1_example() -> dict:
    """Creates a random input example for the G1 policy."""
    return {
        "state": np.random.rand(G1_STATE_DIM).astype(np.float32),  # 32 dims
        "images": {
            "cam_head": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        },
        "prompt": "pick up the bottle",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H, W, C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class G1Inputs(transforms.DataTransformFn):
    """
    Transform inputs for G1 robot policy.
    
    Handles two input scenarios:
    
    1. Training (from HDF5 via repack):
       - state: qpos [28]
       - loco_state: [17] containing rpy at [1:4] and gyroscope at [11:14]
       - Builds: [qpos(28), rpy(3), yaw_rate(1)] = 32 dims
    
    2. Inference (from g1_remote_client):
       - state: [32] already in correct format
       - Passes through directly
    
    Output state: [32] = qpos(28) + rpy(3) + yaw_rate(1)
    """

    model_type: _model.ModelType
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_head",)

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        
        # Get base image (head camera)
        # Support both "cam_head" and "ego_cam" as input names
        if "cam_head" in in_images:
            base_image = _parse_image(in_images["cam_head"])
        elif "ego_cam" in in_images:
            base_image = _parse_image(in_images["ego_cam"])
        else:
            raise ValueError(f"Expected 'cam_head' or 'ego_cam' in images, got {tuple(in_images)}")

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # G1 has no wrist cameras - fill with zeros
        # For pi05, we set masks to True even for padding images
        for dest in ["left_wrist_0_rgb", "right_wrist_0_rgb"]:
            images[dest] = np.zeros_like(base_image)
            image_masks[dest] = np.True_ if self.model_type != _model.ModelType.PI0 else np.False_

        # Build 32-dim state
        input_state = np.asarray(data["state"], dtype=np.float32)
        
        if len(input_state) == G1_STATE_DIM:
            # Inference path: state is already 32 dims, pass through
            state = input_state
        elif len(input_state) == G1_QPOS_DIM:
            # Training path: state is qpos (28), need to add loco info
            qpos = input_state
            
            # Get RPY and yaw_rate from loco_state if available
            if "loco_state" in data and data["loco_state"] is not None:
                loco_state = np.asarray(data["loco_state"], dtype=np.float32)
                rpy = loco_state[1:4]         # indices 1-3: roll, pitch, yaw
                yaw_rate = loco_state[13:14]  # index 13: gyroscope z (wz)
            else:
                rpy = np.zeros(G1_RPY_DIM, dtype=np.float32)
                yaw_rate = np.zeros(G1_YAWRATE_DIM, dtype=np.float32)
            
            # Build state: [qpos(28), rpy(3), yaw_rate(1)] = 32 dims
            state = np.concatenate([qpos, rpy, yaw_rate])
        else:
            raise ValueError(f"Expected state dim {G1_STATE_DIM} or {G1_QPOS_DIM}, got {len(input_state)}")

        # Create inputs dict
        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,  # [32]
        }

        # Actions are only available during training
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class G1Outputs(transforms.DataTransformFn):
    """
    Transform outputs from model back to G1 action format.
    
    Model outputs 32 dims: [upper_body(28), vx, vy, vyaw, padding]
    
    Args:
        action_dim: Number of action dimensions to return.
            - 32: Full action space (default)
            - 31: Without padding [upper_body(28), vx, vy, vyaw]
            - 28: Upper body only [arm(14), hand(14)]
    """
    
    action_dim: int = 32  # Return full 32-dim actions by default

    def __call__(self, data: dict) -> dict:
        # Extract first action_dim dimensions from model output
        return {"actions": np.asarray(data["actions"][:, :self.action_dim])}


@dataclasses.dataclass(frozen=True)
class G1ActionsFromHDF5(transforms.DataTransformFn):
    """
    Build 32-dim actions from HDF5 data during training.
    
    Combines:
    - action: [28] upper body joint targets
    - loco_action: [20] joystick/button inputs, we use [0:3] as vx, vy, vyaw
    
    Output: [32] = [upper_body(28), vx, vy, vyaw, 0]
    
    Note: loco_action format from HDF5:
        [0] Lx  -> maps to vy (strafe, but we use Ly for forward)
        [1] Ly  -> maps to vx (forward/back)
        [2] Rx  -> maps to vyaw (turn)
        [3] Ry  -> not used
        [4:20] buttons -> not used for action
    
    The joystick mapping matches the teleop convention where:
    - Ly (left stick Y) = forward/backward
    - Lx (left stick X) = strafe
    - Rx (right stick X) = turn
    """

    def __call__(self, data: dict) -> dict:
        # Get upper body action (28 dims)
        upper_body = np.asarray(data["actions"], dtype=np.float32)
        
        if len(upper_body.shape) == 1:
            upper_body = upper_body.reshape(1, -1)
        
        batch_size = upper_body.shape[0]
        
        # Get locomotion commands from loco_action if available
        if "loco_action" in data and data["loco_action"] is not None:
            loco_action = np.asarray(data["loco_action"], dtype=np.float32)
            if len(loco_action.shape) == 1:
                loco_action = loco_action.reshape(1, -1)
            
            # Map joystick to velocity commands
            # Teleop convention: Ly=forward, Lx=strafe, Rx=turn
            # loco_action: [Lx, Ly, Rx, Ry, ...]
            vx = -loco_action[:, 1:2]   # Ly -> forward (negated for intuitive control)
            vy = -loco_action[:, 0:1]   # Lx -> strafe (negated for intuitive control)
            vyaw = -loco_action[:, 2:3] # Rx -> turn (negated for intuitive control)
        else:
            # No locomotion data, use zeros
            vx = np.zeros((batch_size, 1), dtype=np.float32)
            vy = np.zeros((batch_size, 1), dtype=np.float32)
            vyaw = np.zeros((batch_size, 1), dtype=np.float32)
        
        # Padding
        padding = np.zeros((batch_size, 1), dtype=np.float32)
        
        # Build 32-dim action: [upper_body(28), vx, vy, vyaw, padding]
        actions = np.concatenate([
            upper_body[:, :G1_UPPER_BODY_DIM],
            vx, vy, vyaw, padding
        ], axis=1)
        
        return {**data, "actions": actions}
