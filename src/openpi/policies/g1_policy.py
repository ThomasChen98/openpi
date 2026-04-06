"""
G1 Policy Transforms

G1 robot configuration:
- Arms: 14 DOF (7 per arm)
- Hands: 14 DOF Dex3 (7 per hand)
- Waist: 1 DOF (waist_yaw)
- Camera: 1 (ego_cam / cam_head)

State Space (29 dims):
    [0:28]  qpos        - arm (14) + hand (14) joint positions
    [28]    waist_yaw   - waist yaw joint position

Action Space (29 dims):
    [0:28]  upper_body  - arm (14) + hand (14) joint targets
    [28]    waist_yaw   - waist yaw joint target

HDF5 Data format (from episode_writer_hdf5.py):
/observations/
    qpos: [T, 29] - arm (14) + hand (14) + waist_yaw (1) joint positions
    qvel: [T, 29] - joint velocities (same structure)
    images/
        cam_head: [T, H, W, 3] - RGB images
/action: [T, 29] - joint targets (arm + hand + waist_yaw)
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# G1 dimension constants - 29-dim format
G1_UPPER_BODY_DIM = 28  # 14 arm + 14 Dex3 hand
G1_WAIST_DIM = 1        # waist yaw
G1_STATE_DIM = 29       # upper_body(28) + waist_yaw(1)
G1_ACTION_DIM = 29      # same as state


def make_g1_example() -> dict:
    """Creates a random input example for the G1 policy."""
    return {
        "state": np.random.rand(G1_STATE_DIM).astype(np.float32),  # 29 dims
        "images": {
            "cam_head": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        },
        "prompt": "pick up the bottle",
    }


@dataclasses.dataclass(frozen=True)
class G1AlignDatasetActionsToPolicyDim(transforms.DataTransformFn):
    """Map LeRobot dataset actions to the policy action width before delta / normalization.

    - 28 from 29: drop waist yaw (last dim).
    - 16 from 29: [arm(14), mean(left_hand), mean(right_hand)] for binary-style training.
    If actions already match ``policy_action_dim``, this is a no-op.
    """

    policy_action_dim: int

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        a = np.asarray(data["actions"], dtype=np.float32)
        if a.shape[-1] == self.policy_action_dim:
            return data
        if self.policy_action_dim == 28 and a.shape[-1] >= 28:
            data["actions"] = a[..., :28]
            return data
        if self.policy_action_dim == 16 and a.shape[-1] >= 29:
            left_m = np.mean(a[..., 14:21], axis=-1, keepdims=True)
            right_m = np.mean(a[..., 21:28], axis=-1, keepdims=True)
            data["actions"] = np.concatenate([a[..., :14], left_m, right_m], axis=-1).astype(np.float32)
            return data
        raise ValueError(
            f"Cannot align actions with shape {a.shape} to policy_action_dim={self.policy_action_dim}"
        )


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
       - state: qpos [29] = upper_body(28) + waist_yaw(1)
       - Passes through directly
    
    2. Inference (from g1_remote_client):
       - state: [29] already in correct format
       - Passes through directly
    
    Output state: [29] = upper_body(28) + waist_yaw(1)
    """

    model_type: _model.ModelType
    # Match H1 format: accept wrist cameras (will be zero-padded in dataset)
    # G1Inputs ignores them and creates zeros anyway
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_head", "cam_left_wrist", "cam_right_wrist")

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

        # State should be 29-dim for the model: upper_body(28) + waist_yaw(1)
        input_state = np.asarray(data["state"], dtype=np.float32)

        if len(input_state) == G1_STATE_DIM:
            state = input_state
        elif len(input_state) == G1_UPPER_BODY_DIM:
            state = np.concatenate([input_state, np.zeros(G1_WAIST_DIM, dtype=np.float32)])
        elif len(input_state) == 16:
            # Execution client: [arm(14), bin_L, bin_R] — pad hands + waist for a 29-D qpos-like state
            state = np.concatenate(
                [input_state[:14], np.zeros(14, dtype=np.float32), np.zeros(G1_WAIST_DIM, dtype=np.float32)]
            )
        else:
            raise ValueError(
                f"Expected state dim {G1_STATE_DIM}, {G1_UPPER_BODY_DIM}, or 16, got {len(input_state)}"
            )

        # Create inputs dict
        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,  # [29]
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
    
    Model outputs are sliced to the policy action width (pi05 uses a wider head).
    
    Args:
        action_dim: Number of action dimensions to return.
            - 29: arm(14)+hand(14)+waist_yaw(1)
            - 28: arm(14)+hand(14)
            - 16: arm(14)+binary grippers (2)
    """
    
    action_dim: int = 29  # Return full 29-dim actions by default

    def __call__(self, data: dict) -> dict:
        # Extract first action_dim dimensions from model output
        return {"actions": np.asarray(data["actions"][:, :self.action_dim])}
