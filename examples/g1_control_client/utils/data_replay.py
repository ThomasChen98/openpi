"""G1 HDF5 Data Replay Utilities

Data loading and joint mapping utilities for G1 robot visualization.

G1 Data Format:
- qpos/action: 28 DOF (14 arm + 14 hand)
  - [0:7] left arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)
  - [7:14] right arm
  - [14:21] left hand (thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1)
  - [21:28] right hand

- loco_state: 17 DOF
  [0] mode_machine (robot state: 0=damping, 1=stand, 5=walk, etc.)
  [1:4] imu rpy (roll, pitch, yaw in radians)
  [4:8] imu quaternion (w, x, y, z)
  [8:11] accelerometer (x, y, z in m/s^2)
  [11:14] gyroscope (x, y, z in rad/s)
  [14:17] leg joints (left_knee, right_knee, avg_knee - proxy for body height)

- loco_action: 20 DOF
  [0:4] joysticks (Lx, Ly, Rx, Ry in range [-1, 1])
  [4:20] buttons (L1, L2, R1, R2, A, B, X, Y, Up, Down, Left, Right, Select, Start, F1, F3)

G1 URDF Joint Order (43 revolute joints):
  [0:6] left leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
  [6:12] right leg
  [12:15] waist (yaw, roll, pitch)
  [15:22] left arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)
  [22:29] left hand (thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1)
  [29:36] right arm
  [36:43] right hand
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


# Mode machine state names
MODE_MACHINE_NAMES = {
    0: "DAMPING",
    1: "STAND",
    2: "WALK_W",
    3: "WALK_S",
    4: "WALK_WS",
    5: "WALK",
    6: "AI",
    7: "SIT",
    8: "FORCE_STAND",
    9: "FORCE_SIT",
}


def load_hdf5_data(hdf5_path: str) -> dict:
    """Load data from G1 HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
        
    Returns:
        Dictionary containing:
            - actions: (N, 28) array for arm+hand joints
            - qpos: (N, 28) array
            - qvel: (N, 28) array if available
            - loco_state: (N, 17) array if available
            - loco_action: (N, 20) array if available
            - camera_data: dict of camera images
            - image_formats: dict of image formats per camera
            - camera_topics: list of available camera names
            - num_frames: total number of frames
            - num_joints: action dimensions (28)
            - has_loco_data: boolean indicating if locomotion data exists
            - robot_name: robot name from attributes
            - fps: recording FPS from attributes
    """
    print(f"Loading HDF5 file: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # Load basic joint data
        actions = f['/action'][:]
        qpos = f['/observations/qpos'][:]
        
        # Load qvel if available
        qvel = None
        if '/observations/qvel' in f:
            qvel = f['/observations/qvel'][:]
        
        # Load locomotion data if available
        loco_state = None
        loco_action = None
        has_loco_data = f.attrs.get('has_loco_data', False)
        
        if '/observations/loco_state' in f:
            loco_state = f['/observations/loco_state'][:]
            print(f"Loaded loco_state: {loco_state.shape}")
        
        if '/loco_action' in f:
            loco_action = f['/loco_action'][:]
            print(f"Loaded loco_action: {loco_action.shape}")
        
        # Load metadata
        robot_name = f.attrs.get('robot_name', 'G1_29')
        if isinstance(robot_name, bytes):
            robot_name = robot_name.decode('utf-8')
        fps = f.attrs.get('fps', 30.0)
        
        # Find all available camera topics
        camera_topics = []
        if '/observations/images' in f:
            for key in f['/observations/images'].keys():
                camera_topics.append(key)
        
        print(f"Available camera topics: {camera_topics}")
        
        # Load image data for all available cameras
        camera_data = {}
        image_formats = {}
        
        for topic in camera_topics:
            topic_path = f'/observations/images/{topic}'
            dataset = f[topic_path]
            
            # Check if it's direct array format (4D: N, H, W, 3) or JPEG encoded (1D: N,)
            if len(dataset.shape) == 4 and dataset.shape[-1] == 3:
                # Direct array format: (N, H, W, 3)
                print(f"Loading {topic}: {dataset.shape} (direct array format)")
                camera_data[topic] = dataset[:]
                image_formats[topic] = "array"
            else:
                # JPEG encoded format: (N,)
                print(f"Loading {topic}: {dataset.shape} (JPEG encoded format)")
                camera_data[topic] = []
                for i in tqdm(range(len(dataset)), desc=f"Loading {topic}", unit="frames"):
                    camera_data[topic].append(dataset[i])
                image_formats[topic] = "jpeg"
        
        num_frames = len(actions)
        
    print(f"Loaded {num_frames} frames with {actions.shape[1]} joints")
    print(f"Robot: {robot_name}, FPS: {fps}, Has loco data: {has_loco_data}")
    
    return {
        'actions': actions,
        'qpos': qpos,
        'qvel': qvel,
        'loco_state': loco_state,
        'loco_action': loco_action,
        'camera_data': camera_data,
        'image_formats': image_formats,
        'camera_topics': camera_topics,
        'num_frames': num_frames,
        'num_joints': actions.shape[1],
        'has_loco_data': has_loco_data or (loco_state is not None),
        'robot_name': robot_name,
        'fps': fps,
    }


def decode_jpeg_image(img_data: bytes) -> Image.Image:
    """Decode JPEG image data.
    
    Args:
        img_data: JPEG encoded image bytes
        
    Returns:
        PIL Image
    """
    return Image.open(io.BytesIO(img_data))


def euler_to_quaternion(pitch: float, yaw: float, roll: float) -> tuple[float, float, float, float]:
    """Convert Euler angles to quaternion (w, x, y, z).
    
    Args:
        pitch: Rotation around X-axis (degrees)
        yaw: Rotation around Y-axis (degrees) 
        roll: Rotation around Z-axis (degrees)
        
    Returns:
        Quaternion as (w, x, y, z)
    """
    # Convert degrees to radians
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    roll_rad = np.radians(roll)
    
    # Compute quaternions for each axis rotation
    cy = np.cos(yaw_rad * 0.5)
    sy = np.sin(yaw_rad * 0.5)
    cp = np.cos(pitch_rad * 0.5)
    sp = np.sin(pitch_rad * 0.5)
    cr = np.cos(roll_rad * 0.5)
    sr = np.sin(roll_rad * 0.5)
    
    # Quaternion multiplication: q = qy * qp * qr
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return (w, x, y, z)


def extract_joints_for_urdf_g1_28dof(
    joint_positions: np.ndarray,
    loco_state: np.ndarray | None = None,
) -> np.ndarray:
    """Map 28 DOF G1 joint positions to 43 DOF URDF joint order.
    
    G1 data format (28 DOF):
    - [0:7] left arm joints
    - [7:14] right arm joints  
    - [14:21] left hand joints
    - [21:28] right hand joints
    
    G1 URDF joint order (43 DOF):
    - [0:6] left leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
    - [6:12] right leg
    - [12:15] waist (yaw, roll, pitch)
    - [15:22] left arm
    - [22:29] left hand
    - [29:36] right arm
    - [36:43] right hand
    
    Args:
        joint_positions: Raw joint positions from HDF5 data (28 dims)
        loco_state: Optional locomotion state (17 dims) for leg joint positions
        
    Returns:
        Joint positions for URDF visualization (43 dims)
    """
    if len(joint_positions) != 28:
        raise ValueError(f"Expected 28 DoF input, got {len(joint_positions)} DoF")
    
    # Initialize all joints to zero
    urdf_joints = np.zeros(43)
    
    # If loco_state is available, use knee joint positions for leg visualization
    if loco_state is not None and len(loco_state) >= 17:
        # loco_state[14] = left_knee, loco_state[15] = right_knee
        # These are indices into the leg joints array
        # left leg: [0:6] = hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
        # right leg: [6:12] = same structure
        urdf_joints[3] = loco_state[14]  # left_knee_joint
        urdf_joints[9] = loco_state[15]  # right_knee_joint
    
    # Extract arm joints
    left_arm = joint_positions[0:7]   # shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
    right_arm = joint_positions[7:14]
    
    # Extract hand joints
    left_hand = joint_positions[14:21]   # thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1
    right_hand = joint_positions[21:28]
    
    # Map to URDF order
    urdf_joints[15:22] = left_arm    # left arm
    urdf_joints[22:29] = left_hand   # left hand
    urdf_joints[29:36] = right_arm   # right arm
    urdf_joints[36:43] = right_hand  # right hand
    
    return urdf_joints


def get_mode_machine_name(mode: int) -> str:
    """Get human-readable name for mode_machine value.
    
    Args:
        mode: mode_machine integer value
        
    Returns:
        Human-readable mode name
    """
    return MODE_MACHINE_NAMES.get(int(mode), f"UNKNOWN({int(mode)})")


def format_loco_state(loco_state: np.ndarray) -> dict:
    """Format loco_state array into a readable dictionary.
    
    Args:
        loco_state: 17-dim loco_state array
        
    Returns:
        Dictionary with named fields
    """
    if loco_state is None or len(loco_state) < 17:
        return {}
    
    return {
        'mode_machine': get_mode_machine_name(loco_state[0]),
        'rpy': {
            'roll': float(loco_state[1]),
            'pitch': float(loco_state[2]),
            'yaw': float(loco_state[3]),
        },
        'quaternion': {
            'w': float(loco_state[4]),
            'x': float(loco_state[5]),
            'y': float(loco_state[6]),
            'z': float(loco_state[7]),
        },
        'accelerometer': {
            'x': float(loco_state[8]),
            'y': float(loco_state[9]),
            'z': float(loco_state[10]),
        },
        'gyroscope': {
            'x': float(loco_state[11]),
            'y': float(loco_state[12]),
            'z': float(loco_state[13]),
        },
        'leg_joints': {
            'left_knee': float(loco_state[14]),
            'right_knee': float(loco_state[15]),
            'avg_knee': float(loco_state[16]),
        },
    }


def format_loco_action(loco_action: np.ndarray) -> dict:
    """Format loco_action array into a readable dictionary.
    
    Args:
        loco_action: 20-dim loco_action array
        
    Returns:
        Dictionary with named fields
    """
    if loco_action is None or len(loco_action) < 20:
        return {}
    
    button_names = ['L1', 'L2', 'R1', 'R2', 'A', 'B', 'X', 'Y', 
                    'Up', 'Down', 'Left', 'Right', 'Select', 'Start', 'F1', 'F3']
    
    buttons = {}
    for i, name in enumerate(button_names):
        buttons[name] = bool(loco_action[4 + i] > 0.5)
    
    return {
        'joysticks': {
            'Lx': float(loco_action[0]),
            'Ly': float(loco_action[1]),
            'Rx': float(loco_action[2]),
            'Ry': float(loco_action[3]),
        },
        'buttons': buttons,
        'active_buttons': [name for name, pressed in buttons.items() if pressed],
    }


def get_urdf_path() -> str:
    """Get the path to the G1 URDF file.
    
    Returns:
        Absolute path to G1 URDF
    """
    script_dir = Path(__file__).parent.parent
    urdf_path = script_dir / "assets" / "g1" / "g1_body29_hand14.urdf"
    return str(urdf_path)
