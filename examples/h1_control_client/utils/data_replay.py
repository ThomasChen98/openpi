"""HDF5 Data Replay with Viser

Replay robot motion data from HDF5 files frame by frame using viser visualization.

Features:
* Load HDF5 data containing robot joint positions and camera images
* Visualize robot motion using viser URDF viewer
* Frame-by-frame playback with play/pause controls
* Adjustable playback speed
* Display camera images alongside robot visualization
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import h5py
import numpy as np
import tyro
from PIL import Image
from yourdfpy import URDF
import io
from tqdm import tqdm

import viser
from viser.extras import ViserUrdf

# Global rotation settings - modify these values to change camera orientations
EGO_PITCH = -40.0    # Ego camera pitch angle (degrees)
EGO_YAW = 0.0      # Ego camera yaw angle (degrees) 
EGO_ROLL = 0.0     # Ego camera roll angle (degrees)

BIRDVIEW_PITCH = 20.0    # Birdview camera pitch angle (degrees)
BIRDVIEW_YAW = 0.0      # Birdview camera yaw angle (degrees)
BIRDVIEW_ROLL = 0.0     # Birdview camera roll angle (degrees)


def load_hdf5_data(hdf5_path: str) -> dict:
    """Load data from HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
        
    Returns:
        Dictionary containing:
            - actions: (N, num_joints) array
            - qpos: (N, num_joints) array
            - ego_cam: list of N JPEG encoded images (if available)
            - birdview_cam: list of N JPEG encoded images (if available)
            - num_frames: total number of frames
            - has_ego_cam: boolean indicating if ego camera data exists
            - has_birdview_cam: boolean indicating if birdview camera data exists
    """
    print(f"Loading HDF5 file: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # Load joint data
        actions = f['/action'][:]
        qpos = f['/observations/qpos'][:]
        
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
                # Use tqdm for progress bar
                for i in tqdm(range(len(dataset)), desc=f"Loading {topic}", unit="frames"):
                    camera_data[topic].append(dataset[i])
                image_formats[topic] = "jpeg"
        
        num_frames = len(actions)
        
    print(f"Loaded {num_frames} frames with {actions.shape[1]} joints")
    
    return {
        'actions': actions,
        'qpos': qpos,
        'camera_data': camera_data,
        'image_formats': image_formats,
        'camera_topics': camera_topics,
        'num_frames': num_frames,
        'num_joints': actions.shape[1]
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


def extract_hand_joints_for_urdf_26dof(joint_positions: np.ndarray) -> np.ndarray:
    """Extract and map hand joints for 26 DoF datasets to URDF joint order.
    
    For 26 DoF datasets: 14 DoF upper body + 12 DoF hands
    - First 14 dimensions: upper body joints
    - Last 12 dimensions: hand joints (6 per hand)
    
    The hand joints are already in the correct order:
    - Left hand (indices 14-19): thumb_yaw, thumb_pitch, index, middle, ring, pinky
    - Right hand (indices 20-25): thumb_yaw, thumb_pitch, index, middle, ring, pinky
    
    We need to:
    1. Pad the 14 upper body joints to 27 robot joints (add 13 zeros for legs)
    2. Scale the 12 hand joints from encoder values (0-1000) to radians
    
    Args:
        joint_positions: Raw joint positions from HDF5 data (26 dims)
        
    Returns:
        Joint positions padded and scaled for URDF (39 dims: 27 robot + 12 hand)
    """
    if len(joint_positions) != 26:
        raise ValueError(f"Expected 26 DoF input, got {len(joint_positions)} DoF")
    
    # Extract upper body joints (first 14 dimensions)
    upper_body_joints = joint_positions[:14]
    
    # Pad with 13 zeros for leg joints to make it 27 robot joints total
    # The leg joints will be at zero position (neutral pose)
    leg_joints_zeros = np.zeros(13)
    robot_joints = np.concatenate([leg_joints_zeros, upper_body_joints])
    
    # Extract hand joints (last 12 dimensions)
    hand_joint_values = joint_positions[14:26]
    
    # Scale hand joints with different ranges for thumb joints
    hand_joint_values_scaled = np.zeros_like(hand_joint_values)
    
    # Left hand thumb joints (indices 0, 1): thumb_yaw, thumb_pitch
    # Right hand thumb joints (indices 6, 7): thumb_yaw, thumb_pitch
    thumb_yaw_indices = [0, 6]  # Left and right thumb_yaw
    thumb_pitch_indices = [1, 7]  # Left and right thumb_pitch
    
    # Scale thumb_yaw: (0, 1000) -> (1.3, -0.1)
    for idx in thumb_yaw_indices:
        hand_joint_values_scaled[idx] = 1.3 - (hand_joint_values[idx] * 1.4 / 1000.0)
    
    # Scale thumb_pitch: (0, 1000) -> (0.6, -0.1)
    for idx in thumb_pitch_indices:
        hand_joint_values_scaled[idx] = 0.6 - (hand_joint_values[idx] * 0.7 / 1000.0)
    
    # Scale other finger joints: (0, 1000) -> (1.7, 0)
    # Left hand: index, middle, ring, pinky (indices 2, 3, 4, 5)
    # Right hand: index, middle, ring, pinky (indices 8, 9, 10, 11)
    other_indices = [2, 3, 4, 5, 8, 9, 10, 11]
    for idx in other_indices:
        hand_joint_values_scaled[idx] = 1.7 - (hand_joint_values[idx] * 1.7 / 1000.0)
    
    # Combine robot joints with scaled hand joints
    all_joints = np.concatenate([robot_joints, hand_joint_values_scaled])
    
    return all_joints


def extract_hand_joints_for_urdf_51dof(joint_positions: np.ndarray) -> np.ndarray:
    """Extract and map hand joints for 51 DoF datasets to URDF joint order.
    
    The action array has 51 dimensions: 27 robot DoF + 24 hand dimensions.
    Only specific indices contain actual hand joint values:
    - Left hand: 27, 29, 31, 33, 35, 36 (index, little, middle, ring, thumb_yaw, thumb_pitch)
    - Right hand: 39, 41, 43, 45, 47, 48 (index, little, middle, ring, thumb_yaw, thumb_pitch)

    The URDF expects hand joint values in the range 0-1.7 radians, but the encoder values are in the range 0-1000.
    We need to scale the encoder values to the range 1.7-0 radians.

    The URDF joint order is:
    - Left hand: thumb_yaw, thumb_pitch, index, middle, ring, pinky
    - Right hand: thumb_yaw, thumb_pitch, index, middle, ring, pinky
    
    Args:
        joint_positions: Raw joint positions from HDF5 data (51 dims)
        
    Returns:
        Joint positions with hand joints mapped to URDF order (27 robot + 12 hand = 39 dims)
    """
    # Robot joints (first 27 dimensions)
    robot_joints = joint_positions[:27]
    
    # Extract actual hand joint values from specific indices
    hand_joint_indices = [35, 36, 27, 31, 33, 29, 47, 48, 39, 41, 43, 45]
    hand_joint_values = joint_positions[hand_joint_indices]
    
    # Scale hand joints with different ranges for thumb joints
    hand_joint_values_scaled = np.zeros_like(hand_joint_values)
    
    # Thumb joints (indices 0, 1, 6, 7): thumb_yaw, thumb_pitch for both hands
    thumb_yaw_indices = [0, 6]  # Left and right thumb_yaw
    thumb_pitch_indices = [1, 7]  # Left and right thumb_pitch
    
    # Scale thumb_yaw: (0, 1000) -> (1.3, -0.1)
    for idx in thumb_yaw_indices:
        hand_joint_values_scaled[idx] = 1.3 - (hand_joint_values[idx] * 1.4 / 1000.0)
    
    # Scale thumb_pitch: (0, 1000) -> (0.6, -0.1)
    for idx in thumb_pitch_indices:
        hand_joint_values_scaled[idx] = 0.6 - (hand_joint_values[idx] * 0.7 / 1000.0)
    
    # Scale other finger joints: (0, 1000) -> (1.7, 0)
    other_indices = [2, 3, 4, 5, 8, 9, 10, 11]  # index, middle, ring, pinky for both hands
    for idx in other_indices:
        hand_joint_values_scaled[idx] = 1.7 - (hand_joint_values[idx] * 1.7 / 1000.0)
    
    # Combine robot joints with scaled hand joints
    all_joints = np.concatenate([robot_joints, hand_joint_values_scaled])
    
    return all_joints


def extract_hand_joints_for_urdf(joint_positions: np.ndarray) -> np.ndarray:
    """Auto-detect dataset type and extract hand joints for URDF.
    
    Args:
        joint_positions: Raw joint positions from HDF5 data
        
    Returns:
        Joint positions mapped to URDF order (39 dims: 27 robot + 12 hand)
    """
    if len(joint_positions) == 26:
        return extract_hand_joints_for_urdf_26dof(joint_positions)
    elif len(joint_positions) == 51:
        return extract_hand_joints_for_urdf_51dof(joint_positions)
    else:
        raise ValueError(f"Unsupported action space dimension: {len(joint_positions)}. Expected 26 or 51 DoF.")


def main(
    hdf5_path: str = "../processed_data/circular.hdf5",
    urdf_path: str = "../assets/h1_2/h1_2.urdf",
    fps: float = 30.0,
    start_frame: int = 0,
    load_meshes: bool = True,
    load_collision_meshes: bool = False,
) -> None:
    """Replay HDF5 data with viser visualization.
    
    Args:
        hdf5_path: Path to HDF5 file containing robot data
        urdf_path: Path to robot URDF file
        fps: Frames per second for playback
        start_frame: Frame to start playback from
        load_meshes: Whether to load visual meshes
        load_collision_meshes: Whether to load collision meshes
    """
    
    # Convert relative paths to absolute
    script_dir = Path(__file__).parent
    if not os.path.isabs(hdf5_path):
        hdf5_path = str(script_dir / hdf5_path)
    if not os.path.isabs(urdf_path):
        urdf_path = str(script_dir / urdf_path)
    
    # Load HDF5 data
    data = load_hdf5_data(hdf5_path)
    
    # Start viser server
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    
    # Load URDF
    print(f"Loading URDF: {urdf_path}")
    urdf = URDF.load(
        urdf_path,
        load_meshes=load_meshes,
        build_scene_graph=load_meshes,
        load_collision_meshes=load_collision_meshes,
        build_collision_scene_graph=load_collision_meshes,
    )
    
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf,
        load_meshes=load_meshes,
        load_collision_meshes=load_collision_meshes,
        collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
    )
    
    # Create grid
    trimesh_scene = viser_urdf._urdf.scene or viser_urdf._urdf.collision_scene
    server.scene.add_grid(
        "/grid",
        width=10,
        height=10,
        position=(
            0.0,
            0.0,
            trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0,
        ),
    )
    
    # Create GUI controls
    with server.gui.add_folder("Playback Control"):
        play_button = server.gui.add_button("Play/Pause")
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=data['num_frames'] - 1,
            step=1,
            initial_value=start_frame,
        )
        speed_slider = server.gui.add_slider(
            "Speed",
            min=0.1,
            max=5.0,
            step=0.1,
            initial_value=1.0,
        )
        fps_display = server.gui.add_number(
            "FPS",
            initial_value=fps,
            disabled=True,
        )
        frame_info = server.gui.add_text(
            "Frame Info",
            initial_value=f"Frame {start_frame}/{data['num_frames'] - 1}",
            disabled=True,
        )
    
    with server.gui.add_folder("Visualization"):
        show_meshes_cb = server.gui.add_checkbox(
            "Show meshes",
            initial_value=viser_urdf.show_visual,
        )
        show_collision_meshes_cb = server.gui.add_checkbox(
            "Show collision meshes",
            initial_value=viser_urdf.show_collision,
        )
        use_actions = server.gui.add_checkbox(
            "Use actions (vs qpos)",
            initial_value=True,
        )
    
    with server.gui.add_folder("Camera Views", expand_by_default=False):
        camera_checkboxes = {}
        
        for topic in data['camera_topics']:
            display_name = topic.replace('_', ' ').title()
            camera_checkboxes[topic] = server.gui.add_checkbox(f"Show {display_name}", initial_value=True)
    
    # Visibility callbacks
    @show_meshes_cb.on_update
    def _(_):
        viser_urdf.show_visual = show_meshes_cb.value
    
    @show_collision_meshes_cb.on_update
    def _(_):
        viser_urdf.show_collision = show_collision_meshes_cb.value
    
    # Playback state
    is_playing = False
    current_frame = start_frame
    
    @play_button.on_click
    def _(_):
        nonlocal is_playing
        is_playing = not is_playing
    
    @frame_slider.on_update
    def _(_):
        nonlocal current_frame
        current_frame = int(frame_slider.value)
    
    # Hide collision meshes checkbox if not loaded
    show_collision_meshes_cb.visible = load_collision_meshes
    
    # Set initial configuration
    initial_joints = data['actions'][start_frame] if use_actions.value else data['qpos'][start_frame]
    initial_joints = extract_hand_joints_for_urdf(initial_joints)
    viser_urdf.update_cfg(initial_joints[:viser_urdf._urdf.num_actuated_joints])
    
    # Camera handles are now managed dynamically per topic
    
    print(f"\n{'='*80}")
    print("Data Replay Started!")
    print(f"{'='*80}")
    print(f"Total frames: {data['num_frames']}")
    print(f"Number of joints: {data['num_joints']}")
    print(f"FPS: {fps}")
    
    # Detect dataset type and print appropriate info
    if data['num_joints'] == 26:
        print(f"Dataset type: 26 DoF (14 upper body + 12 hand joints)")
        print(f"- Upper body joints (0-13) used directly")
        print(f"- Leg joints padded with zeros (neutral pose)")
        print(f"- Hand joints (14-25) scaled from encoder values to radians")
    elif data['num_joints'] == 51:
        print(f"Dataset type: 51 DoF (27 robot + 24 hand dimensions)")
        print(f"- Robot joints (0-26) used directly")
        print(f"- Hand joints extracted from specific indices and scaled")
    else:
        print(f"Dataset type: {data['num_joints']} DoF (auto-detection)")
    
    print(f"Camera topics and formats:")
    for topic in data['camera_topics']:
        print(f"  - {topic}: {data['image_formats'][topic]}")
    print(f"Camera rotation angles (modify global variables at top of file):")
    print(f"  - Ego-style cameras: Pitch={EGO_PITCH}°, Yaw={EGO_YAW}°, Roll={EGO_ROLL}°")
    print(f"  - Birdview-style cameras: Pitch={BIRDVIEW_PITCH}°, Yaw={BIRDVIEW_YAW}°, Roll={BIRDVIEW_ROLL}°")
    print(f"Viser server running at: http://localhost:8080")
    print(f"{'='*80}\n")
    print("Controls:")
    print("  - Click 'Play/Pause' to start/stop playback")
    print("  - Use 'Frame' slider to jump to specific frame")
    print("  - Adjust 'Speed' to change playback speed")
    print("  - Toggle visualization options in the GUI")
    print("  - Modify global variables at top of file to change camera rotations")
    print(f"{'='*80}\n")
    
    # Main replay loop
    last_update_time = time.time()
    frame_time = 1.0 / fps
    
    while True:
        current_time = time.time()
        elapsed = current_time - last_update_time
        
        # Update frame based on playback state
        if is_playing and elapsed >= (frame_time / speed_slider.value):
            current_frame += 1
            if current_frame >= data['num_frames']:
                current_frame = 0  # Loop back to start
            
            frame_slider.value = current_frame
            last_update_time = current_time
        else:
            current_frame = int(frame_slider.value)
        
        # Update robot configuration
        if use_actions.value:
            joint_positions = data['actions'][current_frame]
        else:
            joint_positions = data['qpos'][current_frame]
        
        # Extract and map hand joints for URDF
        joint_positions = extract_hand_joints_for_urdf(joint_positions)
        
        # Update only the actuated joints (first N joints)
        viser_urdf.update_cfg(joint_positions[:viser_urdf._urdf.num_actuated_joints])
        
        # Update frame info
        frame_info.value = f"Frame {current_frame}/{data['num_frames'] - 1}"
        
        # Update camera images
        try:
            # Handle all camera topics generically
            for topic in data['camera_topics']:
                if camera_checkboxes[topic].value:
                    # Get image data based on format
                    if data['image_formats'][topic] == "array":
                        # Direct array format: (N, H, W, 3)
                        img_array = data['camera_data'][topic][current_frame]
                        # Convert to uint8 if needed
                        if img_array.dtype != np.uint8:
                            img_array = (img_array * 255).astype(np.uint8)
                        # Convert BGR to RGB (swap channels 0 and 2)
                        img_array = img_array[:, :, [2, 1, 0]]
                    else:
                        # JPEG encoded format: decode first
                        img_data = data['camera_data'][topic][current_frame]
                        img = decode_jpeg_image(img_data)
                        img_array = np.array(img)
                    
                    # Apply transformations
                    img_array = np.flipud(img_array)  # Flip vertically
                    if topic in ['ego_cam', 'cam_high']:  # Apply different rotations based on topic
                        img_array = np.rot90(img_array, k=1)  # Counterclockwise
                    else:
                        img_array = np.rot90(img_array, k=-1)  # Clockwise
                    
                    # Convert Euler angles to quaternion for 3D rotation
                    if topic in ['ego_cam', 'cam_high']:
                        w, x, y, z = euler_to_quaternion(EGO_PITCH, EGO_YAW, EGO_ROLL)
                        pos_x, pos_y, pos_z = 1.0, 0.0, -0.4
                        render_width = 0.4
                    elif topic == 'cam_left_wrist':
                        w, x, y, z = euler_to_quaternion(EGO_PITCH, EGO_YAW, EGO_ROLL)
                        pos_x, pos_y, pos_z = 0.5, -0.5, -0.4  # Left side of ego camera
                        render_width = 0.3
                    elif topic == 'cam_right_wrist':
                        w, x, y, z = euler_to_quaternion(EGO_PITCH, EGO_YAW, EGO_ROLL)
                        pos_x, pos_y, pos_z = 0.5, 0.5, -0.4   # Right side of ego camera
                        render_width = 0.3
                    else:
                        w, x, y, z = euler_to_quaternion(BIRDVIEW_PITCH, BIRDVIEW_YAW, BIRDVIEW_ROLL)
                        pos_x, pos_y, pos_z = 1.0, 0.0, 0.8
                        render_width = 0.3
                    
                    # Create or update image handle
                    handle_name = f"/{topic}"
                    if not hasattr(server.scene, f"{topic}_handle"):
                        setattr(server.scene, f"{topic}_handle", None)
                    
                    handle = getattr(server.scene, f"{topic}_handle")
                    
                    if handle is None:
                        handle = server.scene.add_image(
                            handle_name,
                            image=img_array,
                            render_width=render_width,
                            render_height=render_width * img_array.shape[0] / img_array.shape[1],
                            position=(pos_x, pos_y, pos_z),
                            wxyz=(w, x, y, z),
                        )
                        setattr(server.scene, f"{topic}_handle", handle)
                    else:
                        handle.image = img_array
                        handle.position = (pos_x, pos_y, pos_z)
                        handle.wxyz = (w, x, y, z)
                else:
                    # Remove handle if checkbox is unchecked
                    handle_name = f"/{topic}"
                    if hasattr(server.scene, f"{topic}_handle"):
                        handle = getattr(server.scene, f"{topic}_handle")
                        if handle is not None:
                            handle.remove()
                            setattr(server.scene, f"{topic}_handle", None)
                
        except Exception as e:
            print(f"Error decoding images at frame {current_frame}: {e}")
        
        # Sleep to maintain frame rate
        time.sleep(0.01)  # Small sleep to prevent busy waiting


if __name__ == "__main__":
    tyro.cli(main)

