#!/usr/bin/env python3
"""G1 HDF5 Data Replay with Viser

Replay robot motion data from HDF5 files frame by frame using viser visualization.

Features:
* Load G1 HDF5 data containing robot joint positions and camera images
* Visualize robot motion using viser URDF viewer with Dex3 hands
* Display locomotion state (mode_machine, IMU, leg positions)
* Display controller inputs (joysticks, buttons)
* Frame-by-frame playback with play/pause controls
* Adjustable playback speed

Usage:
    python g1_data_replay.py --hdf5_path /path/to/episode.hdf5
    python g1_data_replay.py  # Uses default test path
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import tyro
from yourdfpy import URDF

import viser
from viser.extras import ViserUrdf

from utils.data_replay import (
    load_hdf5_data,
    extract_joints_for_urdf_g1_28dof,
    decode_jpeg_image,
    euler_to_quaternion,
    format_loco_state,
    format_loco_action,
    get_urdf_path,
)


# Camera rotation settings
EGO_PITCH = -40.0
EGO_YAW = 0.0
EGO_ROLL = 0.0


def main(
    hdf5_path: str = "./g1_data_processed/cabinet_bottle/episode_02.hdf5",
    urdf_path: str | None = None,
    fps: float | None = None,
    start_frame: int = 0,
    load_meshes: bool = True,
    load_collision_meshes: bool = False,
    port: int = 8080,
) -> None:
    """Replay G1 HDF5 data with viser visualization.
    
    Args:
        hdf5_path: Path to HDF5 file containing robot data
        urdf_path: Path to robot URDF file (defaults to G1 URDF)
        fps: Frames per second for playback (defaults to file's fps)
        start_frame: Frame to start playback from
        load_meshes: Whether to load visual meshes
        load_collision_meshes: Whether to load collision meshes
        port: Port for viser server
    """
    
    # Convert relative paths to absolute
    script_dir = Path(__file__).parent
    
    if not os.path.isabs(hdf5_path):
        hdf5_path = str(script_dir / hdf5_path)
    
    if urdf_path is None:
        urdf_path = get_urdf_path()
    elif not os.path.isabs(urdf_path):
        urdf_path = str(script_dir / urdf_path)
    
    # Load HDF5 data
    data = load_hdf5_data(hdf5_path)
    
    # Use file's FPS if not specified
    if fps is None:
        fps = data['fps']
    
    # Start viser server
    server = viser.ViserServer(port=port)
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
    
    # Camera view controls
    with server.gui.add_folder("Camera Views", expand_by_default=False):
        camera_checkboxes = {}
        for topic in data['camera_topics']:
            display_name = topic.replace('_', ' ').title()
            camera_checkboxes[topic] = server.gui.add_checkbox(
                f"Show {display_name}", 
                initial_value=True
            )
    
    # Locomotion data display (if available)
    loco_folder = None
    loco_displays = {}
    
    if data['has_loco_data']:
        with server.gui.add_folder("Locomotion State", expand_by_default=True):
            loco_displays['mode'] = server.gui.add_text(
                "Mode",
                initial_value="--",
                disabled=True,
            )
            loco_displays['rpy'] = server.gui.add_text(
                "RPY (deg)",
                initial_value="R:-- P:-- Y:--",
                disabled=True,
            )
            loco_displays['accel'] = server.gui.add_text(
                "Accel (m/s^2)",
                initial_value="X:-- Y:-- Z:--",
                disabled=True,
            )
            loco_displays['gyro'] = server.gui.add_text(
                "Gyro (rad/s)",
                initial_value="X:-- Y:-- Z:--",
                disabled=True,
            )
            loco_displays['knees'] = server.gui.add_text(
                "Knee Angles",
                initial_value="L:-- R:-- Avg:--",
                disabled=True,
            )
        
        with server.gui.add_folder("Controller Input", expand_by_default=True):
            loco_displays['joysticks'] = server.gui.add_text(
                "Joysticks",
                initial_value="Lx:-- Ly:-- Rx:-- Ry:--",
                disabled=True,
            )
            loco_displays['buttons'] = server.gui.add_text(
                "Active Buttons",
                initial_value="--",
                disabled=True,
            )
    
    # Joint info display
    with server.gui.add_folder("Joint Info", expand_by_default=False):
        joint_displays = {}
        joint_displays['left_arm'] = server.gui.add_text(
            "Left Arm",
            initial_value="--",
            disabled=True,
        )
        joint_displays['right_arm'] = server.gui.add_text(
            "Right Arm",
            initial_value="--",
            disabled=True,
        )
        joint_displays['left_hand'] = server.gui.add_text(
            "Left Hand",
            initial_value="--",
            disabled=True,
        )
        joint_displays['right_hand'] = server.gui.add_text(
            "Right Hand",
            initial_value="--",
            disabled=True,
        )
    
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
    initial_loco = data['loco_state'][start_frame] if data['loco_state'] is not None else None
    urdf_joints = extract_joints_for_urdf_g1_28dof(initial_joints, initial_loco)
    viser_urdf.update_cfg(urdf_joints[:viser_urdf._urdf.num_actuated_joints])
    
    # Print startup info
    print(f"\n{'='*80}")
    print("G1 Data Replay Started!")
    print(f"{'='*80}")
    print(f"Total frames: {data['num_frames']}")
    print(f"Number of joints: {data['num_joints']} (28 DOF: 14 arm + 14 hand)")
    print(f"Robot: {data['robot_name']}")
    print(f"FPS: {fps}")
    print(f"Has locomotion data: {data['has_loco_data']}")
    print(f"Camera topics: {data['camera_topics']}")
    print(f"\nURDF: {urdf_path}")
    print(f"URDF actuated joints: {viser_urdf._urdf.num_actuated_joints}")
    print(f"\nViser server running at: http://localhost:{port}")
    print(f"{'='*80}")
    print("\nControls:")
    print("  - Click 'Play/Pause' to start/stop playback")
    print("  - Use 'Frame' slider to jump to specific frame")
    print("  - Adjust 'Speed' to change playback speed")
    print("  - Toggle visualization options in the GUI")
    print(f"{'='*80}\n")
    
    # Image handles storage
    image_handles = {}
    
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
        
        # Get joint positions
        if use_actions.value:
            joint_positions = data['actions'][current_frame]
        else:
            joint_positions = data['qpos'][current_frame]
        
        # Get locomotion state if available
        loco_state = None
        loco_action = None
        if data['loco_state'] is not None:
            loco_state = data['loco_state'][current_frame]
        if data['loco_action'] is not None:
            loco_action = data['loco_action'][current_frame]
        
        # Map joints to URDF order
        urdf_joints = extract_joints_for_urdf_g1_28dof(joint_positions, loco_state)
        
        # Update robot configuration
        viser_urdf.update_cfg(urdf_joints[:viser_urdf._urdf.num_actuated_joints])
        
        # Update frame info
        frame_info.value = f"Frame {current_frame}/{data['num_frames'] - 1}"
        
        # Update joint info displays
        joint_displays['left_arm'].value = f"{joint_positions[0:7].round(3)}"
        joint_displays['right_arm'].value = f"{joint_positions[7:14].round(3)}"
        joint_displays['left_hand'].value = f"{joint_positions[14:21].round(3)}"
        joint_displays['right_hand'].value = f"{joint_positions[21:28].round(3)}"
        
        # Update locomotion displays
        if data['has_loco_data'] and loco_state is not None:
            loco_info = format_loco_state(loco_state)
            loco_displays['mode'].value = loco_info.get('mode_machine', '--')
            
            rpy = loco_info.get('rpy', {})
            loco_displays['rpy'].value = (
                f"R:{np.degrees(rpy.get('roll', 0)):.1f} "
                f"P:{np.degrees(rpy.get('pitch', 0)):.1f} "
                f"Y:{np.degrees(rpy.get('yaw', 0)):.1f}"
            )
            
            accel = loco_info.get('accelerometer', {})
            loco_displays['accel'].value = (
                f"X:{accel.get('x', 0):.2f} "
                f"Y:{accel.get('y', 0):.2f} "
                f"Z:{accel.get('z', 0):.2f}"
            )
            
            gyro = loco_info.get('gyroscope', {})
            loco_displays['gyro'].value = (
                f"X:{gyro.get('x', 0):.3f} "
                f"Y:{gyro.get('y', 0):.3f} "
                f"Z:{gyro.get('z', 0):.3f}"
            )
            
            legs = loco_info.get('leg_joints', {})
            loco_displays['knees'].value = (
                f"L:{legs.get('left_knee', 0):.3f} "
                f"R:{legs.get('right_knee', 0):.3f} "
                f"Avg:{legs.get('avg_knee', 0):.3f}"
            )
        
        if data['has_loco_data'] and loco_action is not None:
            action_info = format_loco_action(loco_action)
            joy = action_info.get('joysticks', {})
            loco_displays['joysticks'].value = (
                f"Lx:{joy.get('Lx', 0):.2f} "
                f"Ly:{joy.get('Ly', 0):.2f} "
                f"Rx:{joy.get('Rx', 0):.2f} "
                f"Ry:{joy.get('Ry', 0):.2f}"
            )
            
            active = action_info.get('active_buttons', [])
            loco_displays['buttons'].value = ', '.join(active) if active else 'None'
        
        # Update camera images
        try:
            for topic in data['camera_topics']:
                if camera_checkboxes[topic].value:
                    # Get image data based on format
                    if data['image_formats'][topic] == "array":
                        img_array = data['camera_data'][topic][current_frame]
                        if img_array.dtype != np.uint8:
                            img_array = (img_array * 255).astype(np.uint8)
                        # Convert BGR to RGB if needed
                        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                            img_array = img_array[:, :, [2, 1, 0]]
                    else:
                        img_data = data['camera_data'][topic][current_frame]
                        img = decode_jpeg_image(img_data)
                        img_array = np.array(img)
                    
                    # Apply transformations
                    img_array = np.flipud(img_array)
                    img_array = np.rot90(img_array, k=1)
                    
                    # Camera positioning
                    w, x, y, z = euler_to_quaternion(EGO_PITCH, EGO_YAW, EGO_ROLL)
                    pos_x, pos_y, pos_z = 1.0, 0.0, -0.4
                    render_width = 0.4
                    
                    # Create or update image handle
                    handle_name = f"/{topic}"
                    
                    if topic not in image_handles or image_handles[topic] is None:
                        image_handles[topic] = server.scene.add_image(
                            handle_name,
                            image=img_array,
                            render_width=render_width,
                            render_height=render_width * img_array.shape[0] / img_array.shape[1],
                            position=(pos_x, pos_y, pos_z),
                            wxyz=(w, x, y, z),
                        )
                    else:
                        image_handles[topic].image = img_array
                else:
                    # Remove handle if checkbox is unchecked
                    if topic in image_handles and image_handles[topic] is not None:
                        image_handles[topic].remove()
                        image_handles[topic] = None
                        
        except Exception as e:
            print(f"Error updating images at frame {current_frame}: {e}")
        
        # Sleep to maintain responsiveness
        time.sleep(0.01)


if __name__ == "__main__":
    tyro.cli(main)
