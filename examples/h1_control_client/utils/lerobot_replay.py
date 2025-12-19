#!/usr/bin/env python3
"""LeRobot Dataset Replay with Viser

Replay robot motion data from LeRobot format (parquet) with frame scrubbing.
Supports sending positions to robot including hand control (26 DOF).

Usage:
    # Visualize only (no robot)
    python utils/lerobot_replay.py --dataset-path h1_data_lerobot/h1_place_kettle

    # With robot connection for scrubbing to positions
    python utils/lerobot_replay.py --dataset-path h1_data_lerobot/h1_place_kettle --robot-host localhost --robot-port 5007
"""

import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
import viser
from PIL import Image
from viser.extras import ViserUrdf
from yourdfpy import URDF


def load_lerobot_episode(dataset_path: str, episode_idx: int = 0) -> dict:
    """Load a single episode from LeRobot dataset.
    
    Args:
        dataset_path: Path to LeRobot dataset directory
        episode_idx: Episode index to load
        
    Returns:
        Dictionary with actions, qpos, images, num_frames
    """
    dataset_path = Path(dataset_path)
    
    # Find the parquet file
    parquet_path = dataset_path / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")
    
    print(f"Loading episode {episode_idx} from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Extract actions and qpos
    actions = np.stack(df['action'].values)
    qpos = np.stack(df['qpos'].values)
    
    # Load images
    images = []
    image_dir = dataset_path / "images" / "ego_cam" / f"episode_{episode_idx:06d}"
    if image_dir.exists():
        for i in range(len(df)):
            img_path = image_dir / f"frame_{i:06d}.png"
            if img_path.exists():
                images.append(np.array(Image.open(img_path)))
            else:
                images.append(None)
    
    print(f"Loaded {len(actions)} frames, action_dim={actions.shape[1]}, qpos_dim={qpos.shape[1]}")
    
    return {
        'actions': actions,
        'qpos': qpos,
        'images': images,
        'num_frames': len(actions),
        'action_dim': actions.shape[1],
    }


def extract_joints_for_urdf(joint_positions: np.ndarray) -> np.ndarray:
    """Convert 14 or 26 DOF joint positions to URDF format (39 DOF).
    
    Args:
        joint_positions: (14,) or (26,) array
        
    Returns:
        (39,) array for URDF visualization (27 robot + 12 hand)
    """
    if len(joint_positions) == 14:
        # 14 DOF: arm only, pad with zeros
        upper_body = joint_positions[:14]
        leg_joints = np.zeros(13)
        robot_joints = np.concatenate([leg_joints, upper_body])
        hand_joints = np.zeros(12)
    elif len(joint_positions) == 26:
        # 26 DOF: arm (14) + hand (12)
        upper_body = joint_positions[:14]
        leg_joints = np.zeros(13)
        robot_joints = np.concatenate([leg_joints, upper_body])
        
        # Scale hand joints from encoder values (0-1000) to radians
        hand_raw = joint_positions[14:26]
        hand_joints = np.zeros(12)
        
        # Thumb joints: (0, 1000) -> (1.3, -0.1) for yaw, (0.6, -0.1) for pitch
        thumb_yaw_indices = [0, 6]
        thumb_pitch_indices = [1, 7]
        for idx in thumb_yaw_indices:
            hand_joints[idx] = 1.3 - (hand_raw[idx] * 1.4 / 1000.0)
        for idx in thumb_pitch_indices:
            hand_joints[idx] = 0.6 - (hand_raw[idx] * 0.7 / 1000.0)
        
        # Other fingers: (0, 1000) -> (1.7, 0)
        other_indices = [2, 3, 4, 5, 8, 9, 10, 11]
        for idx in other_indices:
            hand_joints[idx] = 1.7 - (hand_raw[idx] * 1.7 / 1000.0)
    else:
        raise ValueError(f"Unsupported joint dimension: {len(joint_positions)}")
    
    return np.concatenate([robot_joints, hand_joints])


async def send_to_robot(host: str, port: int, action: np.ndarray) -> bool:
    """Send joint positions to robot via WebSocket.
    
    Args:
        host: Robot server host
        port: Robot server port
        action: (26,) array of joint positions
        
    Returns:
        True if successful
    """
    try:
        import websockets
        uri = f"ws://{host}:{port}"
        async with websockets.connect(uri) as websocket:
            # Send as replay_teleop command with single frame
            command = {
                "cmd": "replay_teleop",
                "actions": [action.tolist()]  # Single frame
            }
            await websocket.send(json.dumps(command))
            response = await websocket.recv()
            result = json.loads(response)
            return result.get("status") == "success"
    except Exception as e:
        print(f"Error sending to robot: {e}")
        return False


def main(
    dataset_path: str = "h1_data_lerobot/h1_place_kettle",
    episode_idx: int = 0,
    urdf_path: str = "assets/h1_2/h1_2.urdf",
    robot_host: str | None = None,
    robot_port: int = 5007,
    fps: float = 30.0,
) -> None:
    """Replay LeRobot dataset with viser visualization and optional robot control.
    
    Args:
        dataset_path: Path to LeRobot dataset directory
        episode_idx: Episode index to load
        urdf_path: Path to robot URDF file
        robot_host: Host for robot WebSocket server (None = visualization only)
        robot_port: Port for robot WebSocket server
        fps: Playback frame rate
    """
    # Resolve paths
    script_dir = Path(__file__).parent.parent
    if not os.path.isabs(dataset_path):
        dataset_path = str(script_dir / dataset_path)
    if not os.path.isabs(urdf_path):
        urdf_path = str(script_dir / urdf_path)
    
    # Load episode
    data = load_lerobot_episode(dataset_path, episode_idx)
    
    # Start viser server
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    
    # Load URDF
    print(f"Loading URDF: {urdf_path}")
    urdf = URDF.load(urdf_path, load_meshes=True, build_scene_graph=True)
    viser_urdf = ViserUrdf(server, urdf_or_path=urdf, load_meshes=True)
    
    # Add grid
    server.scene.add_grid("/grid", width=10, height=10, position=(0.0, 0.0, 0.0))
    
    # Create GUI controls
    with server.gui.add_folder("Playback Control"):
        play_button = server.gui.add_button("Play/Pause")
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=data['num_frames'] - 1,
            step=1,
            initial_value=0,
        )
        speed_slider = server.gui.add_slider(
            "Speed",
            min=0.1,
            max=3.0,
            step=0.1,
            initial_value=1.0,
        )
        frame_info = server.gui.add_text(
            "Info",
            initial_value=f"Frame 0/{data['num_frames'] - 1} | {data['action_dim']} DOF",
            disabled=True,
        )
    
    with server.gui.add_folder("Robot Control"):
        if robot_host:
            send_button = server.gui.add_button("Send Current Frame to Robot")
            robot_status = server.gui.add_text(
                "Robot Status",
                initial_value=f"Connected: {robot_host}:{robot_port}",
                disabled=True,
            )
        else:
            server.gui.add_text(
                "Robot Status",
                initial_value="No robot connection (use --robot-host to enable)",
                disabled=True,
            )
    
    # Playback state
    is_playing = False
    current_frame = 0
    
    @play_button.on_click
    def _(_):
        nonlocal is_playing
        is_playing = not is_playing
    
    @frame_slider.on_update
    def _(_):
        nonlocal current_frame
        current_frame = int(frame_slider.value)
    
    if robot_host:
        @send_button.on_click
        def _(event):
            action = data['actions'][current_frame]
            robot_status.value = f"Sending frame {current_frame}..."
            
            # Run async send in background
            loop = asyncio.new_event_loop()
            success = loop.run_until_complete(send_to_robot(robot_host, robot_port, action))
            loop.close()
            
            if success:
                robot_status.value = f"Sent frame {current_frame} ({data['action_dim']} DOF)"
            else:
                robot_status.value = f"Failed to send frame {current_frame}"
    
    # Image handle
    image_handle = None
    
    print(f"\n{'='*60}")
    print("LeRobot Replay Started!")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_path}")
    print(f"Episode: {episode_idx}")
    print(f"Frames: {data['num_frames']}")
    print(f"Action DOF: {data['action_dim']}")
    print(f"Robot: {'Connected to ' + robot_host + ':' + str(robot_port) if robot_host else 'Visualization only'}")
    print(f"Viser server: http://localhost:8080")
    print(f"{'='*60}\n")
    
    # Main loop
    last_update = time.time()
    frame_time = 1.0 / fps
    
    while True:
        current_time = time.time()
        elapsed = current_time - last_update
        
        # Update frame if playing
        if is_playing and elapsed >= (frame_time / speed_slider.value):
            current_frame += 1
            if current_frame >= data['num_frames']:
                current_frame = 0
            frame_slider.value = current_frame
            last_update = current_time
        else:
            current_frame = int(frame_slider.value)
        
        # Update robot visualization
        qpos = data['qpos'][current_frame]
        urdf_joints = extract_joints_for_urdf(qpos)
        viser_urdf.update_cfg(urdf_joints[:viser_urdf._urdf.num_actuated_joints])
        
        # Update image display
        if data['images'] and data['images'][current_frame] is not None:
            img = data['images'][current_frame]
            img = np.flipud(img)
            img = np.rot90(img, k=1)
            
            if image_handle is None:
                image_handle = server.scene.add_image(
                    "/ego_cam",
                    image=img,
                    render_width=0.4,
                    render_height=0.4 * img.shape[0] / img.shape[1],
                    position=(1.0, 0.0, -0.4),
                )
            else:
                image_handle.image = img
        
        # Update frame info
        action = data['actions'][current_frame]
        if data['action_dim'] == 26:
            arm_range = f"[{action[:14].min():.2f}, {action[:14].max():.2f}]"
            hand_range = f"[{action[14:].min():.0f}, {action[14:].max():.0f}]"
            frame_info.value = f"Frame {current_frame}/{data['num_frames']-1} | Arm:{arm_range} Hand:{hand_range}"
        else:
            frame_info.value = f"Frame {current_frame}/{data['num_frames']-1} | {data['action_dim']} DOF"
        
        time.sleep(0.01)


if __name__ == "__main__":
    tyro.cli(main)



