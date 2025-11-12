"""H1 Policy Inference Visualization Client

Loads observations from H1 dataset and sends them to the policy server for inference,
then visualizes the predicted action chunks using viser.

Features:
* Load observations (images + state) from HDF5 dataset
* Send observations to policy server
* Receive and visualize predicted action chunks
* Interactive frame selection and playback controls
* Display robot motion using viser URDF viewer
* Display camera images alongside robot visualization

Usage:
1. Start policy server:
   uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_h1_finetune --policy.dir=checkpoints/pi05_h1_finetune/pi05_h1_H50/1999
   
2. Run this client:
   python h1_policy_viz_client.py --hdf5-path processed_data/circular.hdf5
"""

from __future__ import annotations

import dataclasses
import io
import os
import time
from pathlib import Path
import asyncio
import json
import websockets

import cv2
import einops
import h5py
import numpy as np
import tyro
from openpi_client import websocket_client_policy as _websocket_client_policy
from PIL import Image
from tqdm import tqdm
from yourdfpy import URDF

import viser
from viser.extras import ViserUrdf

# Global rotation settings for camera orientations
EGO_PITCH = -40.0    # Ego camera pitch angle (degrees)
EGO_YAW = 0.0        # Ego camera yaw angle (degrees) 
EGO_ROLL = 0.0       # Ego camera roll angle (degrees)

BIRDVIEW_PITCH = 20.0    # Birdview camera pitch angle (degrees)
BIRDVIEW_YAW = 0.0       # Birdview camera yaw angle (degrees)
BIRDVIEW_ROLL = 0.0      # Birdview camera roll angle (degrees)


@dataclasses.dataclass
class Args:
    """Command line arguments."""
    
    # Data paths
    hdf5_path: str = "processed_data/circular.hdf5"
    urdf_path: str = "assets/h1_2/h1_2.urdf"
    
    # Policy server connection
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None
    
    # Task/prompt
    prompt: str = "move arm circularly"
    
    # Visualization settings
    fps: float = 30.0
    start_frame: int = 0
    load_meshes: bool = True
    load_collision_meshes: bool = False
    
    # Robot execution settings
    robot_execution: bool = False
    """Enable robot execution (connects to h1_remote_client in listen mode)"""
    
    robot_host: str = "localhost"
    """Robot client host (via SSH reverse tunnel)"""
    
    robot_port: int = 5007
    """Robot client port"""


def load_hdf5_data(hdf5_path: str) -> dict:
    """Load data from HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
        
    Returns:
        Dictionary containing observations, actions, and metadata
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


def decode_jpeg_image(img_data: bytes) -> np.ndarray:
    """Decode JPEG image data to numpy array.
    
    Args:
        img_data: JPEG encoded image bytes
        
    Returns:
        Image as numpy array (H, W, 3)
    """
    pil_img = Image.open(io.BytesIO(img_data))
    return np.array(pil_img)


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


def get_observation_at_frame(data: dict, frame_idx: int, prompt: str, target_size: tuple = (224, 224)) -> dict:
    """Get observation at specific frame in the format expected by the policy.
    
    Args:
        data: Loaded HDF5 data
        frame_idx: Frame index to get observation from
        prompt: Task prompt/instruction
        target_size: Target image size (height, width)
        
    Returns:
        Observation dictionary compatible with H1 policy
    """
    # Get state (first 14 dimensions of qpos)
    state = data['qpos'][frame_idx][:14].astype(np.float32)
    
    # Map camera names from dataset to policy expected names
    camera_mapping = {
        'ego_cam': 'cam_head',
        'cam_left_wrist': 'cam_left_wrist',
        'cam_right_wrist': 'cam_right_wrist',
    }
    
    images = {}
    for dataset_name, policy_name in camera_mapping.items():
        if dataset_name in data['camera_data']:
            # Get image
            if data['image_formats'][dataset_name] == 'array':
                img = data['camera_data'][dataset_name][frame_idx]
            else:  # JPEG format
                img = decode_jpeg_image(data['camera_data'][dataset_name][frame_idx])
            
            # Resize to target size
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            
            # Convert to (C, H, W) uint8 format expected by policy
            img = einops.rearrange(img, 'h w c -> c h w')
            images[policy_name] = img.astype(np.uint8)
    
    return {
        "state": state,
        "images": images,
        "prompt": prompt,
    }


async def send_robot_command(host: str, port: int, command: dict) -> dict:
    """Send command to robot client and get response.
    
    Args:
        host: Robot client host
        port: Robot client port
        command: Command dictionary to send
        
    Returns:
        Response dictionary from robot
    """
    uri = f"ws://{host}:{port}"
    try:
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps(command))
            response = await ws.recv()
            return json.loads(response)
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def get_live_observation_from_robot(host: str, port: int, prompt: str) -> dict:
    """Get live observation from robot (camera feeds + state).
    
    Args:
        host: Robot client host
        port: Robot client port
        prompt: Task prompt for the policy
        
    Returns:
        Observation dictionary formatted for policy inference:
        {
            "state": np.ndarray (14,),
            "images": {
                "cam_head": np.ndarray (224, 224, 3),
                "cam_left_wrist": np.ndarray (224, 224, 3),
                "cam_right_wrist": np.ndarray (224, 224, 3)
            },
            "prompt": str
        }
    """
    import base64
    from io import BytesIO
    
    # Request observation from robot
    response = await send_robot_command(host, port, {"command": "get_observation"})
    
    if response["status"] != "success":
        raise RuntimeError(f"Failed to get observation: {response.get('message', 'Unknown error')}")
    
    # Decode base64 images
    def decode_base64_img(b64_str: str) -> np.ndarray:
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(BytesIO(img_bytes))
        img_array = np.array(img)
        # Resize to 224x224 if needed
        if img_array.shape[:2] != (224, 224):
            img = img.resize((224, 224), Image.LANCZOS)
            img_array = np.array(img)
        return img_array
    
    # Extract and format observation (using policy's expected camera names)
    state = np.array(response["state"], dtype=np.float32)
    images = {
        "cam_head": decode_base64_img(response["images"]["cam_head"]),
        "cam_left_wrist": decode_base64_img(response["images"]["cam_left_wrist"]),
        "cam_right_wrist": decode_base64_img(response["images"]["cam_right_wrist"]),
    }
    
    return {
        "state": state,
        "images": images,
        "prompt": prompt
    }


def extract_hand_joints_for_urdf(joint_positions: np.ndarray) -> np.ndarray:
    """Convert 14-dim actions to URDF format (27 robot joints + 12 hand joints).
    
    For visualization, we pad with zeros since we only have upper body control.
    
    Args:
        joint_positions: 14-dim joint positions
        
    Returns:
        39-dim joint positions for URDF (27 robot + 12 hand)
    """
    if len(joint_positions) == 14:
        # Pad upper body with leg joints (13 zeros) to make 27 total
        leg_joints_zeros = np.zeros(13)
        robot_joints = np.concatenate([leg_joints_zeros, joint_positions])
        # Add hand joints (12 zeros for neutral hand pose)
        hand_joints_zeros = np.zeros(12)
        all_joints = np.concatenate([robot_joints, hand_joints_zeros])
        return all_joints
    else:
        # If actions are already in a different format, handle accordingly
        # This is a fallback for compatibility
        return joint_positions


def main(args: Args) -> None:
    """Run H1 policy inference visualization client."""
    
    # Helper to run async functions from sync callbacks
    def run_async(coro):
        """Run async coroutine in a new thread with its own event loop."""
        import threading
        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(coro)
            finally:
                loop.close()
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
    
    # Convert relative paths to absolute
    script_dir = Path(__file__).parent
    if not os.path.isabs(args.hdf5_path):
        hdf5_path = str(script_dir / args.hdf5_path)
    else:
        hdf5_path = args.hdf5_path
        
    if not os.path.isabs(args.urdf_path):
        urdf_path = str(script_dir / args.urdf_path)
    else:
        urdf_path = args.urdf_path
    
    # Load HDF5 data
    data = load_hdf5_data(hdf5_path)
    
    # Connect to policy server
    print(f"\nConnecting to policy server at {args.host}:{args.port}")
    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    
    print(f"Server metadata: {policy.get_server_metadata()}")
    
    # Warm up the policy with a test observation
    print("Warming up policy...")
    test_obs = get_observation_at_frame(data, 0, args.prompt)
    policy.infer(test_obs)
    print("Policy ready!")
    
    # Start viser server
    print(f"\nStarting viser server...")
    server = viser.ViserServer(port=8080, host="0.0.0.0")
    server.scene.set_up_direction("+z")
    
    # Load URDF
    print(f"Loading URDF: {urdf_path}")
    urdf = URDF.load(
        urdf_path,
        load_meshes=args.load_meshes,
        build_scene_graph=args.load_meshes,
        load_collision_meshes=args.load_collision_meshes,
        build_collision_scene_graph=args.load_collision_meshes,
    )
    
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf,
        load_meshes=args.load_meshes,
        load_collision_meshes=args.load_collision_meshes,
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
    
    # State variables
    current_frame = args.start_frame
    predicted_actions = None
    action_chunk_horizon = 10  # Default action horizon
    last_live_observation = None  # Store last live observation for display
    is_playing_actions = False
    action_play_idx = 0
    show_ground_truth = True
    
    # Create GUI controls
    with server.gui.add_folder("Policy Inference", expand_by_default=True):
        frame_slider = server.gui.add_slider(
            "Select Frame",
            min=0,
            max=data['num_frames'] - 1,
            step=1,
            initial_value=args.start_frame,
        )
        
        infer_button = server.gui.add_button("🤖 Infer Action Chunk", icon=None)
        
        play_actions_button = server.gui.add_button("▶️ Play Action Chunk", disabled=True)
        
        inference_status = server.gui.add_text(
            "Status",
            initial_value="Ready - Select frame and click 'Infer Action Chunk'",
            disabled=True,
        )
    
    with server.gui.add_folder("Visualization", expand_by_default=True):
        show_gt_cb = server.gui.add_checkbox(
            "Show Ground Truth",
            initial_value=True,
        )
        
        show_predicted_cb = server.gui.add_checkbox(
            "Show Predicted Actions",
            initial_value=False,
            disabled=True,
        )
        
        action_index_slider = server.gui.add_slider(
            "Action Index",
            min=0,
            max=action_chunk_horizon - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        
        speed_slider = server.gui.add_slider(
            "Playback Speed",
            min=0.1,
            max=5.0,
            step=0.1,
            initial_value=1.0,
        )
    
    with server.gui.add_folder("Visualization Options", expand_by_default=False):
        show_meshes_cb = server.gui.add_checkbox(
            "Show meshes",
            initial_value=viser_urdf.show_visual,
        )
        show_collision_meshes_cb = server.gui.add_checkbox(
            "Show collision meshes",
            initial_value=viser_urdf.show_collision,
        )
    
    with server.gui.add_folder("Camera Views", expand_by_default=False):
        camera_checkboxes = {}
        for topic in data['camera_topics']:
            display_name = topic.replace('_', ' ').title()
            camera_checkboxes[topic] = server.gui.add_checkbox(
                f"Show {display_name}",
                initial_value=True
            )
    
    # === NEW: Robot Execution GUI ===
    if args.robot_execution:
        with server.gui.add_folder("🤖 Robot Execution", expand_by_default=True):
            robot_status = server.gui.add_text(
                "Status",
                initial_value="Ready - Connect robot client via SSH tunnel",
                disabled=True,
            )
            
            reset_robot_button = server.gui.add_button(
                "🔄 Reset Robot to Frame",
                hint="Move robot to selected frame's joint positions"
            )
            
            reset_zombie_button = server.gui.add_button(
                "Reset to Zombie Pose",
                hint="Move arms to fully extended forward position"
            )
            
            execute_robot_button = server.gui.add_button(
                "Execute on Robot",
                disabled=True,
                hint="Execute predicted actions on real robot"
            )
            
            replay_teleop_button = server.gui.add_button(
                "Replay Teleop Data",
                hint="Replay raw teleop joint commands from HDF5 (bypass policy)"
            )
            
            use_live_cam = server.gui.add_checkbox(
                "📷 Use Live Cameras",
                initial_value=False,
                hint="Use robot cameras for inference instead of dataset"
            )
            
            single_step_button = server.gui.add_button(
                "Single Step Inference",
                disabled=True,
                hint="Get observation → infer → execute once"
            )
            
            continuous_button = server.gui.add_button(
                "Start Continuous",
                disabled=True,
                hint="Toggle continuous inference loop"
            )
            
            server.gui.add_markdown("### 💾 Recording")
            
            label_name_input = server.gui.add_text(
                "Label Name",
                initial_value="default",
                hint="Task/label name for organizing episodes (e.g., 'pick_cube', 'stack_blocks')"
            )
            
            recording_status = server.gui.add_text(
                "Recording Status",
                initial_value="Not recording",
                disabled=True,
            )
            
            start_recording_button = server.gui.add_button(
                "Start Recording",
                color="green",
                hint="Start recording observations and actions to HDF5"
            )
            
            stop_recording_button = server.gui.add_button(
                "Stop Recording",
                disabled=True,
                color="red",
                hint="Stop and save episode"
            )
            
            server.gui.add_markdown("###  Safety")
            estop_button = server.gui.add_button(
                " EMERGENCY STOP",
                color="red"
            )
    
    # Visibility callbacks
    @show_meshes_cb.on_update
    def _(_):
        viser_urdf.show_visual = show_meshes_cb.value
    
    @show_collision_meshes_cb.on_update
    def _(_):
        viser_urdf.show_collision = show_collision_meshes_cb.value
    
    # Hide collision meshes checkbox if not loaded
    show_collision_meshes_cb.visible = args.load_collision_meshes
    
    # Frame slider callback
    @frame_slider.on_update
    def _(_):
        nonlocal current_frame, show_ground_truth
        current_frame = int(frame_slider.value)
        show_ground_truth = True
        show_gt_cb.value = True
        show_predicted_cb.value = False
        update_visualization()
    
    # Ground truth checkbox callback
    @show_gt_cb.on_update
    def _(_):
        nonlocal show_ground_truth
        show_ground_truth = show_gt_cb.value
        if show_ground_truth:
            show_predicted_cb.value = False
        update_visualization()
    
    # Predicted actions checkbox callback
    @show_predicted_cb.on_update
    def _(_):
        nonlocal show_ground_truth
        if show_predicted_cb.value:
            show_gt_cb.value = False
            show_ground_truth = False
        update_visualization()
    
    # Action index slider callback
    @action_index_slider.on_update
    def _(_):
        if predicted_actions is not None and show_predicted_cb.value:
            update_visualization()
    
    # Live camera checkbox callback (enable/disable live inference buttons)
    if args.robot_execution:
        @use_live_cam.on_update
        def _(_):
            single_step_button.disabled = not use_live_cam.value
            continuous_button.disabled = not use_live_cam.value
    
    # Inference button callback
    @infer_button.on_click
    def _(_):
        nonlocal predicted_actions, action_chunk_horizon, show_ground_truth
        
        async def do_inference():
            nonlocal predicted_actions, action_chunk_horizon, last_live_observation
            
            # Get observation - either from live robot or HDF5 dataset
            if args.robot_execution and use_live_cam.value:
                inference_status.value = "Getting live observation from robot..."
                try:
                    obs = await get_live_observation_from_robot(
                        args.robot_host, args.robot_port, args.prompt
                    )
                    last_live_observation = obs  # Store for camera display
                    # Update camera displays with live feeds
                    update_camera_displays()
                    inference_status.value = "Running inference on live observation..."
                except Exception as e:
                    inference_status.value = f"Error getting observation: {e}"
                    print(f"Error getting live observation: {e}")
                    return
            else:
                inference_status.value = f"Running inference for frame {current_frame}..."
                obs = get_observation_at_frame(data, current_frame, args.prompt)
            
            # Run inference
            start_time = time.time()
            result = policy.infer(obs)
            inference_time = time.time() - start_time
            
            # Extract predicted actions
            predicted_actions = result['actions']
            action_chunk_horizon = predicted_actions.shape[0]
            
            # Update UI
            source = "live robot" if (args.robot_execution and use_live_cam.value) else f"frame {current_frame}"
            inference_status.value = (
                f"✓ Got {action_chunk_horizon} actions from {source} ({inference_time*1000:.1f}ms) "
                f"→ Toggle 'Show Predicted Actions' to view"
            )
            
            # Enable playback controls
            play_actions_button.disabled = False
            show_predicted_cb.disabled = False
            action_index_slider.disabled = False
            action_index_slider.max = action_chunk_horizon - 1
            action_index_slider.value = 0
            
            # Don't automatically switch to predicted view - let user toggle manually
            # This prevents sudden robot movement
            
            print(f"\nInference complete ({source}):")
            print(f"  Action chunk shape: {predicted_actions.shape}")
            print(f"  Inference time: {inference_time*1000:.1f}ms")
            print(f"  → Toggle 'Show Predicted Actions' to view predictions")
            
            # Enable execute button if robot execution is enabled
            if args.robot_execution:
                execute_robot_button.disabled = False
        
        run_async(do_inference())
    
    # === NEW: Robot execution callbacks ===
    if args.robot_execution:
        # Reset robot button callback
        @reset_robot_button.on_click
        def _(event):
            current_frame = int(frame_slider.value)
            target_joints = data['qpos'][current_frame][:14]
            
            # Create confirmation modal
            with server.gui.add_modal(" Confirm Reset") as modal:
                server.gui.add_markdown(
                    f"## Reset Robot to Frame {current_frame}?\n\n"
                    f"The robot will smoothly move to the joint positions from frame {current_frame}.\n\n"
                    f"**Duration:** 2.0 seconds\n\n"
                    f" **Make sure the workspace is clear!**"
                )
                
                confirm_button = server.gui.add_button(" Confirm Reset", color="green")
                cancel_button = server.gui.add_button(" Cancel")
                
                @confirm_button.on_click
                def _(_):
                    modal.close()
                    
                    async def do_reset():
                        robot_status.value = f"🔄 Resetting to frame {current_frame}..."
                        reset_robot_button.disabled = True
                        reset_zombie_button.disabled = True
                        execute_robot_button.disabled = True
                        
                        try:
                            result = await send_robot_command(
                                args.robot_host,
                                args.robot_port,
                                {"command": "reset", "joints": target_joints.tolist(), "duration": 2.0}
                            )
                            
                            if result["status"] == "success":
                                robot_status.value = f" Reset to frame {current_frame} complete"
                            else:
                                robot_status.value = f" Reset failed: {result.get('message')}"
                        except Exception as e:
                            robot_status.value = f" Reset error: {str(e)}"
                        finally:
                            reset_robot_button.disabled = False
                            reset_zombie_button.disabled = False
                            if predicted_actions is not None:
                                execute_robot_button.disabled = False
                    
                    run_async(do_reset())
                
                @cancel_button.on_click
                def _(_):
                    modal.close()
                    robot_status.value = "Reset cancelled"
        
        # Reset to zombie pose button callback
        @reset_zombie_button.on_click
        def _(event):
            # Create confirmation modal
            with server.gui.add_modal("🧟 Confirm Zombie Pose Reset") as modal:
                server.gui.add_markdown(
                    f"## Reset Robot to Zombie Pose?\n\n"
                    f"The robot will move its arms to a fully extended forward position:\n"
                    f"- Shoulders raised to ~90 degrees\n"
                    f"- Elbows fully extended\n\n"
                    f"**Duration:** 3.0 seconds\n\n"
                    f"**Make sure the workspace is clear!**"
                )
                
                confirm_zombie = server.gui.add_button("Confirm Zombie Reset", color="green")
                cancel_zombie = server.gui.add_button("Cancel")
                
                @confirm_zombie.on_click
                def _(_):
                    modal.close()
                    
                    async def do_zombie_reset():
                        robot_status.value = "🧟 Resetting to zombie pose..."
                        reset_robot_button.disabled = True
                        reset_zombie_button.disabled = True
                        execute_robot_button.disabled = True
                        
                        try:
                            result = await send_robot_command(
                                args.robot_host,
                                args.robot_port,
                                {"command": "reset_zombie", "duration": 3.0}
                            )
                            
                            if result["status"] == "success":
                                robot_status.value = "Zombie pose reset complete"
                            else:
                                robot_status.value = f"Zombie reset failed: {result.get('message')}"
                        except Exception as e:
                            robot_status.value = f"Zombie reset error: {str(e)}"
                        finally:
                            reset_robot_button.disabled = False
                            reset_zombie_button.disabled = False
                            if predicted_actions is not None:
                                execute_robot_button.disabled = False
                    
                    run_async(do_zombie_reset())
                
                @cancel_zombie.on_click
                def _(_):
                    modal.close()
                    robot_status.value = "Zombie reset cancelled"
        
        # Execute on robot button callback
        @execute_robot_button.on_click
        def _(event):
            if predicted_actions is None:
                robot_status.value = " No predicted actions - run inference first"
                return
            
            # Create confirmation modal
            with server.gui.add_modal(" Confirm Execution") as modal:
                server.gui.add_markdown(
                    f"##  Execute {len(predicted_actions)} Actions on Robot?\n\n"
                    f"**Action sequence:**\n"
                    f"- Number of actions: {len(predicted_actions)}\n"
                    f"- Duration: ~{len(predicted_actions)/50.0:.2f} seconds\n\n"
                    f"###  SAFETY WARNING \n"
                    f"- Keep emergency stop ready\n"
                    f"- Ensure workspace is clear\n"
                    f"- Be ready to power off if needed\n"
                )
                
                confirm_exec_button = server.gui.add_button(
                    " I Understand - Execute Now",
                    color="red"
                )
                cancel_exec_button = server.gui.add_button(" Cancel")
                
                @confirm_exec_button.on_click
                def _(_):
                    modal.close()
                    
                    async def do_execute():
                        robot_status.value = f"🚀 Executing {len(predicted_actions)} actions..."
                        reset_robot_button.disabled = True
                        execute_robot_button.disabled = True
                        infer_button.disabled = True
                        
                        try:
                            result = await send_robot_command(
                                args.robot_host,
                                args.robot_port,
                                {"command": "execute", "actions": predicted_actions.tolist()}
                            )
                            
                            if result["status"] == "success":
                                robot_status.value = f" Execution complete"
                            else:
                                robot_status.value = f" Execution failed: {result.get('message')}"
                        except Exception as e:
                            robot_status.value = f" Execution error: {str(e)}"
                        finally:
                            reset_robot_button.disabled = False
                            execute_robot_button.disabled = False
                            infer_button.disabled = False
                    
                    run_async(do_execute())
                
                @cancel_exec_button.on_click
                def _(_):
                    modal.close()
                    robot_status.value = "Execution cancelled"
        
        # Replay teleop button callback
        @replay_teleop_button.on_click
        def _(event):
            # Get current frame and determine replay window
            current_frame = int(frame_slider.value)
            
            # Replay 50 frames starting from current frame (or until end of data)
            end_frame = min(current_frame + 50, len(data['actions']))
            num_frames = end_frame - current_frame
            
            if num_frames < 10:
                robot_status.value = f"Not enough frames to replay (need at least 10, have {num_frames})"
                return
            
            # Get raw teleop actions from HDF5
            teleop_actions = data['actions'][current_frame:end_frame]
            
            # Create confirmation modal
            with server.gui.add_modal("Confirm Teleop Replay") as modal:
                server.gui.add_markdown(
                    f"**⚠️ About to replay {num_frames} teleop frames**\n\n"
                    f"Frame range: {current_frame} → {end_frame-1}\n\n"
                    f"This will execute the **raw recorded joint commands** from the HDF5 file.\n\n"
                    f"**This bypasses the policy entirely** - good for testing if the robot "
                    f"can physically perform the recorded motions.\n\n"
                    f"**Duration:** ~{num_frames/50:.1f}s at 50Hz"
                )
                
                confirm_replay_button = server.gui.add_button(
                    "🎮 Start Replay",
                    color="green"
                )
                cancel_replay_button = server.gui.add_button("Cancel")
                
                @confirm_replay_button.on_click
                def _(_):
                    modal.close()
                    
                    async def do_replay():
                        robot_status.value = f"🎮 Replaying {num_frames} teleop frames..."
                        reset_robot_button.disabled = True
                        execute_robot_button.disabled = True
                        replay_teleop_button.disabled = True
                        infer_button.disabled = True
                        
                        try:
                            result = await send_robot_command(
                                args.robot_host,
                                args.robot_port,
                                {"command": "replay_teleop", "actions": teleop_actions.tolist()}
                            )
                            
                            robot_status.value = f"Teleop replay complete: {result.get('message', 'OK')}"
                        except Exception as e:
                            print(f"Teleop replay error: {e}")
                            robot_status.value = f"Teleop replay error: {str(e)}"
                        finally:
                            reset_robot_button.disabled = False
                            execute_robot_button.disabled = False
                            replay_teleop_button.disabled = False
                            infer_button.disabled = False
                    
                    run_async(do_replay())
                
                @cancel_replay_button.on_click
                def _(_):
                    modal.close()
                    robot_status.value = "Replay cancelled"
        
        # Single-step inference button callback
        @single_step_button.on_click
        def _(event):
            async def do_single_step():
                nonlocal predicted_actions, action_chunk_horizon, last_live_observation
                
                robot_status.value = "Getting live observation..."
                single_step_button.disabled = True
                continuous_button.disabled = True
                reset_robot_button.disabled = True
                execute_robot_button.disabled = True
                infer_button.disabled = True
                
                try:
                    # Get live observation from robot
                    obs = await get_live_observation_from_robot(
                        args.robot_host, args.robot_port, args.prompt
                    )
                    last_live_observation = obs  # Store for camera display
                    
                    # Update camera displays with live feeds
                    update_camera_displays()
                    
                    # Run inference
                    robot_status.value = "Running policy inference..."
                    start_time = time.time()
                    result = policy.infer(obs)
                    inference_time = time.time() - start_time
                    
                    # Extract predicted actions
                    predicted_actions = result['actions']
                    action_chunk_horizon = predicted_actions.shape[0]
                    
                    # Update visualization
                    robot_status.value = f"Executing {action_chunk_horizon} actions..."
                    action_index_slider.max = action_chunk_horizon - 1
                    action_index_slider.value = 0
                    show_predicted_cb.disabled = False
                    action_index_slider.disabled = False
                    
                    # Enable predicted actions visualization and update
                    show_predicted_cb.value = True
                    show_gt_cb.value = False
                    update_visualization()
                    
                    # Execute on robot
                    result = await send_robot_command(
                        args.robot_host,
                        args.robot_port,
                        {"command": "execute", "actions": predicted_actions.tolist()}
                    )
                    
                    if result["status"] == "success":
                        robot_status.value = f"✓ Single-step complete ({inference_time*1000:.1f}ms inference)"
                    else:
                        robot_status.value = f"Error: {result.get('message')}"
                    
                except Exception as e:
                    robot_status.value = f"Error: {str(e)}"
                    print(f"Single-step error: {e}")
                finally:
                    single_step_button.disabled = not use_live_cam.value
                    continuous_button.disabled = not use_live_cam.value
                    reset_robot_button.disabled = False
                    execute_robot_button.disabled = False
                    infer_button.disabled = False
            
            run_async(do_single_step())
        
        # Continuous inference button callback
        is_continuous_running = False
        auto_started_recording = False
        
        @continuous_button.on_click
        def _(event):
            nonlocal is_continuous_running, auto_started_recording
            
            if not is_continuous_running:
                # Start continuous mode
                is_continuous_running = True
                continuous_button.name = "Stop Continuous"
                continuous_button.color = "red"
                
                # Disable conflicting buttons
                single_step_button.disabled = True
                reset_robot_button.disabled = True
                execute_robot_button.disabled = True
                infer_button.disabled = True
                replay_teleop_button.disabled = True
                frame_slider.disabled = True
                
                async def continuous_loop():
                    nonlocal predicted_actions, action_chunk_horizon, is_continuous_running, last_live_observation, auto_started_recording
                    
                    # Auto-start recording if not already recording
                    recording_was_started = False
                    label_name = label_name_input.value.strip() or "default"
                    
                    try:
                        status_check = await send_robot_command(
                            args.robot_host,
                            args.robot_port,
                            {"command": "ping"}
                        )
                        
                        # Start recording automatically with current label
                        record_result = await send_robot_command(
                            args.robot_host,
                            args.robot_port,
                            {
                                "command": "start_recording",
                                "save_dir": "./data/policy_episodes",
                                "label_name": label_name
                            }
                        )
                        
                        if record_result["status"] == "success":
                            recording_status.value = f"Recording '{record_result['label_name']}/episode_{record_result['episode_idx']}.hdf5'"
                            stop_recording_button.disabled = False
                            start_recording_button.disabled = True
                            label_name_input.disabled = True
                            recording_was_started = True
                            auto_started_recording = True
                            print(f"Auto-started recording: {record_result['filepath']}")
                    except Exception as e:
                        print(f"Could not start recording: {e}")
                        recording_status.value = f"Recording not available: {e}"
                    
                    loop_count = 0
                    while is_continuous_running:
                        try:
                            loop_count += 1
                            robot_status.value = f"Continuous #{loop_count}: Getting observation..."
                            
                            # Get live observation from robot
                            obs = await get_live_observation_from_robot(
                                args.robot_host, args.robot_port, args.prompt
                            )
                            last_live_observation = obs
                            update_camera_displays()
                            
                            # Run inference
                            robot_status.value = f"Continuous #{loop_count}: Running inference..."
                            start_time = time.time()
                            result = policy.infer(obs)
                            inference_time = time.time() - start_time
                            
                            # Extract predicted actions
                            predicted_actions = result['actions']
                            action_chunk_horizon = predicted_actions.shape[0]
                            
                            # Update UI
                            action_index_slider.max = action_chunk_horizon - 1
                            action_index_slider.value = 0
                            show_predicted_cb.disabled = False
                            action_index_slider.disabled = False
                            show_predicted_cb.value = True
                            show_gt_cb.value = False
                            update_visualization()
                            
                            # Execute on robot (this now handles recording internally)
                            robot_status.value = f"Continuous #{loop_count}: Executing actions..."
                            exec_result = await send_robot_command(
                                args.robot_host,
                                args.robot_port,
                                {"command": "execute", "actions": predicted_actions.tolist()}
                            )
                            
                            if exec_result["status"] == "success":
                                robot_status.value = f"✓ Continuous #{loop_count} complete ({inference_time*1000:.1f}ms)"
                            else:
                                robot_status.value = f"Error in loop #{loop_count}: {exec_result.get('message')}"
                                is_continuous_running = False
                                break
                            
                        except Exception as e:
                            robot_status.value = f"Error in continuous loop: {str(e)}"
                            print(f"Continuous loop error: {e}")
                            is_continuous_running = False
                            break
                    
                    # Auto-stop recording if we started it
                    if recording_was_started and auto_started_recording:
                        try:
                            stop_result = await send_robot_command(
                                args.robot_host,
                                args.robot_port,
                                {"command": "stop_recording"}
                            )
                            if stop_result["status"] == "success":
                                recording_status.value = f"Saved {stop_result['episode_length']} timesteps"
                                start_recording_button.disabled = False
                                label_name_input.disabled = False
                                print(f"Auto-stopped recording: {stop_result['filepath']}")
                            auto_started_recording = False
                        except Exception as e:
                            print(f"Could not stop recording: {e}")
                    
                    # Cleanup after stopping
                    continuous_button.name = "Start Continuous"
                    continuous_button.color = None
                    single_step_button.disabled = not use_live_cam.value
                    reset_robot_button.disabled = False
                    execute_robot_button.disabled = False
                    infer_button.disabled = False
                    replay_teleop_button.disabled = False
                    frame_slider.disabled = False
                    robot_status.value = f"Continuous mode stopped after {loop_count} loops"
                
                run_async(continuous_loop())
            else:
                # Stop continuous mode
                is_continuous_running = False
                robot_status.value = "Stopping continuous mode..."
        
        # Recording button callbacks
        @start_recording_button.on_click
        def _(event):
            async def do_start_recording():
                label_name = label_name_input.value.strip()
                if not label_name:
                    recording_status.value = "Error: Label name cannot be empty"
                    return
                
                recording_status.value = f"Starting recording for '{label_name}'..."
                start_recording_button.disabled = True
                label_name_input.disabled = True
                
                try:
                    result = await send_robot_command(
                        args.robot_host,
                        args.robot_port,
                        {
                            "command": "start_recording",
                            "save_dir": "./data/policy_episodes",
                            "label_name": label_name
                        }
                    )
                    
                    if result["status"] == "success":
                        recording_status.value = f"Recording '{result['label_name']}/episode_{result['episode_idx']}.hdf5'"
                        stop_recording_button.disabled = False
                        robot_status.value = f"Recording started: {result['label_name']}"
                    else:
                        recording_status.value = f"Failed: {result.get('message')}"
                        start_recording_button.disabled = False
                        label_name_input.disabled = False
                except Exception as e:
                    recording_status.value = f"Error: {str(e)}"
                    start_recording_button.disabled = False
                    label_name_input.disabled = False
            
            run_async(do_start_recording())
        
        @stop_recording_button.on_click
        def _(event):
            async def do_stop_recording():
                recording_status.value = "Stopping recording..."
                stop_recording_button.disabled = True
                
                try:
                    result = await send_robot_command(
                        args.robot_host,
                        args.robot_port,
                        {"command": "stop_recording"}
                    )
                    
                    if result["status"] == "success":
                        recording_status.value = f"Saved {result['episode_length']} timesteps"
                        start_recording_button.disabled = False
                        label_name_input.disabled = False
                        robot_status.value = "Recording stopped and saved"
                        print(f"Saved episode: {result['filepath']}")
                    else:
                        recording_status.value = f"Failed: {result.get('message')}"
                        stop_recording_button.disabled = False
                except Exception as e:
                    recording_status.value = f"Error: {str(e)}"
                    stop_recording_button.disabled = False
            
            run_async(do_stop_recording())
        
        # Emergency stop button callback
        @estop_button.on_click
        def _(event):
            async def do_stop():
                robot_status.value = " EMERGENCY STOP ACTIVATED"
                try:
                    await send_robot_command(
                        args.robot_host,
                        args.robot_port,
                        {"command": "emergency_stop"}
                    )
                except Exception as e:
                    robot_status.value = f" EMERGENCY STOP (error: {e})"
            
            run_async(do_stop())
    
    # Play actions button callback
    @play_actions_button.on_click
    def _(_):
        nonlocal is_playing_actions
        is_playing_actions = not is_playing_actions
        
        if is_playing_actions:
            play_actions_button.name = "⏸️ Pause Action Chunk"
        else:
            play_actions_button.name = "▶️ Play Action Chunk"
    
    def update_visualization():
        """Update robot visualization based on current state."""
        if show_ground_truth:
            # Show ground truth qpos
            joints = data['qpos'][current_frame][:14]
        elif predicted_actions is not None and show_predicted_cb.value:
            # Show predicted action at selected index
            action_idx = int(action_index_slider.value)
            joints = predicted_actions[action_idx]
        else:
            return
        
        # Convert to URDF format and update
        joints_urdf = extract_hand_joints_for_urdf(joints)
        viser_urdf.update_cfg(joints_urdf[:viser_urdf._urdf.num_actuated_joints])
        
        # Update camera displays
        update_camera_displays()
    
    def update_camera_displays():
        """Update camera image displays with proper transformations and positioning."""
        # Check if we should use live camera feeds
        use_live = args.robot_execution and use_live_cam.value and last_live_observation is not None
        
        # Map live observation image keys to camera topics
        live_image_map = {
            'cam_head': 'ego_cam',  # or 'cam_high' depending on dataset
            'cam_left_wrist': 'cam_left_wrist',
            'cam_right_wrist': 'cam_right_wrist',
        }
        
        for topic in data['camera_topics']:
            if not camera_checkboxes[topic].value:
                # Remove image if checkbox is unchecked
                handle_name = f"/{topic}"
                try:
                    server.scene.remove(handle_name)
                except:
                    pass
                continue
            
            # Get image data - either from live observation or HDF5
            if use_live:
                # Find matching live image key for this topic
                live_key = None
                for key, mapped_topic in live_image_map.items():
                    if mapped_topic == topic or (topic in ['ego_cam', 'cam_high'] and key == 'cam_head'):
                        live_key = key
                        break
                
                if live_key and live_key in last_live_observation['images']:
                    img_array = last_live_observation['images'][live_key]
                    # Ensure it's uint8
                    if img_array.dtype != np.uint8:
                        img_array = (img_array * 255).astype(np.uint8)
                else:
                    # Skip if no matching live image
                    continue
            else:
                # Get image from HDF5
                if data['image_formats'][topic] == 'array':
                    img_array = data['camera_data'][topic][current_frame]
                    # Convert to uint8 if needed
                    if img_array.dtype != np.uint8:
                        img_array = (img_array * 255).astype(np.uint8)
                    # Convert BGR to RGB (swap channels 0 and 2)
                    img_array = img_array[:, :, [2, 1, 0]]
                else:  # JPEG format
                    img_data = data['camera_data'][topic][current_frame]
                    img = decode_jpeg_image(img_data)
                    img_array = np.array(img)
            
            # Apply transformations (same as data_replay.py)
            img_array = np.flipud(img_array)  # Flip vertically
            if topic in ['ego_cam', 'cam_high']:  # Apply different rotations based on topic
                img_array = np.rot90(img_array, k=1)  # Counterclockwise
            else:
                img_array = np.rot90(img_array, k=-1)  # Clockwise
            
            # Set position and rotation based on camera type (same as data_replay.py)
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
            
            # Add/update image in viser
            handle_name = f"/{topic}"
            server.scene.add_image(
                handle_name,
                image=img_array,
                render_width=render_width,
                render_height=render_width * img_array.shape[0] / img_array.shape[1],
                position=(pos_x, pos_y, pos_z),
                wxyz=(w, x, y, z),
            )
    
    # Set initial configuration
    update_visualization()
    
    print(f"\n{'='*80}")
    print("H1 Policy Inference Visualization Client Started!")
    print(f"{'='*80}")
    print(f"Total frames: {data['num_frames']}")
    print(f"Policy server: {args.host}:{args.port}")
    print(f"\nInstructions:")
    print("1. Use 'Select Frame' slider to choose a frame")
    print("2. Click '🤖 Infer Action Chunk' to get policy predictions")
    print("3. Click '▶️ Play Action Chunk' to visualize the action sequence")
    print("4. Use 'Action Index' slider to manually scrub through actions")
    print(f"{'='*80}\n")
    
    # Main loop for action playback
    last_update = time.time()
    while True:
        time.sleep(0.01)  # Small sleep to prevent busy loop
        
        if is_playing_actions and predicted_actions is not None:
            current_time = time.time()
            dt = current_time - last_update
            
            # Update at target FPS adjusted by speed
            target_dt = 1.0 / (args.fps * speed_slider.value)
            
            if dt >= target_dt:
                last_update = current_time
                
                # Advance action index
                action_play_idx = int(action_index_slider.value)
                action_play_idx += 1
                
                if action_play_idx >= action_chunk_horizon:
                    # Loop back to start
                    action_play_idx = 0
                    is_playing_actions = False
                    play_actions_button.name = "▶️ Play Action Chunk"
                
                action_index_slider.value = action_play_idx
                update_visualization()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))

