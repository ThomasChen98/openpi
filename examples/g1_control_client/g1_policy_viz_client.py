#!/usr/bin/env python3
"""G1 Policy Inference Visualization Client

Loads observations from G1 dataset and sends them to the policy server for inference,
then visualizes the predicted action chunks using viser.

Features:
* Load observations (images + state) from HDF5 datasets
* Send observations to policy server
* Receive and visualize predicted action chunks
* Display locomotion state (mode_machine, IMU, leg positions)
* Display controller inputs (joysticks, buttons)
* Interactive frame selection and playback controls
* Display robot motion using viser URDF viewer with Dex3 hands
* Display camera images alongside robot visualization
* Robot execution: reset, execute actions, replay teleop, continuous inference
* Recording: capture policy execution episodes to HDF5

Usage:
1. Start policy server:
   uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_g1_auto --policy.dir=checkpoints/g1

2. Run this client with HDF5 (visualization only):
   python g1_policy_viz_client.py --data-path /mnt/ssd1/yuxin/g1_data/cabinetbottle/episode_2.hdf5

3. Run with robot execution enabled:
   python g1_policy_viz_client.py --data-path /path/to/episode.hdf5 --robot-execution
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import io
import json
import os
import time
from pathlib import Path

import cv2
import einops
import h5py
import numpy as np
import tyro
import websockets
from PIL import Image
from tqdm import tqdm
from yourdfpy import URDF

import viser
from viser.extras import ViserUrdf

try:
    from openpi_client import websocket_client_policy as _websocket_client_policy
except ImportError:
    print("Warning: openpi_client not found. Policy inference will not work.")
    _websocket_client_policy = None

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


@dataclasses.dataclass
class Args:
    """Command line arguments."""
    
    # Data paths
    data_path: str = "/mnt/ssd1/yuxin/g1_data/cabinetbottle/episode_2.hdf5"
    """Path to HDF5 file"""
    
    urdf_path: str | None = None
    """Path to robot URDF file (defaults to G1 URDF)"""
    
    # Policy server connection
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None
    
    # Task/prompt
    prompt: str = "pick up the bottle and put it in the cabinet"
    
    # Visualization settings
    fps: float | None = None
    """FPS for playback (defaults to file's fps)"""
    
    start_frame: int = 0
    load_meshes: bool = True
    load_collision_meshes: bool = False
    viser_port: int = 8080
    
    # Robot execution settings
    robot_execution: bool = False
    """Enable robot execution (connects to g1_remote_client in listen mode)"""
    
    robot_host: str = "localhost"
    """Robot client host (via SSH reverse tunnel)"""
    
    robot_port: int = 5008
    """Robot client port"""


def get_observation_at_frame(
    data: dict,
    frame_idx: int,
    prompt: str,
    action_dim: int = 28,
    target_size: tuple = (224, 224),
) -> dict:
    """Get observation at specific frame in the format expected by the policy.
    
    Args:
        data: Loaded data from HDF5
        frame_idx: Frame index to get observation from
        prompt: Task prompt/instruction
        action_dim: Action dimension (28 for G1 arms+hands)
        target_size: Target image size (height, width)
        
    Returns:
        Observation dictionary compatible with G1 policy
    """
    # Get state (first action_dim dimensions of qpos)
    state = data['qpos'][frame_idx][:action_dim].astype(np.float32)
    
    # Policy expected camera name for G1
    policy_cameras = ['cam_head']  # G1 only has head camera in current setup
    
    # Map dataset names to policy names
    camera_aliases = {
        'ego_cam': 'cam_head',
    }
    
    images = {}
    for policy_name in policy_cameras:
        # Check if camera exists under policy name or original name
        source_name = None
        if policy_name in data['camera_data']:
            source_name = policy_name
        else:
            # Check aliases
            for orig, mapped in camera_aliases.items():
                if mapped == policy_name and orig in data['camera_data']:
                    source_name = orig
                    break
        
        if source_name is not None:
            # Get image
            if data['image_formats'][source_name] == 'array':
                img = data['camera_data'][source_name][frame_idx]
            else:  # JPEG format
                img = decode_jpeg_image(data['camera_data'][source_name][frame_idx])
                img = np.array(img)
            
            # Resize to target size
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            
            # Convert to (C, H, W) uint8 format expected by policy
            img = einops.rearrange(img, 'h w c -> c h w')
            images[policy_name] = img.astype(np.uint8)
    
    # Add loco_state if available (for 32-dim state)
    loco_state = None
    if data['loco_state'] is not None:
        loco_state = data['loco_state'][frame_idx].astype(np.float32)
    
    return {
        "state": state,
        "images": images,
        "prompt": prompt,
        "loco_state": loco_state,
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
        async with websockets.connect(uri, ping_timeout=30) as ws:
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
            "state": np.ndarray (28,),
            "images": {
                "cam_head": np.ndarray (224, 224, 3),
            },
            "prompt": str
        }
    """
    # Request observation from robot
    response = await send_robot_command(host, port, {"command": "get_observation"})
    
    if response["status"] != "success":
        raise RuntimeError(f"Failed to get observation: {response.get('message', 'Unknown error')}")
    
    # Decode base64 images
    def decode_base64_img(b64_str: str) -> np.ndarray:
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(img_bytes))
        img_array = np.array(img)
        # Resize to 224x224 if needed
        if img_array.shape[:2] != (224, 224):
            img = img.resize((224, 224), Image.LANCZOS)
            img_array = np.array(img)
        return img_array
    
    # Extract and format observation
    state = np.array(response["state"], dtype=np.float32)
    images = {
        "cam_head": decode_base64_img(response["images"]["cam_head"]),
    }
    
    return {
        "state": state,
        "images": images,
        "prompt": prompt
    }


def main(args: Args) -> None:
    """Run G1 policy inference visualization client."""
    
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
    if not os.path.isabs(args.data_path):
        data_path = str(script_dir / args.data_path)
    else:
        data_path = args.data_path
    
    if args.urdf_path is None:
        urdf_path = get_urdf_path()
    elif not os.path.isabs(args.urdf_path):
        urdf_path = str(script_dir / args.urdf_path)
    else:
        urdf_path = args.urdf_path
    
    # Load HDF5 data
    data = load_hdf5_data(data_path)
    
    # Use file's FPS if not specified
    fps = args.fps if args.fps is not None else data['fps']
    
    # Connect to policy server (if available)
    policy = None
    server_metadata = {}
    action_dim = 28  # Default for G1
    
    if _websocket_client_policy is not None:
        try:
            print(f"\nConnecting to policy server at {args.host}:{args.port}")
            policy = _websocket_client_policy.WebsocketClientPolicy(
                host=args.host,
                port=args.port,
                api_key=args.api_key,
            )
            
            server_metadata = policy.get_server_metadata()
            print(f"Server metadata: {server_metadata}")
            
            action_dim = server_metadata.get('action_dim', 28)
            print(f"Using action_dim: {action_dim}")
            
            # Warm up the policy with a test observation
            print("Warming up policy...")
            test_obs = get_observation_at_frame(data, 0, args.prompt, action_dim=action_dim)
            policy.infer(test_obs)
            print("Policy ready!")
        except Exception as e:
            print(f"Warning: Could not connect to policy server: {e}")
            print("Running in visualization-only mode.")
            policy = None
    
    # Start viser server
    print(f"\nStarting viser server on port {args.viser_port}...")
    server = viser.ViserServer(port=args.viser_port, host="0.0.0.0")
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
    action_chunk_horizon = 10
    is_playing_actions = False
    action_play_idx = 0
    show_ground_truth = True
    image_handles = {}
    last_live_observation = None  # Store last live observation for display
    
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
        if policy is None:
            infer_button.disabled = True
        
        play_actions_button = server.gui.add_button("▶️ Play Action Chunk", disabled=True)
        
        inference_status = server.gui.add_text(
            "Status",
            initial_value="Ready - Select frame and click 'Infer Action Chunk'" if policy else "No policy server connected",
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
        use_actions_cb = server.gui.add_checkbox(
            "Use actions (vs qpos)",
            initial_value=True,
        )
    
    with server.gui.add_folder("Camera Views", expand_by_default=False):
        camera_checkboxes = {}
        for topic in data['camera_topics']:
            display_name = topic.replace('_', ' ').title()
            camera_checkboxes[topic] = server.gui.add_checkbox(
                f"Show {display_name}",
                initial_value=True
            )
    
    # Locomotion data display (if available)
    loco_displays = {}
    
    if data['has_loco_data']:
        with server.gui.add_folder("Locomotion State", expand_by_default=False):
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
        
        with server.gui.add_folder("Controller Input", expand_by_default=False):
            loco_displays['joysticks'] = server.gui.add_text(
                "Joysticks",
                initial_value="Lx:-- Ly:-- Rx:-- Ry:--",
                disabled=True,
            )
            loco_displays['predicted_loco'] = server.gui.add_text(
                "Policy Loco Cmd",
                initial_value="vx:-- vy:-- vyaw:--",
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
            "Left Arm [0:7]",
            initial_value="--",
            disabled=True,
        )
        joint_displays['right_arm'] = server.gui.add_text(
            "Right Arm [7:14]",
            initial_value="--",
            disabled=True,
        )
        joint_displays['left_hand'] = server.gui.add_text(
            "Left Hand [14:21]",
            initial_value="--",
            disabled=True,
        )
        joint_displays['right_hand'] = server.gui.add_text(
            "Right Hand [21:28]",
            initial_value="--",
            disabled=True,
        )
    
    # === ROBOT EXECUTION GUI ===
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
            
            execute_robot_button = server.gui.add_button(
                "🚀 Execute on Robot",
                disabled=True,
                hint="Execute predicted actions on real robot"
            )
            
            replay_teleop_button = server.gui.add_button(
                "🎮 Replay Teleop Data",
                hint="Replay raw teleop joint commands from HDF5 (bypass policy)"
            )
            
            use_live_cam = server.gui.add_checkbox(
                "📷 Use Live Cameras",
                initial_value=False,
                hint="Use robot cameras for inference instead of dataset"
            )
            
            single_step_button = server.gui.add_button(
                "⏯️ Single Step Inference",
                disabled=True,
                hint="Get observation → infer → execute once"
            )
            
            continuous_button = server.gui.add_button(
                "🔄 Start Continuous",
                disabled=True,
                hint="Toggle continuous inference loop"
            )
            
            server.gui.add_markdown("### 💾 Recording")
            
            label_name_input = server.gui.add_text(
                "Label Name",
                initial_value="default",
                hint="Task/label name for organizing episodes (e.g., 'pick_bottle')"
            )
            
            recording_status = server.gui.add_text(
                "Recording Status",
                initial_value="Not recording",
                disabled=True,
            )
            
            start_recording_button = server.gui.add_button(
                "🔴 Start Recording",
                color="green",
                hint="Start recording observations and actions to HDF5"
            )
            
            stop_recording_button = server.gui.add_button(
                "⏹️ Stop Recording",
                disabled=True,
                color="red",
                hint="Stop and save episode"
            )
            
            server.gui.add_markdown("### ⚠️ Safety")
            estop_button = server.gui.add_button(
                "🛑 EMERGENCY STOP",
                color="red"
            )
    
    # Visibility callbacks
    @show_meshes_cb.on_update
    def _(_):
        viser_urdf.show_visual = show_meshes_cb.value
    
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
                obs = get_observation_at_frame(data, current_frame, args.prompt, action_dim=action_dim)
            
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
            
            print(f"\nInference complete ({source}):")
            print(f"  Action chunk shape: {predicted_actions.shape}")
            print(f"  Inference time: {inference_time*1000:.1f}ms")
            
            # Enable execute button if robot execution is enabled
            if args.robot_execution:
                execute_robot_button.disabled = False
        
        if policy is None:
            inference_status.value = "No policy server connected"
            return
        
        run_async(do_inference())
    
    # === ROBOT EXECUTION CALLBACKS ===
    if args.robot_execution:
        # Reset robot button callback
        @reset_robot_button.on_click
        def _(event):
            current_frame_val = int(frame_slider.value)
            target_joints = data['actions'][current_frame_val]  # 28 DOF
            
            # Create confirmation modal
            with server.gui.add_modal("⚠️ Confirm Reset") as modal:
                server.gui.add_markdown(
                    f"## Reset Robot to Frame {current_frame_val}?\n\n"
                    f"The robot will smoothly move to the joint positions from frame {current_frame_val}.\n\n"
                    f"**Duration:** 2.0 seconds\n\n"
                    f"⚠️ **Make sure the workspace is clear!**"
                )
                
                confirm_button = server.gui.add_button("✅ Confirm Reset", color="green")
                cancel_button = server.gui.add_button("❌ Cancel")
                
                @confirm_button.on_click
                def _(_):
                    modal.close()
                    
                    async def do_reset():
                        robot_status.value = f"🔄 Resetting to frame {current_frame_val}..."
                        reset_robot_button.disabled = True
                        execute_robot_button.disabled = True
                        
                        try:
                            result = await send_robot_command(
                                args.robot_host,
                                args.robot_port,
                                {"command": "reset", "joints": target_joints.tolist(), "duration": 2.0}
                            )
                            
                            if result["status"] == "success":
                                robot_status.value = f"✓ Reset to frame {current_frame_val} complete"
                            else:
                                robot_status.value = f"❌ Reset failed: {result.get('message')}"
                        except Exception as e:
                            robot_status.value = f"❌ Reset error: {str(e)}"
                        finally:
                            reset_robot_button.disabled = False
                            if predicted_actions is not None:
                                execute_robot_button.disabled = False
                    
                    run_async(do_reset())
                
                @cancel_button.on_click
                def _(_):
                    modal.close()
                    robot_status.value = "Reset cancelled"
        
        # Execute on robot button callback
        @execute_robot_button.on_click
        def _(event):
            if predicted_actions is None:
                robot_status.value = "❌ No predicted actions - run inference first"
                return
            
            # Create confirmation modal
            with server.gui.add_modal("⚠️ Confirm Execution") as modal:
                server.gui.add_markdown(
                    f"## 🚀 Execute {len(predicted_actions)} Actions on Robot?\n\n"
                    f"**Action sequence:**\n"
                    f"- Number of actions: {len(predicted_actions)}\n"
                    f"- Duration: ~{len(predicted_actions)/30.0:.2f} seconds\n\n"
                    f"### ⚠️ SAFETY WARNING ⚠️\n"
                    f"- Keep emergency stop ready\n"
                    f"- Ensure workspace is clear\n"
                    f"- Be ready to power off if needed\n"
                )
                
                confirm_exec_button = server.gui.add_button(
                    "🚀 I Understand - Execute Now",
                    color="red"
                )
                cancel_exec_button = server.gui.add_button("❌ Cancel")
                
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
                                robot_status.value = f"✓ Execution complete"
                            else:
                                robot_status.value = f"❌ Execution failed: {result.get('message')}"
                        except Exception as e:
                            robot_status.value = f"❌ Execution error: {str(e)}"
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
            current_frame_val = int(frame_slider.value)
            
            # Replay 50 frames starting from current frame (or until end of data)
            end_frame = min(current_frame_val + 50, len(data['actions']))
            num_frames = end_frame - current_frame_val
            
            if num_frames < 10:
                robot_status.value = f"Not enough frames to replay (need at least 10, have {num_frames})"
                return
            
            # Get raw teleop actions from HDF5
            teleop_actions = data['actions'][current_frame_val:end_frame]
            
            # Create confirmation modal
            with server.gui.add_modal("🎮 Confirm Teleop Replay") as modal:
                server.gui.add_markdown(
                    f"**⚠️ About to replay {num_frames} teleop frames**\n\n"
                    f"Frame range: {current_frame_val} → {end_frame-1}\n\n"
                    f"This will execute the **raw recorded joint commands** from the HDF5 file.\n\n"
                    f"**This bypasses the policy entirely** - good for testing if the robot "
                    f"can physically perform the recorded motions.\n\n"
                    f"**Duration:** ~{num_frames/30:.1f}s at 30Hz"
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
                                {"command": "execute", "actions": teleop_actions.tolist()}
                            )
                            
                            robot_status.value = f"✓ Teleop replay complete: {result.get('message', 'OK')}"
                        except Exception as e:
                            print(f"Teleop replay error: {e}")
                            robot_status.value = f"❌ Teleop replay error: {str(e)}"
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
                
                robot_status.value = "📷 Getting live observation..."
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
                    robot_status.value = "🤖 Running policy inference..."
                    start_time = time.time()
                    result = policy.infer(obs)
                    inference_time = time.time() - start_time
                    
                    # Extract predicted actions
                    predicted_actions = result['actions']
                    action_chunk_horizon = predicted_actions.shape[0]
                    
                    # Update visualization
                    robot_status.value = f"🚀 Executing {action_chunk_horizon} actions..."
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
                        robot_status.value = f"❌ Error: {result.get('message')}"
                    
                except Exception as e:
                    robot_status.value = f"❌ Error: {str(e)}"
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
                continuous_button.name = "⏹️ Stop Continuous"
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
                        
                        if record_result.get("status") == "success":
                            recording_status.value = f"🔴 Recording '{record_result.get('label_name', label_name)}/episode_{record_result.get('episode_idx', 0)}.hdf5'"
                            stop_recording_button.disabled = False
                            start_recording_button.disabled = True
                            label_name_input.disabled = True
                            recording_was_started = True
                            auto_started_recording = True
                            print(f"Auto-started recording: {record_result.get('filepath', 'unknown')}")
                    except Exception as e:
                        print(f"Could not start recording: {e}")
                        recording_status.value = f"Recording not available: {e}"
                    
                    loop_count = 0
                    while is_continuous_running:
                        try:
                            loop_count += 1
                            robot_status.value = f"🔄 Continuous #{loop_count}: Getting observation..."
                            
                            # Get live observation from robot
                            obs = await get_live_observation_from_robot(
                                args.robot_host, args.robot_port, args.prompt
                            )
                            last_live_observation = obs
                            update_camera_displays()
                            
                            # Run inference
                            robot_status.value = f"🤖 Continuous #{loop_count}: Running inference..."
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
                            robot_status.value = f"🚀 Continuous #{loop_count}: Executing actions..."
                            exec_result = await send_robot_command(
                                args.robot_host,
                                args.robot_port,
                                {"command": "execute", "actions": predicted_actions.tolist()}
                            )
                            
                            if exec_result["status"] == "success":
                                robot_status.value = f"✓ Continuous #{loop_count} complete ({inference_time*1000:.1f}ms)"
                            else:
                                robot_status.value = f"❌ Error in loop #{loop_count}: {exec_result.get('message')}"
                                is_continuous_running = False
                                break
                            
                        except Exception as e:
                            robot_status.value = f"❌ Error in continuous loop: {str(e)}"
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
                            if stop_result.get("status") == "success":
                                recording_status.value = f"💾 Saved {stop_result.get('episode_length', '?')} timesteps"
                                start_recording_button.disabled = False
                                label_name_input.disabled = False
                                print(f"Auto-stopped recording: {stop_result.get('filepath', 'unknown')}")
                            auto_started_recording = False
                        except Exception as e:
                            print(f"Could not stop recording: {e}")
                    
                    # Cleanup after stopping
                    continuous_button.name = "🔄 Start Continuous"
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
                robot_status.value = "⏹️ Stopping continuous mode..."
        
        # Recording button callbacks
        @start_recording_button.on_click
        def _(event):
            async def do_start_recording():
                label_name = label_name_input.value.strip()
                if not label_name:
                    recording_status.value = "❌ Error: Label name cannot be empty"
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
                    
                    if result.get("status") == "success":
                        recording_status.value = f"🔴 Recording '{result.get('label_name', label_name)}/episode_{result.get('episode_idx', 0)}.hdf5'"
                        stop_recording_button.disabled = False
                        robot_status.value = f"Recording started: {result.get('label_name', label_name)}"
                    else:
                        recording_status.value = f"❌ Failed: {result.get('message', 'Unknown error')}"
                        start_recording_button.disabled = False
                        label_name_input.disabled = False
                except Exception as e:
                    recording_status.value = f"❌ Error: {str(e)}"
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
                    
                    if result.get("status") == "success":
                        recording_status.value = f"💾 Saved {result.get('episode_length', '?')} timesteps"
                        start_recording_button.disabled = False
                        label_name_input.disabled = False
                        robot_status.value = "Recording stopped and saved"
                        print(f"Saved episode: {result.get('filepath', 'unknown')}")
                    else:
                        recording_status.value = f"❌ Failed: {result.get('message', 'Unknown error')}"
                        stop_recording_button.disabled = False
                except Exception as e:
                    recording_status.value = f"❌ Error: {str(e)}"
                    stop_recording_button.disabled = False
            
            run_async(do_stop_recording())
        
        # Emergency stop button callback
        @estop_button.on_click
        def _(event):
            async def do_stop():
                robot_status.value = "🛑 EMERGENCY STOP ACTIVATED"
                try:
                    await send_robot_command(
                        args.robot_host,
                        args.robot_port,
                        {"command": "emergency_stop"}
                    )
                except Exception as e:
                    robot_status.value = f"🛑 EMERGENCY STOP (error: {e})"
            
            run_async(do_stop())
    
    # Play actions button callback
    @play_actions_button.on_click
    def _(_):
        nonlocal is_playing_actions
        is_playing_actions = not is_playing_actions
        
        if is_playing_actions:
            play_actions_button.name = "⏸️ Pause Action Chunk"
            show_predicted_cb.value = True
            show_gt_cb.value = False
        else:
            play_actions_button.name = "▶️ Play Action Chunk"
    
    def update_visualization():
        """Update robot visualization based on current state."""
        nonlocal current_frame
        
        # Get joint positions based on mode
        predicted_loco_cmd = None  # vx, vy, vyaw from policy
        if show_ground_truth:
            if use_actions_cb.value:
                joints = data['actions'][current_frame]
            else:
                joints = data['qpos'][current_frame]
        elif predicted_actions is not None and show_predicted_cb.value:
            action_idx = int(action_index_slider.value)
            full_action = predicted_actions[action_idx]
            # Policy outputs 32 dims: [28 DOF joints, vx, vy, vyaw, padding]
            if len(full_action) >= 32:
                joints = full_action[:28]  # Extract only joint positions for URDF
                predicted_loco_cmd = full_action[28:31]  # vx, vy, vyaw
            else:
                joints = full_action
        else:
            return
        
        # Get locomotion state if available
        loco_state = None
        loco_action = None
        if data['loco_state'] is not None:
            loco_state = data['loco_state'][current_frame]
        if data['loco_action'] is not None:
            loco_action = data['loco_action'][current_frame]
        
        # Map joints to URDF order (expects 28 DOF)
        urdf_joints = extract_joints_for_urdf_g1_28dof(joints, loco_state)
        
        # Update robot configuration
        viser_urdf.update_cfg(urdf_joints[:viser_urdf._urdf.num_actuated_joints])
        
        # Update joint info displays
        if len(joints) >= 28:
            joint_displays['left_arm'].value = f"{joints[0:7].round(3)}"
            joint_displays['right_arm'].value = f"{joints[7:14].round(3)}"
            joint_displays['left_hand'].value = f"{joints[14:21].round(3)}"
            joint_displays['right_hand'].value = f"{joints[21:28].round(3)}"
        
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
        
        # Display predicted locomotion commands from policy
        if predicted_loco_cmd is not None:
            loco_displays['predicted_loco'].value = (
                f"vx:{predicted_loco_cmd[0]:.3f} "
                f"vy:{predicted_loco_cmd[1]:.3f} "
                f"vyaw:{predicted_loco_cmd[2]:.3f}"
            )
        else:
            loco_displays['predicted_loco'].value = "vx:-- vy:-- vyaw:--"
        
        # Update camera images
        update_camera_displays()
    
    def update_camera_displays():
        """Update camera image displays."""
        nonlocal image_handles
        
        # Check if we should use live camera feeds
        use_live = args.robot_execution and hasattr(args, 'robot_execution') and use_live_cam.value if args.robot_execution else False
        use_live = use_live and last_live_observation is not None
        
        for topic in data['camera_topics']:
            if not camera_checkboxes[topic].value:
                # Remove image if checkbox is unchecked
                if topic in image_handles and image_handles[topic] is not None:
                    image_handles[topic].remove()
                    image_handles[topic] = None
                continue
            
            # Get image data - either from live observation or HDF5
            if use_live:
                # Map dataset camera names to live observation keys
                live_key = None
                if topic in ['ego_cam', 'cam_head']:
                    live_key = 'cam_head'
                
                if live_key and live_key in last_live_observation['images']:
                    img_array = last_live_observation['images'][live_key]
                    if img_array.dtype != np.uint8:
                        img_array = (img_array * 255).astype(np.uint8)
                else:
                    continue  # Skip if no matching live image
            else:
                # Get image from HDF5
                if data['image_formats'][topic] == 'array':
                    img_array = data['camera_data'][topic][current_frame]
                    if img_array.dtype != np.uint8:
                        img_array = (img_array * 255).astype(np.uint8)
                    # Convert BGR to RGB if needed
                    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                        img_array = img_array[:, :, [2, 1, 0]]
                else:  # JPEG format
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
    
    # Set initial configuration
    update_visualization()
    
    print(f"\n{'='*80}")
    print("G1 Policy Inference Visualization Client Started!")
    print(f"{'='*80}")
    print(f"Total frames: {data['num_frames']}")
    print(f"Number of joints: {data['num_joints']} (28 DOF: 14 arm + 14 hand)")
    print(f"Robot: {data['robot_name']}")
    print(f"FPS: {fps}")
    print(f"Has locomotion data: {data['has_loco_data']}")
    print(f"Policy server: {args.host}:{args.port} ({'connected' if policy else 'not connected'})")
    if args.robot_execution:
        print(f"Robot execution: ENABLED (connecting to {args.robot_host}:{args.robot_port})")
    else:
        print(f"Robot execution: DISABLED (use --robot-execution to enable)")
    print(f"\nInstructions:")
    print("1. Use 'Select Frame' slider to choose a frame")
    print("2. Click '🤖 Infer Action Chunk' to get policy predictions")
    print("3. Click '▶️ Play Action Chunk' to visualize the action sequence")
    print("4. Use 'Action Index' slider to manually scrub through actions")
    if args.robot_execution:
        print("5. Enable '📷 Use Live Cameras' to use robot's camera feeds")
        print("6. Click '🚀 Execute on Robot' to run predictions on the real robot")
    print(f"\nViser server running at: http://localhost:{args.viser_port}")
    print(f"{'='*80}\n")
    
    # Main loop for playback
    last_update = time.time()
    frame_time = 1.0 / fps
    is_playing_gt = False
    
    while True:
        time.sleep(0.01)  # Small sleep to prevent busy loop
        
        current_time = time.time()
        dt = current_time - last_update
        
        # Update at target FPS adjusted by speed
        target_dt = frame_time / speed_slider.value
        
        if dt >= target_dt:
            last_update = current_time
            
            if is_playing_gt:
                # Advance ground truth frame
                current_frame += 1
                if current_frame >= data['num_frames']:
                    current_frame = 0  # Loop back
                    is_playing_gt = False
                
                frame_slider.value = current_frame
                update_visualization()
            
            elif is_playing_actions and predicted_actions is not None:
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
