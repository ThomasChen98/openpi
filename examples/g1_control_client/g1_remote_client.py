#!/usr/bin/env python3
"""
G1 Remote Policy Client - Dex3 Hand Mode

Connects to OpenPi policy server and executes actions on G1 robot with Dex3 hands.

Architecture:
    - Arms: Controlled via DDS (unitree_sdk2py)
    - Hands: Dex3 hands controlled via DDS
    - Camera: Head camera from robot via ZMQ (image_server)

Usage:
    # On robot: Start image_server
    python3 image_server.py --camera-id 0

    # On GPU server: Start OpenPi policy server
    uv run scripts/serve_policy.py policy:checkpoint --policy.config=g1_config

    # On laptop: Run this client
    python g1_remote_client.py --server-host <gpu-ip> --server-port 8000

    # Or in listen mode for viz client control:
    python g1_remote_client.py --listen-mode --listen-port 5007
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import signal
import sys
import threading
import time
from io import BytesIO
from multiprocessing import shared_memory, Array, Lock
from pathlib import Path

import cv2
import numpy as np
import websockets
from websockets.server import serve

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# OpenPi client import
try:
    from openpi_client import websocket_client_policy, image_tools
except ImportError:
    logger.warning("openpi_client not installed, policy inference will not work")
    websocket_client_policy = None
    image_tools = None

# Import from xr_teleoperate
xr_teleoperate_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'xr_teleoperate')
if os.path.exists(xr_teleoperate_path):
    sys.path.insert(0, xr_teleoperate_path)

try:
    from teleop.robot_control.robot_arm import G1_29_ArmController
    from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
    from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
    from teleop.image_server.image_client import ImageClient
except ImportError as e:
    logger.error(f"Could not import from xr_teleoperate: {e}")
    logger.error("Make sure xr_teleoperate is cloned next to openpi/")
    sys.exit(1)

# Global for signal handler
controller = None


def signal_handler(sig, frame):
    global controller
    logger.info("Caught exit signal...")
    if controller is not None:
        controller.cleanup()
    sys.exit(0)


class G1RemoteClient:
    """
    G1 Remote Policy Client with Dex3 hands.
    
    Supports:
    - Policy inference from OpenPi server
    - WebSocket command server for viz client
    - Head camera streaming from robot
    - Arm and Dex3 hand control
    """

    def __init__(
        self,
        server_host: str = "localhost",
        server_port: int = 8000,
        head_camera_server_ip: str = "192.168.123.164",
        head_camera_server_port: int = 5555,
        prompt: str = "pick up the bottle and put it in the cabinet",
        control_fps: int = 30,
        motion_mode: bool = True,
    ):
        self.server_host = server_host
        self.server_port = server_port
        self.policy_client = None
        self.prompt = prompt
        self.control_fps = control_fps
        
        logger.info(f"Control frequency: {self.control_fps}Hz")
        
        # Initialize IK solver
        logger.info("Initializing IK solver...")
        self.ik_solver = G1_29_ArmIK(Unit_Test=False, Visualization=False)
        logger.info("IK solver ready")
        
        # Initialize robot arm controller (this initializes DDS)
        logger.info("Initializing arm controller...")
        self.robot = G1_29_ArmController(
            motion_mode=motion_mode,
            simulation_mode=False,
            dds_already_initialized=False
        )
        logger.info("Arm controller ready")
        
        # Initialize Dex3 hand controller
        logger.info("Initializing Dex3 hand controller...")
        self._init_hand_controller()
        logger.info("Hand controller ready")
        
        # Initialize camera in background
        logger.info("Initializing cameras (in background)...")
        self.cameras_ready = False
        self.head_camera_server_ip = head_camera_server_ip
        self.head_camera_server_port = head_camera_server_port
        self.head_img_array = None
        self.head_camera_client = None
        
        camera_init_thread = threading.Thread(target=self._init_head_camera, daemon=True)
        camera_init_thread.start()
        
        # Frame counter for performance monitoring
        self.frame_count = 0
        self.session_start_time = time.time()
        
        # Recording state
        self.is_recording = False
        self.episode_writer = None
        
        # Track last hand command for state construction
        self.last_hand_command = np.zeros(14, dtype=np.float32)  # 7 per hand for Dex3

    def _init_hand_controller(self):
        """Initialize Dex3 hand controller"""
        # Create shared arrays for hand data
        self.left_hand_array = Array('d', 26, lock=True)
        self.right_hand_array = Array('d', 26, lock=True)
        self.dual_hand_data_lock = Lock()
        self.dual_hand_state_array = Array('d', 14, lock=True)  # 7 joints per hand
        self.dual_hand_action_array = Array('d', 14, lock=True)
        
        # Initialize the hand controller (uses DDS, already initialized by arm controller)
        self.hand_ctrl = Dex3_1_Controller(
            left_hand_array_in=self.left_hand_array,
            right_hand_array_in=self.right_hand_array,
            dual_hand_data_lock=self.dual_hand_data_lock,
            dual_hand_state_array_out=self.dual_hand_state_array,
            dual_hand_action_array_out=self.dual_hand_action_array,
            fps=self.control_fps,
            Unit_Test=False,
            simulation_mode=False,
            dds_already_initialized=True  # DDS was initialized by arm controller
        )

    def _init_head_camera(self):
        """Initialize client to receive head camera from robot"""
        try:
            # Head camera: single RealSense at 480x640
            self.head_img_shape = (480, 640, 3)
            
            # Create shared memory buffer for head camera
            self.head_img_shm = shared_memory.SharedMemory(
                create=True,
                size=np.prod(self.head_img_shape) * np.uint8().itemsize
            )
            self.head_img_array = np.ndarray(
                self.head_img_shape, dtype=np.uint8, buffer=self.head_img_shm.buf
            )
            
            # Initialize image client
            self.head_camera_client = ImageClient(
                tv_img_shape=self.head_img_shape,
                tv_img_shm_name=self.head_img_shm.name,
                wrist_img_shape=None,
                wrist_img_shm_name=None,
                server_address=self.head_camera_server_ip,
                port=self.head_camera_server_port,
                image_show=False
            )
            
            # Start image receiving thread
            self.head_camera_thread = threading.Thread(
                target=self.head_camera_client.receive_process,
                daemon=True
            )
            self.head_camera_thread.start()
            
            # Wait briefly for first frame
            for _ in range(50):  # 5 seconds
                time.sleep(0.1)
                test_frame = self.head_img_array.copy()
                if np.any(test_frame != 0):
                    logger.info("Head camera connected!")
                    self.cameras_ready = True
                    return
            
            logger.warning("No frames received from head camera")
            self.cameras_ready = True
            
        except Exception as e:
            logger.error(f"Failed to initialize head camera: {e}")
            self.head_img_array = None
            self.head_camera_client = None
            self.cameras_ready = True

    def connect_to_policy_server(self):
        """Connect to the remote OpenPi policy server"""
        if websocket_client_policy is None:
            logger.error("openpi_client not installed")
            return False
        
        try:
            logger.info(f"Connecting to policy server at {self.server_host}:{self.server_port}...")
            self.policy_client = websocket_client_policy.WebsocketClientPolicy(
                host=self.server_host,
                port=self.server_port
            )
            
            metadata = self.policy_client.get_server_metadata()
            logger.info(f"Connected! Action dim: {metadata.get('action_dim', 'N/A')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to policy server: {e}")
            return False

    def get_observation(self) -> dict:
        """Construct observation for policy inference"""
        # Get current arm joint positions (14 DOF)
        current_arm_q = self.robot.get_current_dual_arm_q()
        
        # Get current hand joint positions (14 DOF)
        with self.dual_hand_data_lock:
            current_hand_q = np.array(self.dual_hand_state_array[:], dtype=np.float32)
        
        # Build 28-dim state: [arm_qpos(14), hand_qpos(14)]
        state_28 = np.concatenate([current_arm_q, current_hand_q])
        
        # Create dummy image (224x224 RGB, gray)
        dummy_image = np.full((224, 224, 3), 128, dtype=np.uint8)
        
        # Get head camera
        if self.head_img_array is not None:
            try:
                head_image = self.head_img_array.copy()
                if np.any(head_image != 0):
                    base_image = cv2.resize(head_image, (224, 224))
                    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
                else:
                    base_image = dummy_image
            except Exception:
                base_image = dummy_image
        else:
            base_image = dummy_image
        
        self.frame_count += 1
        
        observation = {
            "image": {
                "cam_head": base_image,
            },
            "state": state_28,
            "prompt": self.prompt,
        }
        
        return observation

    def execute_action_chunk(self, policy_actions: np.ndarray):
        """
        Execute a chunk of policy actions on the robot.
        
        Args:
            policy_actions: (N, 28) array of joint actions
                Format: [left_arm(7), right_arm(7), left_hand(7), right_hand(7)]
        """
        action_dim = policy_actions.shape[1]
        logger.info(f"Executing {len(policy_actions)} actions ({action_dim} DOF)...")
        
        if action_dim != 28:
            logger.warning(f"Expected 28 DOF, got {action_dim}")
        
        # Execute at control_fps
        for i, action in enumerate(policy_actions):
            # Extract arm and hand joints
            arm_joints = action[:14]
            hand_joints = action[14:28]
            
            # Send arm command
            self.robot.ctrl_dual_arm(
                q_target=arm_joints,
                tauff_target=np.zeros(14)
            )
            
            # Send hand command via shared arrays
            # Dex3 expects radians directly, no scaling needed
            left_hand = hand_joints[:7]
            right_hand = hand_joints[7:14]
            
            with self.dual_hand_data_lock:
                for j in range(7):
                    self.dual_hand_action_array[j] = left_hand[j]
                    self.dual_hand_action_array[7 + j] = right_hand[j]
            
            # Log every 10th action
            if i % 10 == 0:
                logger.info(f"  Step {i}/{len(policy_actions)}")
            
            time.sleep(1.0 / self.control_fps)
        
        logger.info("Action chunk execution complete")

    async def command_server(self, port: int = 5007):
        """
        WebSocket server that listens for commands from viz client.
        
        Supported commands:
        - {"command": "execute", "actions": [[...], ...]}  # Execute action chunk
        - {"command": "reset", "joints": [...]}  # Reset to joint positions
        - {"command": "get_state"}  # Get current robot state
        - {"command": "get_observation"}  # Get live observation
        - {"command": "emergency_stop"}  # Stop robot
        """
        async def handle_command(websocket):
            logger.info(f"Viz client connected from {websocket.remote_address}")
            
            try:
                async for message in websocket:
                    data = json.loads(message)
                    cmd = data.get("command")
                    
                    logger.info(f"Received command: {cmd}")
                    
                    try:
                        if cmd == "ping":
                            response = {"status": "success", "message": "pong"}
                        
                        elif cmd == "execute":
                            actions = np.array(data["actions"], dtype=np.float32)
                            self.execute_action_chunk(actions)
                            response = {"status": "success", "message": f"Executed {len(actions)} actions"}
                        
                        elif cmd == "reset":
                            target = np.array(data["joints"], dtype=np.float32)
                            duration = data.get("duration", 2.0)
                            
                            logger.info(f"Resetting to joints over {duration}s...")
                            
                            # Extract arm target (first 14)
                            arm_target = target[:14]
                            
                            # Smooth interpolation for arms
                            current = self.robot.get_current_dual_arm_q()
                            steps = int(duration * 250)
                            
                            for i in range(steps):
                                alpha = (i + 1) / steps
                                interp = current * (1 - alpha) + arm_target * alpha
                                self.robot.ctrl_dual_arm(
                                    q_target=interp,
                                    tauff_target=np.zeros(14)
                                )
                                await asyncio.sleep(1.0 / 250)
                            
                            # Set hand target if provided (indices 14-28)
                            if len(target) >= 28:
                                hand_target = target[14:28]
                                with self.dual_hand_data_lock:
                                    for j in range(14):
                                        self.dual_hand_action_array[j] = hand_target[j]
                            
                            response = {"status": "success", "message": "Reset complete"}
                        
                        elif cmd == "get_state":
                            arm_state = self.robot.get_current_dual_arm_q()
                            with self.dual_hand_data_lock:
                                hand_state = np.array(self.dual_hand_state_array[:], dtype=np.float32)
                            full_state = np.concatenate([arm_state, hand_state])
                            response = {
                                "status": "success",
                                "state": full_state.tolist()
                            }
                        
                        elif cmd == "get_observation":
                            obs = self.get_observation()
                            
                            def img_to_base64(img):
                                from PIL import Image
                                pil_img = Image.fromarray(img)
                                buf = BytesIO()
                                pil_img.save(buf, format="JPEG", quality=85)
                                return base64.b64encode(buf.getvalue()).decode()
                            
                            response = {
                                "status": "success",
                                "state": obs["state"].tolist(),
                                "images": {
                                    "cam_head": img_to_base64(obs["image"]["cam_head"]),
                                }
                            }
                        
                        elif cmd == "emergency_stop":
                            logger.warning("EMERGENCY STOP")
                            current = self.robot.get_current_dual_arm_q()
                            self.robot.ctrl_dual_arm(
                                q_target=current,
                                tauff_target=np.zeros(14)
                            )
                            response = {"status": "success", "message": "Emergency stop"}
                        
                        else:
                            response = {"status": "error", "message": f"Unknown command: {cmd}"}
                    
                    except Exception as e:
                        logger.error(f"Error: {e}", exc_info=True)
                        response = {"status": "error", "message": str(e)}
                    
                    await websocket.send(json.dumps(response))
            
            except websockets.exceptions.ConnectionClosed:
                logger.info("Viz client disconnected")
        
        logger.info(f"Starting command server on 0.0.0.0:{port}")
        logger.info("Waiting for viz client...")
        
        async with serve(handle_command, "0.0.0.0", port):
            await asyncio.Future()  # Run forever

    def cleanup(self):
        """Cleanup resources before exit"""
        logger.info("Cleaning up resources...")
        
        # Stop head camera client
        if self.head_camera_client is not None:
            try:
                self.head_camera_client.running = False
                if hasattr(self, 'head_camera_thread') and self.head_camera_thread.is_alive():
                    self.head_camera_thread.join(timeout=2.0)
            except Exception as e:
                logger.error(f"Error stopping head camera: {e}")
            
            if hasattr(self, 'head_img_shm'):
                try:
                    self.head_img_shm.close()
                    self.head_img_shm.unlink()
                except Exception as e:
                    logger.error(f"Error cleaning up shared memory: {e}")
        
        # Stop hand controller
        if hasattr(self, 'hand_ctrl'):
            try:
                self.hand_ctrl.stop()
            except Exception as e:
                logger.error(f"Error stopping hand controller: {e}")
        
        # Move robot to home position
        if hasattr(self, 'robot') and self.robot is not None:
            try:
                self.robot.ctrl_dual_arm_go_home()
            except Exception as e:
                logger.error(f"Error moving robot to home: {e}")
        
        logger.info("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description="G1 Remote Policy Client")
    parser.add_argument("--server-host", type=str, default="localhost",
                       help="Policy server hostname or IP")
    parser.add_argument("--server-port", type=int, default=8000,
                       help="Policy server port")
    parser.add_argument("--head-camera-server-ip", type=str, default="192.168.123.164",
                       help="Head camera server IP (robot IP)")
    parser.add_argument("--head-camera-server-port", type=int, default=5555,
                       help="Head camera server port")
    parser.add_argument("--prompt", type=str, default="pick up the bottle and put it in the cabinet",
                       help="Task prompt for the policy")
    parser.add_argument("--listen-mode", action="store_true",
                       help="Listen for commands from viz client")
    parser.add_argument("--listen-port", type=int, default=5007,
                       help="Port to listen on in listen mode")
    parser.add_argument("--control-fps", type=int, default=30,
                       help="Control loop frequency in Hz")
    parser.add_argument("--no-motion", action="store_true",
                       help="Disable motion mode (use debug mode)")
    
    args = parser.parse_args()
    
    global controller
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 70)
    print("  G1 Remote Policy Client")
    print("=" * 70)
    
    # Create controller
    controller = G1RemoteClient(
        server_host=args.server_host,
        server_port=args.server_port,
        head_camera_server_ip=args.head_camera_server_ip,
        head_camera_server_port=args.head_camera_server_port,
        prompt=args.prompt,
        control_fps=args.control_fps,
        motion_mode=not args.no_motion,
    )
    
    if args.listen_mode:
        # Listen mode - wait for commands from viz client
        print("\nStarting in LISTEN MODE")
        print(f"  Listening on port {args.listen_port}")
        print(f"  Waiting for commands from viz client...")
        print()
        
        # Move to home position first
        print("  Moving to home position...")
        controller.robot.ctrl_dual_arm_go_home()
        print("  Ready!")
        print()
        
        try:
            asyncio.run(controller.command_server(port=args.listen_port))
        except KeyboardInterrupt:
            print("\n  Interrupted")
    else:
        # Normal mode - connect to policy server
        if not controller.connect_to_policy_server():
            print("Failed to connect to policy server. Exiting.")
            return 1
        
        # Move arms to home position
        print("\n  Moving to home position...")
        controller.robot.ctrl_dual_arm_go_home()
        print("  Ready!")
        
        # Simple control loop
        print("\n  Starting control loop (press Ctrl+C to stop)")
        try:
            while True:
                observation = controller.get_observation()
                policy_response = controller.policy_client.infer(observation)
                action_chunk = policy_response["actions"]
                controller.execute_action_chunk(action_chunk)
        except KeyboardInterrupt:
            print("\n  Control loop interrupted")
    
    # Cleanup
    controller.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
