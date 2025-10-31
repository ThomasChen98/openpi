#!/usr/bin/env python3
"""
H1-2 Remote Policy Client - Inspire Bridge Mode

Connects to OpenPi policy server and executes actions on H1-2 robot with Inspire hands.
Matches xr_teleoperate --inspire-bridge architecture:
    - Hands: Connected to laptop via Modbus (inspire-bridge mode)
    - Cameras: Head (from robot via ZMQ), Wrists (laptop USB)

Camera Architecture:
    - Head camera (ego): Streamed from robot via ZMQ (image_server on robot)
    - Left wrist: RealSense D405 connected to laptop (serial: 218622271789)
    - Right wrist: RealSense D405 connected to laptop (serial: 241222076627)

Camera Logging:
     Initialization: Step-by-step camera setup with detailed status
     Testing: Connection testing with frame validation
     Success: Camera connected with frame statistics
     Warning: Camera issues with diagnostic suggestions
     Error: Failed initialization with cleanup details
     Status: Periodic camera health and performance metrics
     Cleanup: Resource cleanup with final performance summary

Usage:
    # On robot: Start image_server
    python3 image_server.py  # Handles head camera automatically

    # On GPU server: Start OpenPi policy server
    uv run scripts/mock_policy_server.py --port 5006

    # On laptop: Run this client
    python h1_remote_client.py --server-host <gpu-ip> --server-port 5006

Author: Integrated from h1_remote_policy_client.py + xr_teleoperate
"""

import time
import sys
import signal
import argparse
import numpy as np
import pinocchio as pin
import cv2
from multiprocessing import shared_memory
import threading
import os
import asyncio
import json
import logging
from io import BytesIO
import base64
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
    print("ERROR: openpi_client not installed!")
    print("Please install it:")
    print("  cd ../../packages/openpi-client")
    print("  pip install -e .")
    sys.exit(1)

# Local imports - IK solver and robot controller
try:
    from robot_control import H1_2_ArmIK, H1_2_ArmController
except ImportError:
    print("ERROR: Could not import robot_control modules!")
    print("Make sure you're running from the h1_control_client directory")
    sys.exit(1)

# Import camera utilities from xr_teleoperate
try:
    xr_teleoperate_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'xr_teleoperate')
    if os.path.exists(xr_teleoperate_path):
        sys.path.insert(0, xr_teleoperate_path)
    from teleop.image_server.image_client import ImageClient
    from teleop.image_server.image_server import RealSenseCamera
except ImportError as e:
    print(f"ERROR: Could not import camera utilities from xr_teleoperate: {e}")
    print("  Make sure xr_teleoperate is cloned next to openpi/")
    print("  Directory structure should be:")
    print("    Projects/")
    print("      ├── openpi/")
    print("      └── xr_teleoperate/")
    sys.exit(1)

# Global for signal handler
controller = None

def signal_handler(sig, frame):
    global controller
    print("\n  Caught exit signal...")
    if controller is not None:
        controller.cleanup()
    sys.exit(0)


class H1RemoteClient:
    """
    Integrated H1-2 remote policy client - Inspire Bridge Mode
    
    Combines:
    - OpenPi policy server client (gets action chunks)
    - IK solver (converts EE poses to joint angles)  
    - Robot controller (executes joint commands)
    - Inspire hand bridges (hands connected to laptop)
    - Camera streaming (head from robot via ZMQ, wrists from laptop USB)
    """
    
    def __init__(self,
                 server_host: str = "localhost",
                 server_port: int = 5006,
                 network_interface: str = "eno1",
                 left_hand_ip: str = "192.168.123.211",
                 right_hand_ip: str = "192.168.123.210",
                 visualization: bool = False,
                 head_camera_server_ip: str = "192.168.123.163",
                 head_camera_server_port: int = 5555,
                 prompt: str = "bimanual manipulation task"):
        
        self.server_host = server_host
        self.server_port = server_port
        self.policy_client = None
        self.prompt = prompt
        
        # Initialize IK solver
        print(" Initializing IK solver...")
        self.ik_solver = H1_2_ArmIK(
            Unit_Test=False,
            Visualization=visualization,
            Hand_Control=True
        )
        print("   IK solver ready")
        
        # Initialize robot controller
        print(" Initializing robot controller...")
        self.robot = H1_2_ArmController(
            simulation_mode=False,
            hand_control=True,
            left_hand_ip=left_hand_ip,
            right_hand_ip=right_hand_ip,
            network_interface=network_interface
        )
        print("   Robot controller ready")
        
        # Initialize cameras
        print(" Initializing cameras...")
        self._init_head_camera_client(head_camera_server_ip, head_camera_server_port)
        self._init_wrist_cameras()
        
        # Report camera status
        working_cameras = []
        if self.head_img_array is not None:
            working_cameras.append("head")
        if self.left_wrist_camera is not None:
            working_cameras.append("left wrist")
        if self.right_wrist_camera is not None:
            working_cameras.append("right wrist")
        
        if len(working_cameras) == 3:
            print("   Cameras ready: All cameras working!")
        elif len(working_cameras) > 0:
            print(f"   Cameras ready: {', '.join(working_cameras)} working (others using dummy images)")
        else:
            print("   Cameras ready: All using dummy images (policy will still run)")
        
        # Frame counter and timing for performance monitoring
        self.frame_count = 0
        self.camera_status_interval = 50  # Log camera status every 50 frames
        self.session_start_time = time.time()
        self.last_frame_time = time.time()

        # Track camera status changes
        self.last_head_ok = None
        self.last_left_wrist_ok = None
        self.last_right_wrist_ok = None

        # Performance tracking
        self.head_frame_count = 0
        self.left_wrist_frame_count = 0
        self.right_wrist_frame_count = 0
    
    def _init_head_camera_client(self, server_ip: str, server_port: int):
        """Initialize client to receive head camera from robot"""
        print(f"   📹 Initializing head camera client...")
        print(f"      Server: {server_ip}:{server_port}")
        print(f"      Resolution: 480x640 RGB")
        print(f"      Shared memory: {self.head_img_shm.name if hasattr(self, 'head_img_shm') else 'Not created yet'}")

        try:
            # Head camera: single RealSense at 480x640
            self.head_img_shape = (480, 640, 3)

            print(f"      ⏳ Creating shared memory buffer...")
            # Create shared memory buffer for head camera
            self.head_img_shm = shared_memory.SharedMemory(
                create=True,
                size=np.prod(self.head_img_shape) * np.uint8().itemsize
            )
            self.head_img_array = np.ndarray(
                self.head_img_shape, dtype=np.uint8, buffer=self.head_img_shm.buf
            )

            print(f"      ✓ Created shared memory buffer: {self.head_img_shm.name}")
            print(f"         Buffer size: {self.head_img_shm.size} bytes")

            # Initialize image client (receives head camera only)
            print(f"      ⏳ Initializing ImageClient...")
            self.head_camera_client = ImageClient(
                tv_img_shape=self.head_img_shape,
                tv_img_shm_name=self.head_img_shm.name,
                wrist_img_shape=None,  # Wrist cameras are captured directly
                wrist_img_shm_name=None,
                server_address=server_ip,
                port=server_port,
                image_show=False
            )

            print(f"      ✓ Created ImageClient")

            # Start image receiving thread
            print(f"      ⏳ Starting receiver thread...")
            self.head_camera_thread = threading.Thread(
                target=self.head_camera_client.receive_process,
                daemon=True
            )
            self.head_camera_thread.start()

            print(f"      ✓ Started receiver thread (PID: {self.head_camera_thread.ident})")

            # Wait briefly and check if we're receiving frames
            print(f"      ⏳ Testing connection (5 seconds)...")
            for i in range(50):  # 5 seconds at 10fps
                time.sleep(0.1)
                test_frame = self.head_img_array.copy()
                if np.any(test_frame != 0):
                    print(f"      ✓ Connected! Receiving frames from robot")
                    print(f"         Frame shape: {test_frame.shape}, dtype: {test_frame.dtype}")
                    print(f"         Data range: [{test_frame.min()}, {test_frame.max()}]")
                    print(f"         Mean intensity: {test_frame.mean():.1f}")
                    return

            print(f"      ⚠ Connected to server but no frames received yet")
            print(f"         Make sure image_server is running on robot:")
            print(f"         python3 image_server.py --camera-id 0")
            print(f"         Check robot logs for image_server errors")

        except Exception as e:
            print(f"       Failed to initialize head camera: {e}")
            print(f"         Will use dummy image for head camera")
            if hasattr(self, 'head_img_shm'):
                try:
                    self.head_img_shm.close()
                    self.head_img_shm.unlink()
                    print(f"         Cleaned up shared memory")
                except Exception as cleanup_e:
                    print(f"         Warning: Failed to cleanup shared memory: {cleanup_e}")
            self.head_img_array = None
            self.head_camera_client = None
    
    def _init_wrist_cameras(self):
        """Initialize direct USB wrist cameras on laptop (RealSense D405)"""
        print(f"   📹 Initializing wrist cameras...")
        print(f"      Resolution: 480x640 RGB")
        print(f"      Type: RealSense D405 cameras")
        print(f"      Left serial: 218622271789")
        print(f"      Right serial: 241222076627")

        # RealSense camera serial numbers from xr_teleoperate
        left_serial = "218622271789"
        right_serial = "241222076627"

        # Try left wrist camera (RealSense)
        print(f"      ⏳ Testing left wrist camera (RealSense)...")
        try:
            left_cam = RealSenseCamera(
                img_shape=[480, 640],  # height, width
                fps=30,
                serial_number=left_serial
            )
            # Test if we can actually read a frame
            test_frame = left_cam.get_frame()
            if test_frame is None:
                print(f"      ⚠ Left wrist camera (serial: {left_serial}) opened but cannot read frames")
                print(f"         Camera device exists but no video stream")
                print(f"         Will use dummy image for left wrist")
                self.left_wrist_camera = None
            else:
                print(f"      ✓ Left wrist camera connected!")
                print(f"         Frame shape: {test_frame.shape}, dtype: {test_frame.dtype}")
                print(f"         Data range: [{test_frame.min()}, {test_frame.max()}]")
                print(f"         Mean intensity: {test_frame.mean():.1f}")
                print(f"         Serial number: {left_serial}")
                self.left_wrist_camera = left_cam
        except ImportError:
            print(f"       RealSense SDK (pyrealsense2) not available")
            print(f"         Install: pip install pyrealsense2")
            print(f"         Will use dummy image for left wrist")
            self.left_wrist_camera = None
        except Exception as e:
            print(f"       Failed to initialize left wrist camera: {e}")
            print(f"         Check if RealSense camera is connected and serial number is correct")
            print(f"         Serial number: {left_serial}")
            print(f"         Will use dummy image for left wrist")
            self.left_wrist_camera = None

        # Try right wrist camera (RealSense)
        print(f"      ⏳ Testing right wrist camera (RealSense)...")
        try:
            right_cam = RealSenseCamera(
                img_shape=[480, 640],  # height, width
                fps=30,
                serial_number=right_serial
            )
            # Test if we can actually read a frame
            test_frame = right_cam.get_frame()
            if test_frame is None:
                print(f"      ⚠ Right wrist camera (serial: {right_serial}) opened but cannot read frames")
                print(f"         Camera device exists but no video stream")
                print(f"         Will use dummy image for right wrist")
                self.right_wrist_camera = None
            else:
                print(f"      ✓ Right wrist camera connected!")
                print(f"         Frame shape: {test_frame.shape}, dtype: {test_frame.dtype}")
                print(f"         Data range: [{test_frame.min()}, {test_frame.max()}]")
                print(f"         Mean intensity: {test_frame.mean():.1f}")
                print(f"         Serial number: {right_serial}")
                self.right_wrist_camera = right_cam
        except ImportError:
            print(f"       RealSense SDK (pyrealsense2) not available")
            print(f"         Install: pip install pyrealsense2")
            print(f"         Will use dummy image for right wrist")
            self.right_wrist_camera = None
        except Exception as e:
            print(f"       Failed to initialize right wrist camera: {e}")
            print(f"         Check if RealSense camera is connected and serial number is correct")
            print(f"         Serial number: {right_serial}")
            print(f"         Will use dummy image for right wrist")
            self.right_wrist_camera = None
    
    def connect_to_policy_server(self):
        """Connect to the remote OpenPi policy server"""
        try:
            print(f" Connecting to policy server at {self.server_host}:{self.server_port}...")
            self.policy_client = websocket_client_policy.WebsocketClientPolicy(
                host=self.server_host,
                port=self.server_port
            )
            
            # Get server metadata
            metadata = self.policy_client.get_server_metadata()
            print(f"   Connected! Server metadata:")
            print(f"     Action dim: {metadata.get('action_dim', 'N/A')}")
            print(f"     Action horizon: {metadata.get('action_horizon', 'N/A')}")
            print(f"     Description: {metadata.get('description', 'N/A')}")
            
            return True
            
        except Exception as e:
            print(f"   Failed to connect to policy server: {e}")
            print(f"   Make sure server is running:")
            print(f"   uv run scripts/mock_policy_server.py --port {self.server_port}")
            return False
    
    def get_observation(self) -> dict:
        """
        Construct observation for policy inference with real camera feeds
        Uses dummy images for any cameras that failed to initialize.
        """
        # Get current arm joint positions (14 DOF)
        current_arm_q = self.robot.get_current_dual_arm_q()
        
        # Create dummy image (224x224 RGB, gray)
        dummy_image = np.full((224, 224, 3), 128, dtype=np.uint8)
        
        # Track camera statuses
        head_ok = False
        left_wrist_ok = False
        right_wrist_ok = False
        
        # Get head camera (from robot via ZMQ)
        if self.head_img_array is not None:
            try:
                head_image = self.head_img_array.copy()
                # Check if we're actually receiving frames (not all zeros)
                if np.any(head_image != 0):
                    base_image = cv2.resize(head_image, (224, 224))
                    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
                    base_image = image_tools.convert_to_uint8(base_image)
                    head_ok = True
                else:
                    # Shared memory initialized but no frames received yet
                    base_image = dummy_image
            except Exception as e:
                print(f"   WARNING: Failed to get head camera frame: {e}")
                base_image = dummy_image
        else:
            base_image = dummy_image
        
        # Get left wrist camera (directly from laptop)
        if self.left_wrist_camera is not None:
            try:
                left_wrist_frame = self.left_wrist_camera.get_frame()
                if left_wrist_frame is not None:
                    left_wrist_image = cv2.resize(left_wrist_frame, (224, 224))
                    left_wrist_image = cv2.cvtColor(left_wrist_image, cv2.COLOR_BGR2RGB)
                    left_wrist_image = image_tools.convert_to_uint8(left_wrist_image)
                    left_wrist_ok = True
                else:
                    left_wrist_image = dummy_image
            except Exception as e:
                print(f"   WARNING: Failed to get left wrist frame: {e}")
                left_wrist_image = dummy_image
        else:
            left_wrist_image = dummy_image
        
        # Get right wrist camera (directly from laptop)
        if self.right_wrist_camera is not None:
            try:
                right_wrist_frame = self.right_wrist_camera.get_frame()
                if right_wrist_frame is not None:
                    right_wrist_image = cv2.resize(right_wrist_frame, (224, 224))
                    right_wrist_image = cv2.cvtColor(right_wrist_image, cv2.COLOR_BGR2RGB)
                    right_wrist_image = image_tools.convert_to_uint8(right_wrist_image)
                    right_wrist_ok = True
                else:
                    right_wrist_image = dummy_image
            except Exception as e:
                print(f"   WARNING: Failed to get right wrist frame: {e}")
                right_wrist_image = dummy_image
        else:
            right_wrist_image = dummy_image
        
        # Detect camera status changes (cameras going down)
        if self.last_head_ok is not None and self.last_head_ok and not head_ok:
            print(f"    WARNING: Head camera stopped sending frames (now using dummy images)")
            print(f"      This usually means the image_server on the robot crashed or network issues")
            print(f"      Check robot logs: journalctl -u image_server -f")
        if self.last_left_wrist_ok is not None and self.last_left_wrist_ok and not left_wrist_ok:
            print(f"    WARNING: Left wrist camera stopped sending frames (now using dummy images)")
            print(f"      Camera device may have been unplugged or driver crashed")
            print(f"      Check device: ls /dev/video* and dmesg | grep video")
        if self.last_right_wrist_ok is not None and self.last_right_wrist_ok and not right_wrist_ok:
            print(f"    WARNING: Right wrist camera stopped sending frames (now using dummy images)")
            print(f"      Camera device may have been unplugged or driver crashed")
            print(f"      Check device: ls /dev/video* and dmesg | grep video")
        
        # Update last status
        self.last_head_ok = head_ok
        self.last_left_wrist_ok = left_wrist_ok
        self.last_right_wrist_ok = right_wrist_ok
        
        # Update frame counters and timing
        self.frame_count += 1
        current_time = time.time()

        if head_ok:
            self.head_frame_count += 1
        if left_wrist_ok:
            self.left_wrist_frame_count += 1
        if right_wrist_ok:
            self.right_wrist_frame_count += 1

        # Periodic camera status logging with performance metrics
        if self.frame_count % self.camera_status_interval == 0:
            elapsed_time = current_time - self.session_start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0

            # Calculate individual camera FPS
            head_fps = self.head_frame_count / elapsed_time if elapsed_time > 0 else 0
            left_fps = self.left_wrist_frame_count / elapsed_time if elapsed_time > 0 else 0
            right_fps = self.right_wrist_frame_count / elapsed_time if elapsed_time > 0 else 0

            status_parts = []
            if head_ok:
                status_parts.append(f"✓ Head(REAL:{head_fps:.1f}fps)")
            else:
                status_parts.append("✗ Head(DUMMY)")
            if left_wrist_ok:
                status_parts.append(f"✓ L_wrist(REAL:{left_fps:.1f}fps)")
            else:
                status_parts.append("✗ L_wrist(DUMMY)")
            if right_wrist_ok:
                status_parts.append(f"✓ R_wrist(REAL:{right_fps:.1f}fps)")
            else:
                status_parts.append("✗ R_wrist(DUMMY)")

            print(f" 📷 Camera status: {' | '.join(status_parts)}")
            print(f"    Overall: {fps:.1f}fps | Total frames: {self.frame_count}")
            print(f"    Head: {self.head_frame_count} frames | Left: {self.left_wrist_frame_count} | Right: {self.right_wrist_frame_count}")

            # Reset counters for next interval
            self.head_frame_count = 0
            self.left_wrist_frame_count = 0
            self.right_wrist_frame_count = 0
            self.session_start_time = current_time
        
        # OpenPi model expects this structure
        observation = {
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": True,
                "left_wrist_0_rgb": True,
                "right_wrist_0_rgb": True,
            },
            "state": current_arm_q,  # 14 arm joints
            "prompt": self.prompt,
        }
        
        return observation
    
    def policy_actions_to_ee_poses(self, policy_actions: np.ndarray):
        """
        Convert policy action chunk to end-effector poses
        
        Policy actions are in joint space (51 DOF), but we only use arm joints (14 DOF).
        Your IK solver then converts these to EE poses if needed.
        
        Args:
            policy_actions: (action_horizon, 51) array
            
        Returns:
            List of (left_ee_pose, right_ee_pose, left_hand, right_hand) tuples
        """
        # Extract arm actions - MATCH VISUALIZATION CLIENT BEHAVIOR
        # The policy outputs 51-dim actions, but the first 14 are the arm joints
        # (same as how viz client uses them in extract_hand_joints_for_urdf)
        
        action_sequence = []
        for action in policy_actions:
            # Extract arm joint commands (14 DOF: 7 per arm)
            # Take first 14 dimensions directly (matches visualization)
            arm_joints = action[13:27]
            
            # Extract hand commands from URDF action space (dims 27-50)
            # Only extract actuated joints (6 per hand), skip mimic joints
            # Left hand actuated: 27 (index_1), 29 (little_1), 31 (middle_1), 
            #                     33 (ring_1), 35 (thumb_1), 36 (thumb_2)
            # Right hand actuated: 39 (index_1), 41 (little_1), 43 (middle_1),
            #                      45 (ring_1), 47 (thumb_1), 48 (thumb_2)
            # Inspire hand order: [little, ring, middle, index, thumb_2(bend), thumb_1(rotation)]
            if len(action) >= 51:
                left_hand = np.array([
                    action[29],  # little_1
                    action[33],  # ring_1
                    action[31],  # middle_1
                    action[27],  # index_1
                    action[36],  # thumb_2 (bend)
                    action[35],  # thumb_1 (rotation)
                ])
                right_hand = np.array([
                    action[41],  # little_1
                    action[45],  # ring_1
                    action[43],  # middle_1
                    action[39],  # index_1
                    action[48],  # thumb_2 (bend)
                    action[47],  # thumb_1 (rotation)
                ])
            else:
                left_hand = np.zeros(6) + 1000  # Default open hand
                right_hand = np.zeros(6) + 1000 # Default open hand
            
            action_sequence.append({
                'arm_joints': arm_joints,
                'left_hand': left_hand,
                'right_hand': right_hand
            })
        
        return action_sequence
    
    def execute_action_chunk(self, policy_actions: np.ndarray):
        """
        Execute a chunk of policy actions on the robot
        
        Args:
            policy_actions: (50, 51) array of actions
        """
        action_sequence = self.policy_actions_to_ee_poses(policy_actions)
        
        print(f"   Executing {len(action_sequence)} actions...")
        
        for i, action in enumerate(action_sequence):
            arm_joints = action['arm_joints']
            left_hand = action['left_hand']
            right_hand = action['right_hand']
            
            # Send arm + hand commands to robot
            self.robot.ctrl_dual_arm(
                q_target=arm_joints,
                tauff_target=np.zeros(14),  # No feedforward torque
                left_hand_gesture=left_hand,   # 6 DOF per hand (0-1000 range)
                right_hand_gesture=right_hand
            )
            
            # Control at 50Hz (policy provides actions at 10Hz, interpolate 5 steps each)
            time.sleep(0.02)
    
    def run_control_loop(self, duration: float = 10.0):
        """
        Main control loop - queries policy and executes actions
        
        Args:
            duration: How long to run (seconds). Use inf for infinite.
        """
        print(f"\n   Starting control loop (duration: {duration}s)")
        print("   Press Ctrl+C to stop\n")
        
        start_time = time.time()
        
        try:
            while True:
                # Check if duration exceeded
                if time.time() - start_time > duration:
                    print("\n   Control loop duration reached")
                    break
                
                # Get current observation (includes camera images + robot state)
                observation = self.get_observation()
                
                # Query policy for action chunk
                policy_response = self.policy_client.infer(observation)
                action_chunk = policy_response["actions"]  # Shape: (50, 51)
                
                # Execute action chunk
                self.execute_action_chunk(action_chunk)
                
        except KeyboardInterrupt:
            print("\n  Control loop interrupted by user")
    
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
            logger.info(f"🔌 Viz client connected from {websocket.remote_address}")
            
            try:
                async for message in websocket:
                    data = json.loads(message)
                    cmd = data.get("command")
                    
                    logger.info(f"📨 Received: {cmd}")
                    
                    try:
                        if cmd == "ping":
                            # Health check
                            response = {"status": "success", "message": "pong"}
                        
                        elif cmd == "execute":
                            # Execute action chunk
                            actions = np.array(data["actions"], dtype=np.float32)
                            logger.info(f"🚀 Executing {len(actions)} actions...")
                            
                            self.execute_action_chunk(actions)
                            
                            response = {"status": "success", "message": f"Executed {len(actions)} actions"}
                        
                        elif cmd == "reset":
                            # Reset to joint positions
                            target = np.array(data["joints"], dtype=np.float32)
                            duration = data.get("duration", 2.0)
                            
                            logger.info(f"🔄 Resetting to joints over {duration}s...")
                            
                            # Smooth interpolation
                            current = self.robot.get_current_dual_arm_q()
                            steps = int(duration * 250)
                            
                            for i in range(steps):
                                alpha = (i + 1) / steps
                                interp = current * (1 - alpha) + target * alpha
                                self.robot.ctrl_dual_arm(
                                    q_target=interp,
                                    tauff_target=np.zeros(14)
                                )
                                await asyncio.sleep(1.0 / 250)
                            
                            response = {"status": "success", "message": "Reset complete"}
                        
                        elif cmd == "get_state":
                            # Get current state
                            state = self.robot.get_current_dual_arm_q()
                            response = {
                                "status": "success",
                                "state": state.tolist()
                            }
                        
                        elif cmd == "emergency_stop":
                            # Emergency stop
                            logger.warning(" EMERGENCY STOP")
                            current = self.robot.get_current_dual_arm_q()
                            self.robot.ctrl_dual_arm(
                                q_target=current,
                                tauff_target=np.zeros(14)
                            )
                            response = {"status": "success", "message": "Emergency stop"}
                        
                        elif cmd == "get_observation":
                            # Get live observation (images + state)
                            logger.info("📷 Getting observation...")
                            obs = self.get_observation()
                            
                            # Encode images as base64
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
                                    "base_0_rgb": img_to_base64(obs["image"]["base_0_rgb"]),
                                    "left_wrist_0_rgb": img_to_base64(obs["image"]["left_wrist_0_rgb"]),
                                    "right_wrist_0_rgb": img_to_base64(obs["image"]["right_wrist_0_rgb"]),
                                }
                            }
                            
                            logger.info(" Observation sent")
                        
                        else:
                            response = {"status": "error", "message": f"Unknown command: {cmd}"}
                    
                    except Exception as e:
                        logger.error(f" Error: {e}", exc_info=True)
                        response = {"status": "error", "message": str(e)}
                    
                    await websocket.send(json.dumps(response))
            
            except websockets.exceptions.ConnectionClosed:
                logger.info("🔌 Viz client disconnected")
        
        logger.info(f" Starting command server on 0.0.0.0:{port}")
        logger.info(" Waiting for viz client...")
        logger.info("")
        logger.info("Next steps:")
        logger.info(f"  1. SSH with reverse tunnel: ssh -R {port}:localhost:{port} -L 8081:localhost:8081 yuxin@msc-server")
        logger.info("  2. On remote: python h1_policy_viz_client.py --robot-execution")
        logger.info("  3. Open browser: http://localhost:8081")
        logger.info("")
        
        async with serve(handle_command, "0.0.0.0", port):
            await asyncio.Future()  # Run forever
    
    def cleanup(self):
        """Cleanup resources before exit"""
        print("\n    Cleaning up resources...")

        # Stop head camera client
        print("   📹 Stopping head camera client...")
        if hasattr(self, 'head_camera_client') and self.head_camera_client is not None:
            try:
                print(f"      Setting running=False for ImageClient...")
                self.head_camera_client.running = False
                if hasattr(self, 'head_camera_thread') and self.head_camera_thread.is_alive():
                    print(f"      Waiting for receiver thread (timeout: 2s)...")
                    self.head_camera_thread.join(timeout=2.0)
                    if self.head_camera_thread.is_alive():
                        print(f"      ⚠ Thread didn't stop gracefully, continuing...")
                    else:
                        print(f"      ✓ Receiver thread stopped")
                else:
                    print(f"      ✓ Receiver thread already stopped")
            except Exception as e:
                print(f"       Error stopping head camera: {e}")

            # Cleanup shared memory
            if hasattr(self, 'head_img_shm'):
                try:
                    print(f"      Closing shared memory: {self.head_img_shm.name}")
                    self.head_img_shm.close()
                    print(f"      Unlinking shared memory...")
                    self.head_img_shm.unlink()
                    print(f"      ✓ Shared memory cleaned up")
                except Exception as e:
                    print(f"       Error cleaning up shared memory: {e}")
        else:
            print("      ✓ Head camera already stopped")

        # Stop wrist cameras
        print("   📹 Stopping wrist cameras...")
        if hasattr(self, 'left_wrist_camera') and self.left_wrist_camera is not None:
            try:
                print(f"      Releasing left wrist camera...")
                self.left_wrist_camera.release()
                print(f"      ✓ Left wrist camera released")
            except Exception as e:
                print(f"       Error releasing left wrist camera: {e}")
        else:
            print("      ✓ Left wrist camera already released")

        if hasattr(self, 'right_wrist_camera') and self.right_wrist_camera is not None:
            try:
                print(f"      Releasing right wrist camera...")
                self.right_wrist_camera.release()
                print(f"      ✓ Right wrist camera released")
            except Exception as e:
                print(f"       Error releasing right wrist camera: {e}")
        else:
            print("      ✓ Right wrist camera already released")

        # Stop robot control
        if self.robot:
            print("    Stopping robot controller...")
            try:
                print(f"      Moving robot to home position...")
                self.robot.ctrl_dual_arm_go_home()
                print(f"      ✓ Robot moved to home position")
                if hasattr(self.robot, 'stop_hand_bridges'):
                    print(f"      Stopping hand bridges...")
                    self.robot.stop_hand_bridges()
                    print(f"      ✓ Hand bridges stopped")
            except Exception as e:
                print(f"       Error stopping robot: {e}")
        else:
            print("    Robot controller already stopped")

        # Print final performance statistics
        if self.frame_count > 0:
            total_time = time.time() - self.session_start_time
            if total_time > 0:
                avg_fps = self.frame_count / total_time
                print(f"    Performance summary:")
                print(f"      Total frames processed: {self.frame_count}")
                print(f"      Total runtime: {total_time:.2f}s")
                print(f"      Average FPS: {avg_fps:.2f}")
                print(f"      Head camera frames: {self.head_frame_count}")
                print(f"      Left wrist frames: {self.left_wrist_frame_count}")
                print(f"      Right wrist frames: {self.right_wrist_frame_count}")

        print("    Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description="H1-2 Remote Policy Client")
    parser.add_argument("--server-host", type=str, default="localhost",
                       help="Policy server hostname or IP")
    parser.add_argument("--server-port", type=int, default=5006,
                       help="Policy server port (default: 5006)")
    parser.add_argument("--network-interface", type=str, default="eno1",
                       help="Network interface for hand bridges")
    parser.add_argument("--left-hand-ip", type=str, default="192.168.123.211",
                       help="Left hand IP address")
    parser.add_argument("--right-hand-ip", type=str, default="192.168.123.210",
                       help="Right hand IP address")
    parser.add_argument("--head-camera-server-ip", type=str, default="192.168.123.163",
                       help="Head camera server IP (robot IP where image_server runs)")
    parser.add_argument("--head-camera-server-port", type=int, default=5555,
                       help="Head camera server port")
    parser.add_argument("--visualization", action="store_true",
                       help="Enable IK visualization (meshcat)")
    parser.add_argument("--duration", type=float, default=float('inf'),
                       help="Control loop duration in seconds (default: infinite)")
    parser.add_argument("--prompt", type=str, default="bimanual manipulation task",
                       help="Task prompt for the policy")
    parser.add_argument("--listen-mode", action="store_true",
                       help="Listen for commands from viz client instead of running control loop")
    parser.add_argument("--listen-port", type=int, default=5007,
                       help="Port to listen on in listen mode (default: 5007)")
    
    args = parser.parse_args()
    
    global controller
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 70)
    print("  H1-2 Remote Policy Client")
    print("=" * 70)
    
    # Create controller
    controller = H1RemoteClient(
        server_host=args.server_host,
        server_port=args.server_port,
        network_interface=args.network_interface,
        left_hand_ip=args.left_hand_ip,
        right_hand_ip=args.right_hand_ip,
        head_camera_server_ip=args.head_camera_server_ip,
        head_camera_server_port=args.head_camera_server_port,
        visualization=args.visualization,
        prompt=args.prompt
    )
    
    if args.listen_mode:
        # Listen mode - wait for commands from viz client
        print("\n🎧 Starting in LISTEN MODE")
        print(f"   Listening on port {args.listen_port}")
        print(f"   Waiting for commands from viz client...")
        print()
        
        # Move to home position first
        print("   Moving to home position...")
        controller.robot.ctrl_dual_arm_go_home()
        print("   Ready!")
        print()
        
        try:
            asyncio.run(controller.command_server(port=args.listen_port))
        except KeyboardInterrupt:
            print("\n  Interrupted")
    else:
        # Normal mode - connect to policy server
        if not controller.connect_to_policy_server():
            print(" Failed to connect to policy server. Exiting.")
            return 1
        
        # Move arms to home position
        print("\n   Moving to home position...")
        controller.robot.ctrl_dual_arm_go_home()
        print("   Ready!")
        
        # Run control loop
        controller.run_control_loop(duration=args.duration)
    
    # Cleanup
    controller.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
