#!/usr/bin/env python3
"""
H1-2 Remote Policy Client - Self-Contained Edition

Connects to OpenPi policy server and executes actions on H1-2 robot with arms only.
This is a complete, self-contained solution for H1-2 control.

Camera Architecture:
    - Head camera (ego): Streamed from robot via ZMQ (image_server on robot)
    - Wrist cameras: Directly connected to laptop via USB (/dev/video2, /dev/video4)

Usage:
    # On robot: Start head camera server
    python <run image server script for head camera>
    
    # On GPU: Start OpenPi policy server
    uv run scripts/mock_policy_server.py --port 5006
    
    # On laptop: Run this client
    python h1_remote_client.py --server-host <gpu-ip> --server-port 5006

Author: Integrated from h1_remote_policy_client.py + camera integration
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
    from teleop.image_server.image_server import OpenCVCamera
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
    Integrated H1-2 remote policy client with camera streaming
    
    Combines:
    - OpenPi policy server client (gets action chunks)
    - IK solver (converts EE poses to joint angles)  
    - Robot controller (executes joint commands)
    - Camera streaming (head from robot, wrists from laptop)
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
                 left_wrist_camera_id: int = 2,
                 right_wrist_camera_id: int = 4,
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
        self._init_wrist_cameras(left_wrist_camera_id, right_wrist_camera_id)
        
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
    
    def _init_head_camera_client(self, server_ip: str, server_port: int):
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
            
            # Initialize image client (receives head camera only)
            self.head_camera_client = ImageClient(
                tv_img_shape=self.head_img_shape,
                tv_img_shm_name=self.head_img_shm.name,
                wrist_img_shape=None,  # Wrist cameras are captured directly
                wrist_img_shm_name=None,
                server_address=server_ip,
                port=server_port,
                image_show=False
            )
            
            # Start image receiving thread
            self.head_camera_thread = threading.Thread(
                target=self.head_camera_client.receive_process,
                daemon=True
            )
            self.head_camera_thread.start()
            print(f"   Head camera client connected to {server_ip}:{server_port}")
            
        except Exception as e:
            print(f"   WARNING: Failed to initialize head camera: {e}")
            print(f"            Will use dummy image for head camera")
            self.head_img_array = None
            self.head_camera_client = None
    
    def _init_wrist_cameras(self, left_id: int, right_id: int):
        """Initialize direct USB wrist cameras on laptop"""
        # Wrist cameras: OpenCV cameras connected to laptop
        # /dev/video2 (left) and /dev/video4 (right)
        
        # Try left wrist camera
        try:
            left_cam = OpenCVCamera(
                device_id=left_id,
                img_shape=[480, 640],  # height, width
                fps=30
            )
            # Test if we can actually read a frame
            test_frame = left_cam.get_frame()
            if test_frame is None:
                print(f"   WARNING: Left wrist camera /dev/video{left_id} opened but cannot read frames")
                print(f"            Will use dummy image for left wrist")
                left_cam.release()
                self.left_wrist_camera = None
            else:
                self.left_wrist_camera = left_cam
                print(f"   Left wrist camera: /dev/video{left_id}")
        except Exception as e:
            print(f"   WARNING: Failed to initialize left wrist camera (/dev/video{left_id}): {e}")
            print(f"            Will use dummy image for left wrist")
            self.left_wrist_camera = None
        
        # Try right wrist camera
        try:
            right_cam = OpenCVCamera(
                device_id=right_id,
                img_shape=[480, 640],  # height, width
                fps=30
            )
            # Test if we can actually read a frame
            test_frame = right_cam.get_frame()
            if test_frame is None:
                print(f"   WARNING: Right wrist camera /dev/video{right_id} opened but cannot read frames")
                print(f"            Will use dummy image for right wrist")
                right_cam.release()
                self.right_wrist_camera = None
            else:
                self.right_wrist_camera = right_cam
                print(f"   Right wrist camera: /dev/video{right_id}")
        except Exception as e:
            print(f"   WARNING: Failed to initialize right wrist camera (/dev/video{right_id}): {e}")
            print(f"            Will use dummy image for right wrist")
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
        
        # Get head camera (from robot via ZMQ)
        if self.head_img_array is not None:
            try:
                head_image = self.head_img_array.copy()
                base_image = cv2.resize(head_image, (224, 224))
                base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
                base_image = image_tools.convert_to_uint8(base_image)
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
                else:
                    right_wrist_image = dummy_image
            except Exception as e:
                print(f"   WARNING: Failed to get right wrist frame: {e}")
                right_wrist_image = dummy_image
        else:
            right_wrist_image = dummy_image
        
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
        # Extract arm actions (dimensions 13-26 = 14 arm joints)
        # Note: Policy uses full URDF (0-26=robot, 27-50=hands)
        # But H1_2_ArmController only controls arms (indices 13-26 in full robot)
        
        # For now, we'll directly use the arm joint commands from policy
        # If your policy outputs EE poses instead, you'd convert here
        
        action_sequence = []
        for action in policy_actions:
            # Extract arm joint commands (14 DOF: 7 per arm)
            arm_joints = action[13:27]  # indices 13-26 from full 51-DOF action
            
            # Extract hand commands if present
            left_hand = action[27:39] if len(action) > 27 else np.zeros(12)
            right_hand = action[39:51] if len(action) > 39 else np.zeros(12)
            
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
            
            # Use IK solver to get smooth trajectory (optional - or use joints directly)
            # For now, send joint commands directly
            self.robot.ctrl_dual_arm(
                q_target=arm_joints,
                tauff_target=np.zeros(14),  # No feedforward torque
                # Note: Hand control would go here if policy provides hand actions
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
    
    def cleanup(self):
        """Cleanup resources before exit"""
        print("\n   Cleaning up...")
        
        # Stop head camera client
        print("  Stopping head camera client...")
        if hasattr(self, 'head_camera_client') and self.head_camera_client is not None:
            try:
                self.head_camera_client.running = False
                if hasattr(self, 'head_camera_thread') and self.head_camera_thread.is_alive():
                    self.head_camera_thread.join(timeout=2.0)
            except Exception as e:
                print(f"    Warning: Error stopping head camera: {e}")
            
            # Cleanup shared memory
            if hasattr(self, 'head_img_shm'):
                try:
                    self.head_img_shm.close()
                    self.head_img_shm.unlink()
                except Exception as e:
                    print(f"    Warning: Error cleaning up shared memory: {e}")
        
        # Stop wrist cameras
        print("  Stopping wrist cameras...")
        if hasattr(self, 'left_wrist_camera') and self.left_wrist_camera is not None:
            try:
                self.left_wrist_camera.release()
            except Exception as e:
                print(f"    Warning: Error releasing left wrist camera: {e}")
        if hasattr(self, 'right_wrist_camera') and self.right_wrist_camera is not None:
            try:
                self.right_wrist_camera.release()
            except Exception as e:
                print(f"    Warning: Error releasing right wrist camera: {e}")
        
        # Stop robot control
        if self.robot:
            print("  Stopping robot controller...")
            try:
                self.robot.ctrl_dual_arm_go_home()
                if hasattr(self.robot, 'stop_hand_bridges'):
                    self.robot.stop_hand_bridges()
            except Exception as e:
                print(f"    Warning: Error stopping robot: {e}")
        
        print("   Cleanup complete")


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
    parser.add_argument("--left-wrist-camera-id", type=int, default=2,
                       help="Left wrist camera device ID (/dev/video<id>)")
    parser.add_argument("--right-wrist-camera-id", type=int, default=4,
                       help="Right wrist camera device ID (/dev/video<id>)")
    parser.add_argument("--visualization", action="store_true",
                       help="Enable IK visualization (meshcat)")
    parser.add_argument("--duration", type=float, default=float('inf'),
                       help="Control loop duration in seconds (default: infinite)")
    parser.add_argument("--prompt", type=str, default="bimanual manipulation task",
                       help="Task prompt for the policy")
    
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
        left_wrist_camera_id=args.left_wrist_camera_id,
        right_wrist_camera_id=args.right_wrist_camera_id,
        visualization=args.visualization,
        prompt=args.prompt
    )
    
    # Connect to policy server
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
