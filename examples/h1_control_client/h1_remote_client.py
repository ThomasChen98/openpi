#!/usr/bin/env python3
"""
H1-2 Remote Policy Client - Self-Contained Edition

Connects to OpenPi policy server and executes actions on H1-2 robot with arms only.
This is a complete, self-contained solution for H1-2 control.

Usage:
    # Start OpenPi server (on GPU machine):
    uv run scripts/mock_policy_server.py --port 8000
    
    # Run this client (on H1-2 control laptop):
    python h1_remote_client.py --server-host <gpu-server-ip> --server-port 8000

Author: Integrated from h1_remote_policy_client.py
"""

import time
import sys
import signal
import argparse
import numpy as np
import pinocchio as pin

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

# Global for signal handler
controller = None

def signal_handler(sig, frame):
    global controller
    print("\n⏹️  Caught exit signal...")
    if controller is not None:
        controller.cleanup()
    sys.exit(0)


class H1RemoteClient:
    """
    Integrated H1-2 remote policy client
    
    Combines:
    - OpenPi policy server client (gets action chunks)
    - IK solver (converts EE poses to joint angles)  
    - Robot controller (executes joint commands)
    """
    
    def __init__(self, 
                 server_host: str = "localhost",
                 server_port: int = 8000,
                 network_interface: str = "eno1",
                 left_hand_ip: str = "192.168.123.211",
                 right_hand_ip: str = "192.168.123.210",
                 visualization: bool = False):
        
        self.server_host = server_host
        self.server_port = server_port
        self.policy_client = None
        
        # Initialize IK solver
        print("🔧 Initializing IK solver...")
        self.ik_solver = H1_2_ArmIK(
            Unit_Test=False,
            Visualization=visualization,
            Hand_Control=True
        )
        print("✅ IK solver ready")
        
        # Initialize robot controller
        print("🤖 Initializing robot controller...")
        self.robot = H1_2_ArmController(
            simulation_mode=False,
            hand_control=True,
            left_hand_ip=left_hand_ip,
            right_hand_ip=right_hand_ip,
            network_interface=network_interface
        )
        print("✅ Robot controller ready")
        
        # Dummy camera images (TODO: replace with real cameras)
        self.dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def connect_to_policy_server(self):
        """Connect to the remote OpenPi policy server"""
        try:
            print(f"🔌 Connecting to policy server at {self.server_host}:{self.server_port}...")
            self.policy_client = websocket_client_policy.WebsocketClientPolicy(
                host=self.server_host,
                port=self.server_port,
            )
            
            metadata = self.policy_client.get_server_metadata()
            print(f"✅ Connected to policy server!")
            print(f"📊 Server metadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to policy server: {e}")
            print(f"   Make sure server is running:")
            print(f"   uv run scripts/mock_policy_server.py --port {self.server_port}")
            return False
    
    def get_observation(self) -> dict:
        """
        Construct observation for policy inference
        
        TODO: Replace dummy images with real camera feeds
        """
        # Get current arm joint positions (14 DOF)
        current_arm_q = self.robot.get_current_dual_arm_q()
        
        observation = {
            "observation/image": self.dummy_image,  # TODO: external camera
            "observation/wrist_image": self.dummy_image,  # TODO: wrist camera
            "observation/state": current_arm_q,  # 14 arm joints
            "prompt": "bimanual manipulation task",  # TODO: task from user
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
        
        print(f"🎯 Executing {len(action_sequence)} actions...")
        
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
        print(f"\n🚀 Starting control loop (duration: {duration}s)")
        print("   Press Ctrl+C to stop\n")
        
        start_time = time.time()
        iteration = 0
        
        try:
            while (time.time() - start_time) < duration:
                iteration += 1
                
                # 1. Get observation
                observation = self.get_observation()
                
                # 2. Query policy server
                try:
                    result = self.policy_client.infer(observation)
                    action_chunk = result["actions"]  # Shape: (50, 51)
                    
                    # Log timing
                    if "server_timing" in result:
                        inference_time = result["server_timing"].get("infer_ms", 0)
                        print(f"[{iteration}] Policy inference: {inference_time:.1f}ms")
                    
                except Exception as e:
                    print(f"❌ Policy query failed: {e}")
                    continue
                
                # 3. Execute action chunk
                self.execute_action_chunk(action_chunk)
                
        except KeyboardInterrupt:
            print("\n⏹️  Control loop interrupted by user")
    
    def cleanup(self):
        """Cleanup resources before exit"""
        print("\n🧹 Cleaning up...")
        
        # Stop robot control
        if self.robot:
            print("  Stopping robot controller...")
            self.robot.ctrl_dual_arm_go_home()
            if hasattr(self.robot, 'stop_hand_bridges'):
                self.robot.stop_hand_bridges()
        
        print("✅ Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description="H1-2 Remote Policy Client")
    parser.add_argument("--server-host", type=str, default="localhost",
                       help="Policy server hostname or IP")
    parser.add_argument("--server-port", type=int, default=8000,
                       help="Policy server port")
    parser.add_argument("--network-interface", type=str, default="eno1",
                       help="Network interface for hand bridges")
    parser.add_argument("--left-hand-ip", type=str, default="192.168.123.211",
                       help="Left hand IP address")
    parser.add_argument("--right-hand-ip", type=str, default="192.168.123.210",
                       help="Right hand IP address")
    parser.add_argument("--visualization", action="store_true",
                       help="Enable IK visualization (meshcat)")
    parser.add_argument("--duration", type=float, default=float('inf'),
                       help="Control loop duration in seconds (default: infinite)")
    
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
        visualization=args.visualization
    )
    
    # Connect to policy server
    if not controller.connect_to_policy_server():
        print("❌ Failed to connect to policy server. Exiting.")
        return 1
    
    # Move arms to home position
    print("\n🏠 Moving to home position...")
    controller.robot.ctrl_dual_arm_go_home()
    print("✅ Ready!")
    
    # Run control loop
    controller.run_control_loop(duration=args.duration)
    
    # Cleanup
    controller.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())

