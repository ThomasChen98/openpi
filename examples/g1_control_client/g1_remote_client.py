#!/usr/bin/env python3
"""
G1 Remote Policy Client - Direct Joint Control Mode

Connects to OpenPi policy server and executes actions on G1 robot with Dex3 hands.
Uses DIRECT JOINT ANGLE control (no VR retargeting) for both arms and hands.

Architecture:
    - Arms: Controlled via DDS (unitree_sdk2py) - direct joint angles
    - Hands: Dex3 hands controlled via DDS - direct joint angles
    - Waist: Direct yaw control via DDS
    - Locomotion: LocoClient for Move commands (wireless controller forwarding only)
    - Camera: Head camera from robot via ZMQ (image_server)

State Space (29 dims):
    [0:28]  qpos        - arm (14) + hand (14) joint positions
    [28]    waist_yaw   - waist yaw joint position

Action Space (29 dims):
    [0:28]  upper_body  - arm (14) + hand (14) joint targets
    [28]    waist_yaw   - waist yaw joint target

Locomotion Control:
    - Wireless controller joystick input forwarded to LocoClient
    - Policy does NOT output locomotion commands
    - Velocity scaling: 0.3 (same as teleop)

Usage:
    # On robot: Start image_server
    python3 image_server.py --camera-id 0

    # On GPU server: Start OpenPi policy server
    uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_g1_auto

    # On laptop: Run this client
    python g1_remote_client.py --server-host <gpu-ip> --server-port 8000

    # Or in listen mode for viz client control:
    python g1_remote_client.py --listen-mode --listen-port 5008
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import signal
import struct
import sys
import threading
import time
from enum import IntEnum
from io import BytesIO
from multiprocessing import shared_memory
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

# Import from local robot_control module (no external xr_teleoperate dependency)
from robot_control import G1_29_ArmController, G1_29_ArmIK, ImageClient
IMAGE_CLIENT_AVAILABLE = True

# Import Unitree SDK for direct Dex3 control
try:
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
    DEX3_AVAILABLE = True
except ImportError:
    logger.warning("Unitree SDK not available - hand control disabled")
    DEX3_AVAILABLE = False

# Try to import LocoClient for locomotion control
try:
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    LOCO_AVAILABLE = True
except ImportError:
    logger.warning("LocoClient not available - locomotion control disabled")
    LOCO_AVAILABLE = False

# Global for signal handler
controller = None

# Dex3 joint indices (matching robot_hand_unitree.py)
class Dex3LeftJointIndex(IntEnum):
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0 = 5
    kLeftHandIndex1 = 6

class Dex3RightJointIndex(IntEnum):
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandMiddle0 = 3
    kRightHandMiddle1 = 4
    kRightHandIndex0 = 5
    kRightHandIndex1 = 6

DEX3_NUM_MOTORS = 7

# DDS Topics
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"


def parse_wireless_remote(wireless_remote):
    """
    Parse wireless_remote bytes (40 bytes) into joystick values.
    Based on Unitree SDK wireless_controller.py
    
    Returns:
        (Lx, Ly, Rx, Ry) joystick values in range [-1, 1]
    """
    Lx = struct.unpack('<f', bytes(wireless_remote[4:8]))[0]
    Rx = struct.unpack('<f', bytes(wireless_remote[8:12]))[0]
    Ry = struct.unpack('<f', bytes(wireless_remote[12:16]))[0]
    Ly = struct.unpack('<f', bytes(wireless_remote[20:24]))[0]
    return Lx, Ly, Rx, Ry


def signal_handler(sig, frame):
    global controller
    logger.info("Caught exit signal...")
    if controller is not None:
        controller.cleanup()
    sys.exit(0)


class Dex3DirectController:
    """
    Direct joint angle controller for Dex3 hands.
    
    Unlike the VR-based Dex3_1_Controller, this class:
    - Accepts joint angles directly (no retargeting)
    - Runs in the same thread (no multiprocessing)
    - Simpler and suitable for policy inference / data replay
    """
    
    def __init__(self, dds_already_initialized: bool = False):
        """
        Initialize Dex3 direct controller.
        
        Args:
            dds_already_initialized: If True, skip DDS initialization
        """
        logger.info("Initializing Dex3DirectController...")
        
        if not DEX3_AVAILABLE:
            raise RuntimeError("Unitree SDK not available for Dex3 control")
        
        # DDS is already initialized by arm controller
        if not dds_already_initialized:
            ChannelFactoryInitialize(0)
        
        # Initialize publishers
        self.left_cmd_publisher = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
        self.left_cmd_publisher.Init()
        self.right_cmd_publisher = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self.right_cmd_publisher.Init()
        
        # Initialize subscribers
        self.left_state_subscriber = ChannelSubscriber(kTopicDex3LeftState, HandState_)
        self.left_state_subscriber.Init()
        self.right_state_subscriber = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self.right_state_subscriber.Init()
        
        # Initialize command messages
        self._init_cmd_messages()
        
        # Current state
        self.left_state = np.zeros(DEX3_NUM_MOTORS, dtype=np.float32)
        self.right_state = np.zeros(DEX3_NUM_MOTORS, dtype=np.float32)
        
        # Start state subscriber thread
        self.running = True
        self.state_thread = threading.Thread(target=self._subscribe_state, daemon=True)
        self.state_thread.start()
        
        # Wait for first state
        for _ in range(50):  # 5 seconds
            if np.any(self.left_state != 0) or np.any(self.right_state != 0):
                break
            time.sleep(0.1)
        
        logger.info("Dex3DirectController initialized")
    
    def _init_cmd_messages(self):
        """Initialize command messages with default gains."""
        q = 0.0
        dq = 0.0
        tau = 0.0
        kp = 1.5
        kd = 0.2
        
        # Left hand command
        self.left_msg = unitree_hg_msg_dds__HandCmd_()
        for joint_id in Dex3LeftJointIndex:
            motor_mode = self._make_motor_mode(joint_id, status=0x01)
            self.left_msg.motor_cmd[joint_id].mode = motor_mode
            self.left_msg.motor_cmd[joint_id].q = q
            self.left_msg.motor_cmd[joint_id].dq = dq
            self.left_msg.motor_cmd[joint_id].tau = tau
            self.left_msg.motor_cmd[joint_id].kp = kp
            self.left_msg.motor_cmd[joint_id].kd = kd
        
        # Right hand command
        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        for joint_id in Dex3RightJointIndex:
            motor_mode = self._make_motor_mode(joint_id, status=0x01)
            self.right_msg.motor_cmd[joint_id].mode = motor_mode
            self.right_msg.motor_cmd[joint_id].q = q
            self.right_msg.motor_cmd[joint_id].dq = dq
            self.right_msg.motor_cmd[joint_id].tau = tau
            self.right_msg.motor_cmd[joint_id].kp = kp
            self.right_msg.motor_cmd[joint_id].kd = kd
    
    def _make_motor_mode(self, motor_id: int, status: int = 0x01, timeout: int = 0) -> int:
        """Create motor mode byte."""
        mode = 0
        mode |= (motor_id & 0x0F)
        mode |= (status & 0x07) << 4
        mode |= (timeout & 0x01) << 7
        return mode
    
    def _subscribe_state(self):
        """Background thread to read hand states."""
        while self.running:
            try:
                # Read left hand state
                left_msg = self.left_state_subscriber.Read()
                if left_msg is not None:
                    for i, joint_id in enumerate(Dex3LeftJointIndex):
                        self.left_state[i] = left_msg.motor_state[joint_id].q
                
                # Read right hand state
                right_msg = self.right_state_subscriber.Read()
                if right_msg is not None:
                    for i, joint_id in enumerate(Dex3RightJointIndex):
                        self.right_state[i] = right_msg.motor_state[joint_id].q
            except Exception as e:
                logger.debug(f"Error reading hand state: {e}")
            
            time.sleep(0.002)  # ~500Hz
    
    def ctrl_dual_hand(self, left_q: np.ndarray, right_q: np.ndarray):
        """
        Send joint angle commands to both hands.
        
        Args:
            left_q: Left hand joint angles (7 DOF) in radians
            right_q: Right hand joint angles (7 DOF) in radians
        """
        # Update left hand command
        for i, joint_id in enumerate(Dex3LeftJointIndex):
            self.left_msg.motor_cmd[joint_id].q = float(left_q[i])
        
        # Update right hand command
        for i, joint_id in enumerate(Dex3RightJointIndex):
            self.right_msg.motor_cmd[joint_id].q = float(right_q[i])
        
        # Publish
        self.left_cmd_publisher.Write(self.left_msg)
        self.right_cmd_publisher.Write(self.right_msg)
    
    def get_hand_state(self) -> np.ndarray:
        """
        Get current hand joint positions.
        
        Returns:
            14-dim array: [left_hand(7), right_hand(7)]
        """
        return np.concatenate([self.left_state, self.right_state])
    
    def stop(self):
        """Stop the controller."""
        self.running = False
        if self.state_thread.is_alive():
            self.state_thread.join(timeout=1.0)
        logger.info("Dex3DirectController stopped")


class G1RemoteClient:
    """
    G1 Remote Policy Client with direct joint control.
    
    Supports:
    - Policy inference from OpenPi server
    - WebSocket command server for viz client
    - Head camera streaming from robot
    - Direct joint angle control for arms, Dex3 hands, and waist yaw
    - Wireless controller locomotion forwarding
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
        self.motion_mode = motion_mode
        
        # Locomotion velocity scaling (same as teleop)
        self.loco_velocity_scale = 0.3
        self.loco_client = None
        
        logger.info(f"Control frequency: {self.control_fps}Hz")
        
        # IK solver uses relative paths, so we need to change directory
        original_cwd = os.getcwd()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        robot_control_dir = os.path.join(script_dir, 'robot_control')
        urdf_check_path = os.path.join(script_dir, 'assets', 'g1', 'g1_body29_hand14.urdf')
        
        if not os.path.exists(urdf_check_path):
            raise RuntimeError(f"URDF not found at {urdf_check_path}. "
                             f"Make sure assets/g1/ folder exists in g1_control_client/")
        
        # Change to robot_control dir so ../assets/ resolves correctly
        os.chdir(robot_control_dir)
        logger.info(f"Changed to {os.getcwd()} for asset path resolution")
        
        try:
            # Initialize IK solver (uses ../assets/g1/)
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
        finally:
            # Restore original directory
            os.chdir(original_cwd)
        
        # Initialize Dex3 hand controller (direct joint control, DDS already initialized)
        logger.info("Initializing Dex3 hand controller (direct joint mode)...")
        if DEX3_AVAILABLE:
            self.hand_ctrl = Dex3DirectController(dds_already_initialized=True)
            logger.info("Hand controller ready")
        else:
            self.hand_ctrl = None
            logger.warning("Hand controller not available")
        
        # Initialize locomotion client for wireless controller forwarding
        if self.motion_mode and LOCO_AVAILABLE:
            logger.info("Initializing locomotion client (wireless controller forwarding)...")
            try:
                self.loco_client = LocoClient()
                self.loco_client.SetTimeout(0.0001)
                self.loco_client.Init()
                logger.info("Locomotion client ready")
            except Exception as e:
                logger.error(f"Failed to initialize LocoClient: {e}")
                self.loco_client = None
        else:
            if not self.motion_mode:
                logger.info("Motion mode disabled - locomotion control not available")
            elif not LOCO_AVAILABLE:
                logger.warning("LocoClient not available - locomotion control disabled")
        
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

    def _init_head_camera(self):
        """Initialize client to receive head camera from robot"""
        if not IMAGE_CLIENT_AVAILABLE:
            logger.warning("ImageClient not available - camera streaming disabled")
            self.cameras_ready = True
            return
            
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
        """
        Construct 29-dim observation for policy inference.
        
        State format (29 dims):
            [0:14]  left_arm    - left arm joint positions
            [14:28] hands       - hand joint positions (left 7 + right 7)
            [28]    waist_yaw   - waist yaw joint position
        
        Returns:
            dict with "image", "state" (29 dims), and "prompt"
        """
        # Get current arm joint positions (14 DOF)
        current_arm_q = self.robot.get_current_dual_arm_q()
        
        # Get current hand joint positions (14 DOF)
        if self.hand_ctrl is not None:
            current_hand_q = self.hand_ctrl.get_hand_state()
        else:
            current_hand_q = np.zeros(14, dtype=np.float32)
        
        # Get waist yaw (1 DOF)
        waist_yaw = self.robot.get_current_waist_yaw()
        
        # Build 29-dim state: [arm(14), hand(14), waist_yaw(1)]
        state = np.concatenate([current_arm_q, current_hand_q, [waist_yaw]])
        
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
        
        return {
            "image": {
                "cam_head": base_image,
            },
            "state": state,  # 29 dims
            "prompt": self.prompt,
        }

    def execute_action_chunk(self, policy_actions: np.ndarray, track_error: bool = False):
        """
        Execute a chunk of policy actions on the robot.
        
        Action format (29 dims):
            [0:14]  arm_joints  - arm joint targets
            [14:28] hand_joints - hand joint targets (left 7 + right 7)
            [28]    waist_yaw   - waist yaw target
        
        Args:
            policy_actions: (N, 28) or (N, 29) array of actions
            track_error: If True, log tracking error at each step
            
        Returns:
            dict with tracking statistics if track_error=True, else None
        """
        action_dim = policy_actions.shape[1]
        
        logger.info(f"Executing {len(policy_actions)} actions ({action_dim} DOF)")
        
        # Track position errors if requested
        if track_error:
            arm_errors = []
            waist_errors = []
        
        # Execute at control_fps
        for i, action in enumerate(policy_actions):
            # Extract arm joints (14 DOF)
            arm_joints = action[:14]
            
            # Extract hand joints (14 DOF): left (7) + right (7)
            if action_dim >= 28:
                hand_joints = action[14:28]
            else:
                hand_joints = np.zeros(14, dtype=np.float32)
            
            # Extract waist yaw (1 DOF)
            waist_yaw = action[28] if action_dim >= 29 else 0.0
            
            # Track position error before sending command
            if track_error:
                current_arm = self.robot.get_current_dual_arm_q()
                current_waist = self.robot.get_current_waist_yaw()
                
                # Compare current position to what we're about to command
                arm_error = np.abs(current_arm - arm_joints)
                waist_error = abs(current_waist - waist_yaw)
                
                arm_errors.append(arm_error)
                waist_errors.append(waist_error)
                
                # Log significant errors
                max_arm_error = np.max(arm_error)
                if max_arm_error > 0.03 or waist_error > 0.05:  # ~1.7 deg or ~2.8 deg
                    logger.warning(f"  Step {i}: Large tracking error - arm_max={np.degrees(max_arm_error):.2f}°, "
                                   f"waist={np.degrees(waist_error):.2f}°")
            
            # Send arm command
            self.robot.ctrl_dual_arm(
                q_target=arm_joints,
                tauff_target=np.zeros(14, dtype=np.float32)
            )
            
            # Send hand command (direct joint angles)
            if self.hand_ctrl is not None:
                left_hand = hand_joints[:7]
                right_hand = hand_joints[7:14]
                self.hand_ctrl.ctrl_dual_hand(left_hand, right_hand)
            
            # Send waist yaw command
            self.robot.ctrl_waist_yaw(waist_yaw)
            
            # Forward wireless controller locomotion (human control only)
            self._forward_controller_locomotion()
            
            # Log every 10th action (without error tracking spam)
            if i % 10 == 0 and not track_error:
                logger.info(f"  Step {i}/{len(policy_actions)}")
            
            time.sleep(1.0 / self.control_fps)
        
        # Report tracking statistics
        if track_error:
            arm_errors = np.array(arm_errors)
            waist_errors = np.array(waist_errors)
            
            stats = {
                "arm_mean_error_deg": float(np.degrees(np.mean(arm_errors))),
                "arm_max_error_deg": float(np.degrees(np.max(arm_errors))),
                "waist_mean_error_deg": float(np.degrees(np.mean(waist_errors))),
                "waist_max_error_deg": float(np.degrees(np.max(waist_errors))),
                "arm_per_joint_mean_deg": [float(np.degrees(np.mean(arm_errors[:, j]))) for j in range(14)],
            }
            
            logger.info(f"Tracking stats: arm_mean={stats['arm_mean_error_deg']:.2f}°, "
                        f"arm_max={stats['arm_max_error_deg']:.2f}°, "
                        f"waist_mean={stats['waist_mean_error_deg']:.2f}°")
            
            # Identify joints with worst tracking
            worst_joints = np.argsort(stats['arm_per_joint_mean_deg'])[-3:][::-1]
            joint_names = ['L_sh_pitch', 'L_sh_roll', 'L_sh_yaw', 'L_elbow', 'L_wr_roll', 'L_wr_pitch', 'L_wr_yaw',
                           'R_sh_pitch', 'R_sh_roll', 'R_sh_yaw', 'R_elbow', 'R_wr_roll', 'R_wr_pitch', 'R_wr_yaw']
            logger.info(f"Worst tracking joints: {[(joint_names[j], f'{stats[\"arm_per_joint_mean_deg\"][j]:.2f}°') for j in worst_joints]}")
            
            return stats
        
        logger.info("Action chunk execution complete")
        return None
    
    def wait_for_convergence(self, target_arm: np.ndarray, target_waist: float = None,
                             threshold_rad: float = 0.02, timeout_s: float = 1.0) -> bool:
        """
        Wait for robot to converge to target position.
        
        Args:
            target_arm: Target arm positions (14 DOF)
            target_waist: Target waist yaw (optional)
            threshold_rad: Convergence threshold in radians (~1.1 degrees)
            timeout_s: Maximum time to wait
            
        Returns:
            True if converged, False if timed out
        """
        start_time = time.time()
        
        while (time.time() - start_time) < timeout_s:
            current_arm = self.robot.get_current_dual_arm_q()
            arm_error = np.max(np.abs(current_arm - target_arm))
            
            if target_waist is not None:
                current_waist = self.robot.get_current_waist_yaw()
                waist_error = abs(current_waist - target_waist)
            else:
                waist_error = 0.0
            
            if arm_error < threshold_rad and waist_error < threshold_rad:
                return True
            
            time.sleep(0.01)  # Check at 100Hz
        
        logger.warning(f"Convergence timeout: arm_error={np.degrees(arm_error):.2f}°, "
                       f"waist_error={np.degrees(waist_error):.2f}°")
        return False

    def _forward_controller_locomotion(self):
        """Forward wireless controller joystick input to locomotion."""
        if self.loco_client is None:
            return
        
        try:
            lowstate = self.robot.get_lowstate_raw()
            if lowstate is not None:
                Lx, Ly, Rx, Ry = parse_wireless_remote(lowstate.wireless_remote)
                # Match teleop convention: negate values
                # Ly controls forward/backward, Lx controls strafe, Rx controls turn
                self.loco_client.Move(
                    -Ly * self.loco_velocity_scale,  # Forward/back
                    -Lx * self.loco_velocity_scale,  # Strafe
                    -Rx * self.loco_velocity_scale   # Turn
                )
        except Exception as e:
            logger.debug(f"Failed to forward controller locomotion: {e}")

    async def command_server(self, port: int = 5008):
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
                            track_error = data.get("track_error", False)
                            wait_converge = data.get("wait_converge", False)
                            
                            stats = self.execute_action_chunk(actions, track_error=track_error)
                            
                            # Optionally wait for convergence after chunk
                            if wait_converge and len(actions) > 0:
                                last_action = actions[-1]
                                waist_target = last_action[28] if len(last_action) >= 29 else None
                                converged = self.wait_for_convergence(
                                    last_action[:14], waist_target,
                                    threshold_rad=0.02, timeout_s=0.5
                                )
                                if not converged:
                                    logger.warning("Did not converge after action chunk")
                            
                            response = {
                                "status": "success",
                                "message": f"Executed {len(actions)} actions",
                                "tracking_stats": stats
                            }
                        
                        elif cmd == "reset":
                            target = np.array(data["joints"], dtype=np.float32)
                            duration = data.get("duration", 2.0)
                            
                            logger.info(f"Resetting to joints over {duration}s...")
                            
                            # Extract targets
                            arm_target = target[:14]
                            waist_target = target[28] if len(target) >= 29 else None
                            
                            # Get current positions
                            current_arm = self.robot.get_current_dual_arm_q()
                            current_waist = self.robot.get_current_waist_yaw() if waist_target is not None else None
                            
                            # Smooth interpolation for arms AND waist together
                            steps = int(duration * 250)
                            
                            for i in range(steps):
                                alpha = (i + 1) / steps
                                
                                # Interpolate arm
                                interp_arm = current_arm * (1 - alpha) + arm_target * alpha
                                self.robot.ctrl_dual_arm(
                                    q_target=interp_arm,
                                    tauff_target=np.zeros(14, dtype=np.float32)
                                )
                                
                                # Interpolate waist (slower, smoother motion)
                                if waist_target is not None:
                                    interp_waist = current_waist * (1 - alpha) + waist_target * alpha
                                    self.robot.ctrl_waist_yaw(interp_waist)
                                
                                await asyncio.sleep(1.0 / 250)
                            
                            # Set hand target if provided (indices 14-28)
                            if len(target) >= 28 and self.hand_ctrl is not None:
                                hand_target = target[14:28]
                                self.hand_ctrl.ctrl_dual_hand(hand_target[:7], hand_target[7:])
                            
                            response = {"status": "success", "message": "Reset complete"}
                        
                        elif cmd == "get_state":
                            arm_state = self.robot.get_current_dual_arm_q()
                            if self.hand_ctrl is not None:
                                hand_state = self.hand_ctrl.get_hand_state()
                            else:
                                hand_state = np.zeros(14, dtype=np.float32)
                            waist_yaw = self.robot.get_current_waist_yaw()
                            full_state = np.concatenate([arm_state, hand_state, [waist_yaw]])
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
                            # Stop arm movement - hold current position
                            current = self.robot.get_current_dual_arm_q()
                            self.robot.ctrl_dual_arm(
                                q_target=current,
                                tauff_target=np.zeros(14, dtype=np.float32)
                            )
                            # Stop locomotion
                            if self.loco_client is not None:
                                try:
                                    self.loco_client.Damp()
                                except Exception as e:
                                    logger.error(f"Failed to damp locomotion: {e}")
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
        if self.hand_ctrl is not None:
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
    parser.add_argument("--listen-port", type=int, default=5008,
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
