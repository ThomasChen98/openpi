#!/usr/bin/env python3
"""
G1 Training Pipeline Client

A systematic training pipeline that alternates between:
1. Policy execution (robot runs inference)
2. Human correction (operator adjusts robot in damping mode)

All data is continuously recorded at 30Hz with phase labels.

State Machine:
    WAITING -> READY -> EXECUTING -> LABELING -> DAMPING -> SAVING -> DECIDING -> (loop or SYNCING)

Key concepts:
    - EPOCH: A training cycle with a specific policy checkpoint. Multiple episodes per epoch.
    - EPISODE: A single rollout/trajectory recorded during EXECUTING state.

State Space (29 dims):
    [0:28]  qpos        - arm (14) + hand (14) joint positions
    [28]    waist_yaw   - waist yaw joint position

Action Space (29 dims):
    [0:28]  upper_body  - arm (14) + hand (14) joint targets
    [28]    waist_yaw   - waist yaw joint target

Usage:
    python g1_execution_client.py --config training_config_g1.yaml
"""

import argparse
import logging
import os
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
import tty
from enum import Enum, auto
from multiprocessing import shared_memory
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# OpenPi client import for image_tools
try:
    from openpi_client import image_tools
    IMAGE_TOOLS_AVAILABLE = True
except ImportError:
    logger.warning("openpi_client.image_tools not available, using basic image conversion")
    IMAGE_TOOLS_AVAILABLE = False

# Import G1 robot control from local module
try:
    from robot_control import G1_29_ArmController, G1_29_ArmIK, ImageClient
    ROBOT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Robot control not available: {e}")
    ROBOT_AVAILABLE = False

# Import Unitree SDK for Dex3 control
try:
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
    DEX3_AVAILABLE = True
except ImportError:
    logger.warning("Unitree SDK not available - hand control disabled")
    DEX3_AVAILABLE = False

# Try to import LocoClient for locomotion control (wireless controller forwarding)
try:
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    LOCO_AVAILABLE = True
except ImportError:
    logger.warning("LocoClient not available - locomotion control disabled")
    LOCO_AVAILABLE = False


class TrainingState(Enum):
    """Training pipeline states"""
    WAITING = auto()      # Waiting for policy server to have new weights
    READY = auto()        # Policy ready, waiting for user to confirm start
    EXECUTING = auto()    # Running policy inference
    LABELING = auto()     # User labels the execution as good/bad
    DAMPING = auto()      # Damping mode - operator adjusts robot
    SAVING = auto()       # Saving episode data
    DECIDING = auto()     # User decides to continue or end
    SYNCING = auto()      # Syncing data to remote server
    FINISHED = auto()     # Training session complete


class KeyboardHandler:
    """Non-blocking keyboard input handler"""
    
    def __init__(self):
        self.old_settings = None
        
    def __enter__(self):
        """Set terminal to raw mode for single character input"""
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except termios.error:
            # Not a terminal (e.g., running in background)
            self.old_settings = None
        return self
        
    def __exit__(self, *args):
        """Restore terminal settings"""
        if self.old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def get_key(self, timeout: float = 0.1) -> Optional[str]:
        """Get a single keypress with timeout."""
        import select
        
        if self.old_settings is None:
            return None
            
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.read(1)
        return None
    
    def wait_for_key(self, valid_keys: set, prompt: str = "") -> str:
        """Block until one of the valid keys is pressed."""
        if prompt:
            print(prompt, end='', flush=True)
            
        while True:
            key = self.get_key(timeout=0.1)
            if key and key.lower() in valid_keys:
                print(key)  # Echo the key
                return key.lower()


# Dex3 joint indices
class Dex3LeftJointIndex:
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0 = 5
    kLeftHandIndex1 = 6

class Dex3RightJointIndex:
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
    """Parse wireless_remote bytes (40 bytes) into joystick values."""
    Lx = struct.unpack('<f', bytes(wireless_remote[4:8]))[0]
    Rx = struct.unpack('<f', bytes(wireless_remote[8:12]))[0]
    Ry = struct.unpack('<f', bytes(wireless_remote[12:16]))[0]
    Ly = struct.unpack('<f', bytes(wireless_remote[20:24]))[0]
    return Lx, Ly, Rx, Ry


class Dex3DirectController:
    """Direct joint angle controller for Dex3 hands."""
    
    def __init__(self, dds_already_initialized: bool = False):
        logger.info("Initializing Dex3DirectController...")
        
        if not DEX3_AVAILABLE:
            raise RuntimeError("Unitree SDK not available for Dex3 control")
        
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
        
        self._init_cmd_messages()
        
        self.left_state = np.zeros(DEX3_NUM_MOTORS, dtype=np.float32)
        self.right_state = np.zeros(DEX3_NUM_MOTORS, dtype=np.float32)
        
        self.running = True
        self.state_thread = threading.Thread(target=self._subscribe_state, daemon=True)
        self.state_thread.start()
        
        for _ in range(50):
            if np.any(self.left_state != 0) or np.any(self.right_state != 0):
                break
            time.sleep(0.1)
        
        logger.info("Dex3DirectController initialized")
    
    def _init_cmd_messages(self):
        q, dq, tau, kp, kd = 0.0, 0.0, 0.0, 1.5, 0.2
        
        self.left_msg = unitree_hg_msg_dds__HandCmd_()
        for joint_id in range(DEX3_NUM_MOTORS):
            self.left_msg.motor_cmd[joint_id].mode = self._make_motor_mode(joint_id, status=0x01)
            self.left_msg.motor_cmd[joint_id].q = q
            self.left_msg.motor_cmd[joint_id].dq = dq
            self.left_msg.motor_cmd[joint_id].tau = tau
            self.left_msg.motor_cmd[joint_id].kp = kp
            self.left_msg.motor_cmd[joint_id].kd = kd
        
        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        for joint_id in range(DEX3_NUM_MOTORS):
            self.right_msg.motor_cmd[joint_id].mode = self._make_motor_mode(joint_id, status=0x01)
            self.right_msg.motor_cmd[joint_id].q = q
            self.right_msg.motor_cmd[joint_id].dq = dq
            self.right_msg.motor_cmd[joint_id].tau = tau
            self.right_msg.motor_cmd[joint_id].kp = kp
            self.right_msg.motor_cmd[joint_id].kd = kd
    
    def _make_motor_mode(self, motor_id: int, status: int = 0x01, timeout: int = 0) -> int:
        mode = 0
        mode |= (motor_id & 0x0F)
        mode |= (status & 0x07) << 4
        mode |= (timeout & 0x01) << 7
        return mode
    
    def _subscribe_state(self):
        while self.running:
            try:
                left_msg = self.left_state_subscriber.Read()
                if left_msg is not None:
                    for i in range(DEX3_NUM_MOTORS):
                        self.left_state[i] = left_msg.motor_state[i].q
                
                right_msg = self.right_state_subscriber.Read()
                if right_msg is not None:
                    for i in range(DEX3_NUM_MOTORS):
                        self.right_state[i] = right_msg.motor_state[i].q
            except Exception as e:
                logger.debug(f"Error reading hand state: {e}")
            time.sleep(0.002)
    
    def ctrl_dual_hand(self, left_q: np.ndarray, right_q: np.ndarray):
        for i in range(DEX3_NUM_MOTORS):
            self.left_msg.motor_cmd[i].q = float(left_q[i])
        for i in range(DEX3_NUM_MOTORS):
            self.right_msg.motor_cmd[i].q = float(right_q[i])
        self.left_cmd_publisher.Write(self.left_msg)
        self.right_cmd_publisher.Write(self.right_msg)
    
    def get_hand_state(self) -> np.ndarray:
        return np.concatenate([self.left_state, self.right_state])
    
    def stop(self):
        self.running = False
        if self.state_thread.is_alive():
            self.state_thread.join(timeout=1.0)
        logger.info("Dex3DirectController stopped")


class G1TrainingClient:
    """
    Training pipeline client for G1 robot.
    
    Implements a state machine that alternates between policy execution
    and human correction phases, with continuous data recording.
    """
    
    def __init__(self, config_path: str):
        """Initialize training client."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info(f"Loaded config from {config_path}")
        
        # State tracking
        self.state = TrainingState.WAITING
        self.epoch_num = 0
        self.episode_num = 0
        self.total_episodes = 0
        self.running = True
        self.last_policy_epoch = -1
        self.episode_rejected = False
        
        # G1 29-dim action space: 28 upper body + 1 waist_yaw
        self.action_dim = 29
        self.upper_body_dim = 28  # 14 arm + 14 hand
        logger.info(f"Action dim: {self.action_dim} (28 upper body + 1 waist_yaw)")
        
        # Control frequency
        self.control_freq = self.config.get('robot', {}).get('control_freq', 30)
        logger.info(f"Control frequency: {self.control_freq}Hz")
        
        # Locomotion velocity scaling (for wireless controller forwarding)
        self.loco_velocity_scale = 0.3
        
        # Components (initialized lazily)
        self.robot = None
        self.ik_solver = None
        self.hand_ctrl = None
        self.loco_client = None
        self.policy_client = None
        self.episode_writer = None
        self.keyboard = KeyboardHandler()
        
        # Camera components
        self.head_img_array = None
        self.head_img_shm = None
        self.head_camera_client = None
        self.head_camera_thread = None
        self.cameras_ready = False
        
        # Recording state
        self.recording_active = False
        self.current_phase = "policy"
        self.current_advantage_label = None
        
        # Position hold state
        self._hold_position_background = False
        
        # Gravity compensation (reduces arm dropping)
        self.use_gravity_compensation = self.config.get('execution', {}).get('gravity_compensation', True)
        if self.use_gravity_compensation:
            logger.info("Gravity compensation ENABLED - arms should track better")
        
        # Reset pose for robot (29 DOF: 14 arm + 14 hand + 1 waist_yaw)
        # Zeros for home position
        self.reset_pose = np.array([
            -0.8327958,   0.68337566, -0.15583088,  0.65219355,  0.6835083,  -0.90980643
            0.40519863, -0.63599086, -0.6096487,   0.1507975,   0.20780647, -0.9646822
            -0.12156798, -0.7018801,  -0.8908997,   0.884532,    0.56234354, -0.30448544
            -0.1900528,  -0.07469787, -0.45043802, -0.99141276, -0.8881592,  -1.1634055
            0.13551766,  0.01854079,  0.03298719,  0.20263577, -0.18577237
        ])
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, sig, frame):
        print("\n\nReceived interrupt signal...")
        self.running = False
        if self.recording_active and self.episode_writer:
            logger.info("Saving current recording before exit...")
            self.episode_writer.stop_recording()
        self.cleanup()
        sys.exit(0)
    
    def initialize_robot(self):
        """Initialize robot controller, IK solver, Dex3 hands, and cameras"""
        logger.info("Initializing G1 robot components...")
        
        if not ROBOT_AVAILABLE:
            logger.error("Robot control module not available")
            return False
        
        try:
            robot_config = self.config.get('robot', {})
            
            # IK solver uses relative paths, so we need to change directory
            original_cwd = os.getcwd()
            script_dir = os.path.dirname(os.path.abspath(__file__))
            robot_control_dir = os.path.join(script_dir, 'robot_control')
            urdf_check_path = os.path.join(script_dir, 'assets', 'g1', 'g1_body29_hand14.urdf')
            
            if not os.path.exists(urdf_check_path):
                logger.error(f"URDF not found at {urdf_check_path}")
                return False
            
            # Change to robot_control dir so ../assets/ resolves correctly
            os.chdir(robot_control_dir)
            logger.info(f"  Changed to {os.getcwd()} for asset path resolution")
            
            try:
                # Initialize IK solver
                logger.info("  Initializing IK solver...")
                self.ik_solver = G1_29_ArmIK(Unit_Test=False, Visualization=False)
                logger.info("  IK solver ready")
                
                # Initialize robot arm controller (this initializes DDS)
                logger.info("  Initializing arm controller...")
                self.robot = G1_29_ArmController(
                    motion_mode=True,
                    simulation_mode=False,
                    dds_already_initialized=False
                )
                logger.info("  Arm controller ready")
            finally:
                os.chdir(original_cwd)
            
            # Initialize Dex3 hand controller
            if DEX3_AVAILABLE:
                logger.info("  Initializing Dex3 hand controller...")
                self.hand_ctrl = Dex3DirectController(dds_already_initialized=True)
                logger.info("  Hand controller ready")
            else:
                self.hand_ctrl = None
                logger.warning("  Hand controller not available")
            
            # Initialize locomotion client (for wireless controller forwarding only)
            if LOCO_AVAILABLE:
                logger.info("  Initializing locomotion client (wireless controller forwarding)...")
                try:
                    self.loco_client = LocoClient()
                    self.loco_client.SetTimeout(0.0001)
                    self.loco_client.Init()
                    logger.info("  Locomotion client ready")
                except Exception as e:
                    logger.error(f"  Failed to initialize LocoClient: {e}")
                    self.loco_client = None
            
            # Initialize cameras in background
            logger.info("  Initializing cameras (in background)...")
            self._init_cameras_background()
            
            logger.info("G1 robot initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize robot: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _init_cameras_background(self):
        """Initialize cameras in a background thread"""
        robot_config = self.config.get('robot', {})
        
        def init_cameras_thread():
            self._init_head_camera(
                robot_config.get('head_camera_server_ip', '192.168.123.164'),
                robot_config.get('head_camera_server_port', 5555)
            )
            
            if self.head_img_array is not None:
                logger.info("  Camera ready: head camera working")
            else:
                logger.info("  Camera ready: using dummy images")
            
            self.cameras_ready = True
        
        camera_thread = threading.Thread(target=init_cameras_thread, daemon=True)
        camera_thread.start()
        logger.info("  Camera initializing in background...")
    
    def _init_head_camera(self, server_ip: str, server_port: int):
        """Initialize head camera client (receives from robot via ZMQ)"""
        try:
            logger.info(f"    Connecting to head camera at {server_ip}:{server_port}...")
            
            self.head_img_shape = (480, 640, 3)
            
            self.head_img_shm = shared_memory.SharedMemory(
                create=True,
                size=np.prod(self.head_img_shape) * np.uint8().itemsize
            )
            self.head_img_array = np.ndarray(
                self.head_img_shape, dtype=np.uint8, buffer=self.head_img_shm.buf
            )
            
            self.head_camera_client = ImageClient(
                tv_img_shape=self.head_img_shape,
                tv_img_shm_name=self.head_img_shm.name,
                wrist_img_shape=None,
                wrist_img_shm_name=None,
                server_address=server_ip,
                port=server_port,
                image_show=False
            )
            
            self.head_camera_thread = threading.Thread(
                target=self.head_camera_client.receive_process,
                daemon=True
            )
            self.head_camera_thread.start()
            
            for _ in range(30):
                time.sleep(0.1)
                if np.any(self.head_img_array != 0):
                    logger.info("    Head camera connected!")
                    return
            
            logger.warning("    Head camera connected but no frames yet")
            
        except Exception as e:
            logger.error(f"    Failed to init head camera: {e}")
            self.head_img_array = None
    
    def initialize_policy_client(self):
        """Initialize connection to policy server"""
        try:
            from openpi_client import websocket_client_policy
            
            server_config = self.config.get('policy_server', {})
            host = server_config.get('host', 'localhost')
            port = server_config.get('port', 8001)  # G1 uses 8001
            
            logger.info(f"Connecting to policy server at {host}:{port}...")
            self.policy_client = websocket_client_policy.WebsocketClientPolicy(
                host=host,
                port=port
            )
            
            metadata = self.policy_client.get_server_metadata()
            logger.info(f"  Connected! Action dim: {metadata.get('action_dim', 'N/A')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to policy server: {e}")
            return False
    
    def poll_training_status(self) -> dict:
        """Poll the policy server for training status."""
        server_config = self.config.get('policy_server', {})
        host = server_config.get('host', 'localhost')
        port = server_config.get('port', 8001)
        
        try:
            response = requests.get(
                f"http://{host}:{port}/training_status",
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException as e:
            logger.debug(f"Failed to poll training status: {e}")
        
        return {"ready": False, "epoch": -1}
    
    def start_recording(self, label_name: str = None):
        """Start recording a new episode within the current epoch."""
        from utils.episode_writer_hdf5 import EpisodeWriterHDF5
        
        fps = self.control_freq
        
        data_config = self.config.get('data', {})
        base_save_dir = data_config.get('save_dir', 'g1_data_auto')
        
        # Convert to absolute path if relative
        if not os.path.isabs(base_save_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_save_dir = os.path.join(script_dir, base_save_dir)
        
        # Normalize the path to remove any ./ or extra slashes
        base_save_dir = os.path.normpath(base_save_dir)
        
        task_config = self.config.get('task', {})
        task_name = task_config.get('name', 'training_session')
        
        # Epoch-based directory: {base_dir}/{task_name}/epoch_{N}/raw/
        epoch_dir = os.path.join(base_save_dir, task_name, f"epoch_{self.epoch_num}", "raw")
        
        # Create the directory explicitly before initializing writer
        os.makedirs(epoch_dir, exist_ok=True)
        logger.info(f"Created save directory: {epoch_dir}")
        
        self.episode_writer = EpisodeWriterHDF5(
            save_dir=epoch_dir,
            label_name="",
            fps=fps
        )
        self.episode_writer.start_recording()
        self.recording_active = True
        logger.info(f"Started recording: {self.episode_writer.filepath}")
        logger.info(f"   Task: {task_name}")
        logger.info(f"   Epoch: {self.epoch_num}, Episode: {self.episode_num}")
    
    def stop_recording(self):
        """Stop and save the current recording"""
        if self.episode_writer:
            if self.episode_writer.get_current_length() > 0:
                filepath = self.episode_writer.filepath
                length = self.episode_writer.get_current_length()
                
                if self.episode_writer.get_advantage_label() is None:
                    self.episode_writer.set_advantage_label(False)
                    logger.warning("No advantage label set, defaulting to False")
                
                self.episode_writer.stop_recording()
                
                logger.info(f"Saved episode: {filepath}")
                logger.info(f"  Total timesteps: {length}")
            
            self.recording_active = False
    
    def get_current_state(self) -> np.ndarray:
        """Get current robot state (29 DOF: arm + hand + waist_yaw)."""
        arm_q = self.robot.get_current_dual_arm_q()  # 14 DOF
        
        if self.hand_ctrl is not None:
            hand_q = self.hand_ctrl.get_hand_state()  # 14 DOF
        else:
            hand_q = np.zeros(14, dtype=np.float32)
        
        waist_yaw = self.robot.get_current_waist_yaw()  # 1 DOF
        
        return np.concatenate([arm_q, hand_q, [waist_yaw]])  # 29 DOF
    
    def get_current_qvel(self) -> np.ndarray:
        """Get current robot joint velocities (29 DOF)."""
        arm_dq = self.robot.get_current_dual_arm_dq()  # 14 DOF
        
        # Hands don't report velocity, use zeros
        hand_dq = np.zeros(14, dtype=np.float32)
        
        # Waist yaw velocity (simplified - use 0)
        waist_yaw_dq = 0.0
        
        return np.concatenate([arm_dq, hand_dq, [waist_yaw_dq]])  # 29 DOF
    
    def get_observation(self, for_policy: bool = False) -> dict:
        """
        Get current observation from robot.
        
        Args:
            for_policy: If True, format for policy inference (29-dim state)
                       If False, format for recording
        """
        # Get current state (29 DOF: arm + hand + waist_yaw)
        current_q = self.get_current_state()
        
        # Dummy image for fallback
        dummy_image = np.full((224, 224, 3), 128, dtype=np.uint8)
        
        # Get head camera
        if self.head_img_array is not None:
            try:
                head_frame = self.head_img_array.copy()
                if np.any(head_frame != 0):
                    head_image = cv2.resize(head_frame, (224, 224))
                    head_image = cv2.cvtColor(head_image, cv2.COLOR_BGR2RGB)
                    if IMAGE_TOOLS_AVAILABLE:
                        head_image = image_tools.convert_to_uint8(head_image)
                else:
                    head_image = dummy_image
            except Exception:
                head_image = dummy_image
        else:
            head_image = dummy_image
        
        if for_policy:
            # Format for policy inference: 29-dim state
            task_config = self.config.get('task', {})
            task_description = task_config.get('description', 'manipulation task')
            
            return {
                "images": {"cam_head": head_image},
                "state": current_q,  # 29 dims
                "prompt": f"{task_description}, Advantage=True",
            }
        else:
            # Format for recording
            return {
                "state": current_q,  # 29 DOF
                "images": {"cam_head": head_image},
            }
    
    def reset_to_pose(self, duration: float = 2.0):
        """Smoothly reset the robot to the configured reset pose."""
        logger.info(f"Resetting robot to configured pose (duration: {duration}s)...")
        
        current_q = self.robot.get_current_dual_arm_q()
        num_steps = int(duration * self.control_freq)
        
        for i in range(num_steps):
            t = (i + 1) / num_steps
            t_smooth = t * t * (3 - 2 * t)
            
            target_q = current_q + t_smooth * (self.reset_pose[:14] - current_q)
            
            self.robot.ctrl_dual_arm(
                q_target=target_q,
                tauff_target=np.zeros(14, dtype=np.float32),
                use_gravity_compensation=self.use_gravity_compensation
            )
            
            # Move hands to reset pose
            if self.hand_ctrl is not None and len(self.reset_pose) >= 28:
                hand_target = self.reset_pose[14:28]
                self.hand_ctrl.ctrl_dual_hand(hand_target[:7], hand_target[7:])
            
            # Move waist yaw to reset pose
            if len(self.reset_pose) >= 29:
                waist_yaw_target = self.reset_pose[28]
                self.robot.ctrl_waist_yaw(waist_yaw_target)
            
            time.sleep(1.0 / self.control_freq)
        
        logger.info("Reset complete")
    
    def query_policy(self) -> np.ndarray:
        """Query the policy server for an action chunk."""
        obs = self.get_observation(for_policy=True)
        response = self.policy_client.infer(obs)
        action_chunk = response["actions"]
        
        logger.info(f"Received action chunk: shape={action_chunk.shape}")
        logger.info(f"   Range: [{action_chunk.min():.3f}, {action_chunk.max():.3f}]")
        
        return action_chunk
    
    def execute_action_chunk(self, action_chunk: np.ndarray, track_error: bool = True) -> int:
        """
        Execute a full action chunk on the robot.
        
        Action format (29 dims):
            [0:14]  arm_joints
            [14:28] hand_joints
            [28]    waist_yaw
            
        Args:
            action_chunk: Array of actions to execute
            track_error: If True, log position tracking errors
        """
        control_period = 1.0 / self.control_freq
        actions_executed = 0
        
        action_dim = action_chunk.shape[1] if len(action_chunk.shape) > 1 else self.action_dim
        
        logger.info(f"   Executing {len(action_chunk)} actions at {self.control_freq}Hz ({action_dim} DOF)...")
        
        # Tracking error statistics
        arm_errors = []
        waist_errors = []
        per_joint_errors = [[] for _ in range(14)]  # Track each arm joint
        
        for i, action in enumerate(action_chunk):
            loop_start = time.time()
            
            # Check for stop key
            key = self.keyboard.get_key(timeout=0.001)
            if key and key.lower() == 's':
                logger.info(f"Stop signal received at action {i}/{len(action_chunk)}")
                break
            
            # Extract arm joints (14 DOF)
            arm_joints = action[:14]
            
            # Extract hand joints (14 DOF)
            hand_joints = action[14:28] if action_dim >= 28 else np.zeros(14, dtype=np.float32)
            
            # Extract waist yaw (1 DOF)
            waist_yaw = action[28] if action_dim >= 29 else 0.0
            
            # Track position error before sending new command
            if track_error:
                current_arm = self.robot.get_current_dual_arm_q()
                current_waist = self.robot.get_current_waist_yaw()
                
                arm_error = np.abs(current_arm - arm_joints)
                arm_max_error = np.max(arm_error)
                waist_error = abs(current_waist - waist_yaw)
                
                arm_errors.append(arm_max_error)
                waist_errors.append(waist_error)
                
                for j in range(14):
                    per_joint_errors[j].append(arm_error[j])
                
                # Log warnings for large errors (threshold: 2 degrees)
                if arm_max_error > 0.035 or waist_error > 0.035:  # ~2 degrees
                    logger.warning(f"  Step {i}: Tracking error - arm_max={np.degrees(arm_max_error):.2f}°, waist={np.degrees(waist_error):.2f}°")
            
            # Send arm command with optional gravity compensation
            self.robot.ctrl_dual_arm(
                q_target=arm_joints,
                tauff_target=np.zeros(14, dtype=np.float32),
                use_gravity_compensation=self.use_gravity_compensation
            )
            
            # Send hand command
            if self.hand_ctrl is not None:
                self.hand_ctrl.ctrl_dual_hand(hand_joints[:7], hand_joints[7:14])
            
            # Send waist yaw command
            self.robot.ctrl_waist_yaw(waist_yaw)
            
            # Forward wireless controller locomotion (human control only)
            self._forward_controller_locomotion()
            
            # Record timestep
            if self.recording_active and self.episode_writer:
                current_q = self.get_current_state()  # 29 DOF
                current_qvel = self.get_current_qvel()  # 29 DOF
                obs = self.get_observation(for_policy=False)
                
                # Action is 29-dim: arm + hand + waist_yaw
                recorded_action = action[:29] if action_dim >= 29 else np.concatenate([action[:28], [waist_yaw]])
                
                self.episode_writer.add_timestep(
                    qpos=current_q,
                    qvel=current_qvel,
                    action=recorded_action,
                    images=obs.get('images'),
                    phase="policy"
                )
            
            actions_executed += 1
            
            if i % 10 == 0:
                logger.info(f"   Step {i}/{len(action_chunk)}")
            
            elapsed = time.time() - loop_start
            sleep_time = max(0, control_period - elapsed)
            time.sleep(sleep_time)
        
        # Log tracking statistics
        if track_error and arm_errors:
            arm_mean = np.degrees(np.mean(arm_errors))
            arm_max = np.degrees(np.max(arm_errors))
            waist_mean = np.degrees(np.mean(waist_errors))
            
            # Find worst joints
            joint_names = ['L_sh_pitch', 'L_sh_roll', 'L_sh_yaw', 'L_elbow', 'L_wr_roll', 'L_wr_pitch', 'L_wr_yaw',
                          'R_sh_pitch', 'R_sh_roll', 'R_sh_yaw', 'R_elbow', 'R_wr_roll', 'R_wr_pitch', 'R_wr_yaw']
            joint_mean_errors = [np.degrees(np.mean(errs)) for errs in per_joint_errors]
            worst_joints = np.argsort(joint_mean_errors)[-3:][::-1]
            
            logger.info(f"Tracking stats: arm_mean={arm_mean:.2f}°, arm_max={arm_max:.2f}°, waist_mean={waist_mean:.2f}°")
            worst_info = [(joint_names[j], f'{joint_mean_errors[j]:.2f}°') for j in worst_joints]
            logger.info(f"Worst tracking joints: {worst_info}")
        
        logger.info(f"Executed {actions_executed} actions")
        return actions_executed
    
    def _forward_controller_locomotion(self):
        """Forward wireless controller joystick input to locomotion."""
        if self.loco_client is None:
            return
        
        try:
            lowstate = self.robot.get_lowstate_raw()
            if lowstate is not None:
                Lx, Ly, Rx, Ry = parse_wireless_remote(lowstate.wireless_remote)
                self.loco_client.Move(
                    -Ly * self.loco_velocity_scale,
                    -Lx * self.loco_velocity_scale,
                    -Rx * self.loco_velocity_scale
                )
        except Exception as e:
            logger.debug(f"Failed to forward controller locomotion: {e}")
    
    def wait_for_convergence(self, target_arm: np.ndarray, target_waist: float, 
                             threshold_deg: float = 1.5, timeout: float = 0.5) -> bool:
        """
        Wait for robot to converge to target position.
        
        Args:
            target_arm: Target arm joint positions (14 DOF)
            target_waist: Target waist yaw position
            threshold_deg: Convergence threshold in degrees
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if converged, False if timed out
        """
        threshold_rad = np.radians(threshold_deg)
        start_time = time.time()
        control_period = 1.0 / self.control_freq
        
        while time.time() - start_time < timeout:
            current_arm = self.robot.get_current_dual_arm_q()
            current_waist = self.robot.get_current_waist_yaw()
            
            arm_error = np.max(np.abs(current_arm - target_arm))
            waist_error = abs(current_waist - target_waist)
            
            if arm_error < threshold_rad and waist_error < threshold_rad:
                return True
            
            # Keep commanding target while waiting
            self.robot.ctrl_dual_arm(
                q_target=target_arm,
                tauff_target=np.zeros(14, dtype=np.float32),
                use_gravity_compensation=self.use_gravity_compensation
            )
            self.robot.ctrl_waist_yaw(target_waist)
            
            time.sleep(control_period)
        
        # Log final error on timeout
        current_arm = self.robot.get_current_dual_arm_q()
        current_waist = self.robot.get_current_waist_yaw()
        arm_error = np.degrees(np.max(np.abs(current_arm - target_arm)))
        waist_error = np.degrees(abs(current_waist - target_waist))
        logger.warning(f"Convergence timeout: arm_error={arm_error:.2f}°, waist_error={waist_error:.2f}°")
        
        return False
    
    def rsync_to_remote(self) -> bool:
        """Sync recorded data to remote server using rsync."""
        data_config = self.config.get('data', {})
        rsync_config = data_config.get('rsync', {})
        
        if not rsync_config.get('enabled', False):
            logger.info("Rsync disabled in config, skipping...")
            return True
        
        target = rsync_config.get('target', '')
        options = rsync_config.get('options', '-avz --progress')
        ssh_key = rsync_config.get('ssh_key', '')
        
        if not target:
            logger.warning("No rsync target configured")
            return False
        
        source_dir = data_config.get('save_dir', './g1_data_auto')
        
        cmd = ['rsync'] + options.split()
        
        if ssh_key:
            ssh_key = os.path.expanduser(ssh_key)
            cmd.extend(['-e', f'ssh -i {ssh_key}'])
        
        cmd.extend([source_dir + '/', target])
        
        logger.info(f"Syncing to {target}...")
        logger.info(f"  Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                logger.info("Sync completed successfully")
                return True
            else:
                logger.error(f"Sync failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("Sync timed out after 5 minutes")
            return False
        except Exception as e:
            logger.error(f"Sync error: {e}")
            return False
    
    # =========================================================================
    # State Machine Methods
    # =========================================================================
    
    def run_waiting_state(self):
        """WAITING state: Poll for new policy weights."""
        print("\n" + "=" * 60)
        print("[WAITING] Polling policy server for new weights...")
        print(f"  Current: epoch {self.epoch_num}, {self.episode_num} episodes collected")
        print("=" * 60)
        
        poll_interval = self.config.get('policy_server', {}).get('poll_interval_sec', 5)
        
        while self.running and self.state == TrainingState.WAITING:
            status = self.poll_training_status()
            
            if status.get('ready', False):
                current_epoch = status.get('epoch', 0)
                if current_epoch > self.last_policy_epoch:
                    self.last_policy_epoch = current_epoch
                    self.epoch_num = current_epoch
                    self.episode_num = 0
                    
                    logger.info(f"New weights available! Starting EPOCH {self.epoch_num}")
                    self.state = TrainingState.READY
                    return
            
            print(f"  Still waiting... (last check: {time.strftime('%H:%M:%S')})", end='\r')
            time.sleep(poll_interval)
    
    def run_ready_state(self):
        """READY state: Wait for user confirmation to start episode."""
        print("\n" + "=" * 60)
        print(f"[READY] Policy ready (Epoch {self.epoch_num})")
        print(f"  Episodes collected this epoch: {self.episode_num}")
        print(f"  Press 'y' to start episode {self.episode_num + 1}")
        print(f"  Press 'n' to finish this epoch and wait for new training")
        print("=" * 60)
        
        with self.keyboard:
            key = self.keyboard.wait_for_key({'y', 'n'}, "Your choice: ")
            
            if key == 'y':
                self.episode_num += 1
                self.total_episodes += 1
                self.start_recording()
                self.state = TrainingState.EXECUTING
            else:
                logger.info(f"Finishing epoch {self.epoch_num} with {self.episode_num} episodes")
                self.state = TrainingState.SYNCING
    
    def run_executing_state(self):
        """EXECUTING state: Run policy inference with recording."""
        print("\n" + "=" * 60)
        print(f"[EXECUTING] Running policy (epoch {self.epoch_num}, episode {self.episode_num})")
        print("  Press 's' to stop execution and enter labeling mode")
        print(f"  Each policy query returns 50 actions at {self.control_freq}Hz")
        print("=" * 60)
        
        print("  Resetting robot to starting pose...")
        self.reset_to_pose(duration=2.0)
        
        self.current_phase = "policy"
        
        chunk_count = 0
        total_actions = 0
        
        # Wait for convergence between chunks to prevent drift accumulation
        wait_for_convergence = self.config.get('execution', {}).get('wait_for_convergence', True)
        convergence_threshold = self.config.get('execution', {}).get('convergence_threshold_deg', 1.5)
        convergence_timeout = self.config.get('execution', {}).get('convergence_timeout', 0.5)
        
        with self.keyboard:
            while self.running and self.state == TrainingState.EXECUTING:
                try:
                    chunk_count += 1
                    logger.info(f"Querying policy for chunk {chunk_count}...")
                    action_chunk = self.query_policy()
                    
                    actions_executed = self.execute_action_chunk(action_chunk, track_error=True)
                    total_actions += actions_executed
                    
                    if actions_executed < len(action_chunk):
                        logger.info(f"User stopped execution after {total_actions} total actions")
                        self.state = TrainingState.LABELING
                        break
                    
                    # Wait for robot to converge to final position before next chunk
                    if wait_for_convergence and len(action_chunk) > 0:
                        last_action = action_chunk[-1]
                        target_arm = last_action[:14]
                        target_waist = last_action[28] if len(last_action) >= 29 else 0.0
                        
                        converged = self.wait_for_convergence(
                            target_arm, target_waist,
                            threshold_deg=convergence_threshold,
                            timeout=convergence_timeout
                        )
                        if not converged:
                            logger.warning("Did not converge after action chunk")
                    
                except Exception as e:
                    logger.error(f"Policy execution error: {e}")
                    import traceback
                    traceback.print_exc()
                    self.state = TrainingState.LABELING
                    break
        
        logger.info(f"Execution complete: {chunk_count} chunks, {total_actions} total actions")
        if self.recording_active and self.episode_writer:
            logger.info(f"Recorded {self.episode_writer.get_current_length()} frames")
    
    def run_labeling_state(self):
        """LABELING state: Prompt for advantage label."""
        frame_count = 0
        if self.recording_active and self.episode_writer:
            frame_count = self.episode_writer.get_current_length()
        
        print("\n" + "=" * 60)
        print(f"[LABELING] Episode {self.episode_num} execution complete")
        print(f"  Recorded {frame_count} frames ({frame_count/self.control_freq:.1f} seconds)")
        print("  Was this execution successful?")
        print("    'g' - GOOD (Advantage=True)")
        print("    'b' - BAD (Advantage=False)")
        print("    'x' - REJECT - Discard this episode")
        print("=" * 60)
        
        if self.recording_active:
            self.recording_active = False
            logger.info(f"Stopped recording: {frame_count} frames captured")
        
        # Hold position while waiting for input
        self._hold_position_background = True
        hold_thread = threading.Thread(target=self._hold_current_position, daemon=True)
        hold_thread.start()
        
        with self.keyboard:
            key = self.keyboard.wait_for_key({'g', 'b', 'x'}, "Label this episode (g/b/x): ")
            
            if key == 'x':
                logger.info("Episode REJECTED - will not be saved")
                
                self._hold_position_background = False
                hold_thread.join(timeout=0.5)
                
                if self.episode_writer:
                    self.episode_writer = None
                
                self.episode_rejected = True
                self.episode_num -= 1
                self.total_episodes -= 1
                
                print("  Resetting robot to starting pose...")
                self.reset_to_pose(duration=2.0)
                
                self.state = TrainingState.DECIDING
                return
            
            if key == 'g':
                self.current_advantage_label = True
                logger.info("Episode labeled as GOOD (Advantage=True)")
            else:
                self.current_advantage_label = False
                logger.info("Episode labeled as BAD (Advantage=False)")
            
            if self.episode_writer:
                self.episode_writer.set_advantage_label(self.current_advantage_label)
            
            print("\n" + "-" * 60)
            print("  Choose next action:")
            print("    'r' - Reset to starting pose")
            print("    'd' - Enter damping mode (manual adjustment)")
            print("-" * 60)
            
            action_key = self.keyboard.wait_for_key({'r', 'd'}, "Your choice (r/d): ")
        
        self._hold_position_background = False
        hold_thread.join(timeout=0.5)
        
        if action_key == 'r':
            print("  Resetting robot to starting pose...")
            self.reset_to_pose(duration=2.0)
            self.state = TrainingState.SAVING
        else:
            print("  Entering damping mode...")
            self.state = TrainingState.DAMPING
    
    def _hold_current_position(self):
        """Background thread to hold robot at current position."""
        control_period = 1.0 / self.control_freq
        
        while self._hold_position_background and self.running:
            try:
                loop_start = time.time()
                
                current_q = self.robot.get_current_dual_arm_q()
                
                self.robot.ctrl_dual_arm(
                    q_target=current_q,
                    tauff_target=np.zeros(14, dtype=np.float32),
                    use_gravity_compensation=self.use_gravity_compensation
                )
                
                elapsed = time.time() - loop_start
                sleep_time = max(0, control_period - elapsed)
                time.sleep(sleep_time)
            except Exception as e:
                logger.warning(f"Hold position error: {e}")
                break
    
    def run_damping_state(self):
        """DAMPING state: Robot in damping mode for safe adjustment."""
        advantage_str = "GOOD" if self.current_advantage_label else "BAD"
        print("\n" + "=" * 60)
        print(f"[DAMPING] Robot in damping mode (Episode labeled: {advantage_str})")
        print("  Adjust robot pose freely - NO data is being recorded")
        print("  Press 'e' to save episode and continue")
        print("=" * 60)
        
        control_period = 1.0 / self.control_freq
        
        with self.keyboard:
            while self.running and self.state == TrainingState.DAMPING:
                loop_start = time.time()
                
                key = self.keyboard.get_key(timeout=0.001)
                if key and key.lower() == 'e':
                    logger.info("End damping signal received")
                    self.state = TrainingState.SAVING
                    break
                
                current_q = self.robot.get_current_dual_arm_q()
                
                self.robot.ctrl_dual_arm(
                    q_target=current_q,
                    tauff_target=np.zeros(14, dtype=np.float32)
                )
                
                elapsed = time.time() - loop_start
                sleep_time = max(0, control_period - elapsed)
                time.sleep(sleep_time)
    
    def run_saving_state(self):
        """SAVING state: Save the episode with advantage label."""
        advantage_str = "GOOD (Advantage=True)" if self.current_advantage_label else "BAD (Advantage=False)"
        print("\n" + "=" * 60)
        print(f"[SAVING] Saving episode {self.episode_num} (Label: {advantage_str})...")
        print("=" * 60)
        
        if self.episode_writer:
            filepath = self.episode_writer.filepath
            length = self.episode_writer.get_current_length()
            
            self.episode_writer.stop_recording()
            
            logger.info(f"Saved episode: {filepath}")
            logger.info(f"   Epoch: {self.epoch_num}, Episode: {self.episode_num}")
            logger.info(f"   Total frames: {length}")
            logger.info(f"   Advantage: {advantage_str}")
        
        self.current_advantage_label = None
        self.episode_rejected = False
        self.state = TrainingState.DECIDING
    
    def run_deciding_state(self):
        """DECIDING state: User decides to continue or end."""
        print("\n" + "=" * 60)
        
        if self.episode_rejected:
            print(f"[DECIDING] Episode rejected - NOT saved")
            print(f"  Total saved episodes this epoch: {self.episode_num}")
            print("  Press 'y' to retry")
            print("  Press 'n' to finish epoch and sync data")
            self.episode_rejected = False
        else:
            print(f"[DECIDING] Epoch {self.epoch_num} - Episode {self.episode_num} complete")
            print(f"  Total episodes this epoch: {self.episode_num}")
            print("  Press 'y' to collect another episode")
            print("  Press 'n' to finish epoch and sync data")
        
        print("=" * 60)
        
        with self.keyboard:
            key = self.keyboard.wait_for_key({'y', 'n'}, "Your choice: ")
            
            if key == 'y':
                self.state = TrainingState.READY
            else:
                print(f"\nFinish epoch {self.epoch_num} with {self.episode_num} saved episodes?")
                confirm = self.keyboard.wait_for_key({'y', 'n'}, "Confirm (y/n): ")
                
                if confirm == 'y':
                    self.state = TrainingState.SYNCING
                else:
                    self.state = TrainingState.READY
    
    def run_syncing_state(self):
        """SYNCING state: Sync data to remote server."""
        print("\n" + "=" * 60)
        print(f"[SYNCING] Epoch {self.epoch_num} complete with {self.episode_num} episodes")
        print("  Uploading data to remote server...")
        print("=" * 60)
        
        rsync_config = self.config.get('data', {}).get('rsync', {})
        if rsync_config.get('enabled', False):
            success = self.rsync_to_remote()
            if success:
                logger.info("Data synced successfully")
            else:
                logger.warning("Data sync failed - data remains local")
        else:
            logger.info("Auto-sync disabled, skipping...")
        
        print("\n" + "=" * 60)
        print("  Data sync complete!")
        print("  Press 'y' to wait for next training epoch")
        print("  Press 'n' to end training session")
        print("=" * 60)
        
        with self.keyboard:
            key = self.keyboard.wait_for_key({'y', 'n'}, "Your choice: ")
            
            if key == 'y':
                logger.info("Returning to WAITING for next epoch...")
                self.state = TrainingState.WAITING
            else:
                self.state = TrainingState.FINISHED
    
    def run(self, start_immediately: bool = False):
        """Main training loop."""
        print("\n" + "=" * 70)
        print("  G1 Training Pipeline Client")
        print("=" * 70)
        
        if not self.initialize_robot():
            logger.error("Failed to initialize robot, exiting")
            return 1
        
        if not self.initialize_policy_client():
            logger.error("Failed to connect to policy server, exiting")
            return 1
        
        logger.info("Moving robot to home position...")
        self.robot.ctrl_dual_arm_go_home()
        
        print("\n" + "=" * 70)
        print("  Training pipeline ready!")
        print("  Controls:")
        print("    's' - Stop policy execution")
        print("    'g' - Label episode as GOOD (Advantage=True)")
        print("    'b' - Label episode as BAD (Advantage=False)")
        print("    'x' - REJECT episode (discard)")
        print("    'r' - Reset robot to starting pose")
        print("    'd' - Enter damping mode for manual adjustment")
        print("    'e' - End damping mode, save episode")
        print("    'y' - Yes/confirm")
        print("    'n' - No/decline")
        print("    Ctrl+C - Emergency exit")
        print("=" * 70)
        
        if start_immediately:
            logger.info("--start-immediately: Skipping WAITING state")
            self.epoch_num = 0
            self.last_policy_epoch = 0
            self.state = TrainingState.READY
        
        try:
            while self.running and self.state != TrainingState.FINISHED:
                if self.state == TrainingState.WAITING:
                    self.run_waiting_state()
                elif self.state == TrainingState.READY:
                    self.run_ready_state()
                elif self.state == TrainingState.EXECUTING:
                    self.run_executing_state()
                elif self.state == TrainingState.LABELING:
                    self.run_labeling_state()
                elif self.state == TrainingState.DAMPING:
                    self.run_damping_state()
                elif self.state == TrainingState.SAVING:
                    self.run_saving_state()
                elif self.state == TrainingState.DECIDING:
                    self.run_deciding_state()
                elif self.state == TrainingState.SYNCING:
                    self.run_syncing_state()
        
        except Exception as e:
            logger.error(f"Training loop error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
        
        print("\n" + "=" * 70)
        print("  Training session complete!")
        print(f"  Completed {self.epoch_num} epoch(s)")
        print(f"  Total episodes collected: {self.total_episodes}")
        print("=" * 70)
        
        return 0
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up...")
        
        if self.recording_active and self.episode_writer:
            logger.info("Saving any remaining recording...")
            self.episode_writer.stop_recording()
        
        if self.head_camera_client is not None:
            try:
                self.head_camera_client.running = False
                if self.head_camera_thread and self.head_camera_thread.is_alive():
                    self.head_camera_thread.join(timeout=2.0)
            except Exception as e:
                logger.warning(f"Error stopping head camera: {e}")
        
        if hasattr(self, 'head_img_shm') and self.head_img_shm is not None:
            try:
                self.head_img_shm.close()
                self.head_img_shm.unlink()
            except Exception as e:
                logger.warning(f"Error cleaning shared memory: {e}")
        
        if self.hand_ctrl is not None:
            try:
                self.hand_ctrl.stop()
            except Exception as e:
                logger.warning(f"Error stopping hand controller: {e}")
        
        if self.robot:
            logger.info("Moving robot to home position...")
            try:
                self.robot.ctrl_dual_arm_go_home()
            except Exception as e:
                logger.warning(f"Error moving robot to home: {e}")
        
        logger.info("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description="G1 Training Pipeline Client")
    parser.add_argument(
        "--config",
        type=str,
        default="training_config_g1.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--start-immediately",
        action="store_true",
        help="Skip WAITING state and start collecting data immediately"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return 1
    
    client = G1TrainingClient(args.config)
    return client.run(start_immediately=args.start_immediately)


if __name__ == "__main__":
    sys.exit(main())
