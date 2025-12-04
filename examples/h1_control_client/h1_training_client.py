#!/usr/bin/env python3
"""
H1-2 Training Pipeline Client

A systematic training pipeline that alternates between:
1. Policy execution (robot runs inference)
2. Human correction (operator adjusts robot in damping mode)

All data is continuously recorded at 50Hz with phase labels.

State Machine:
    WAITING -> READY -> EXECUTING -> LABELING -> DAMPING -> SAVING -> DECIDING -> (loop or SYNCING)

Usage:
    python h1_training_client.py --config training_config.yaml
"""

import argparse
import logging
import os
import signal
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

# Import camera utilities from xr_teleoperate
try:
    xr_teleoperate_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'xr_teleoperate')
    if os.path.exists(xr_teleoperate_path):
        sys.path.insert(0, xr_teleoperate_path)
    from teleop.image_server.image_client import ImageClient
    from teleop.image_server.image_server import RealSenseCamera
    CAMERAS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Camera utilities not available: {e}")
    logger.warning("Will use dummy images for cameras")
    CAMERAS_AVAILABLE = False


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
        """
        Get a single keypress with timeout.
        
        Args:
            timeout: How long to wait for input (seconds)
            
        Returns:
            The key pressed, or None if timeout
        """
        import select
        
        if self.old_settings is None:
            return None
            
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.read(1)
        return None
    
    def wait_for_key(self, valid_keys: set, prompt: str = "") -> str:
        """
        Block until one of the valid keys is pressed.
        
        Args:
            valid_keys: Set of valid key characters
            prompt: Prompt to display
            
        Returns:
            The key that was pressed
        """
        if prompt:
            print(prompt, end='', flush=True)
            
        while True:
            key = self.get_key(timeout=0.1)
            if key and key.lower() in valid_keys:
                print(key)  # Echo the key
                return key.lower()


class H1TrainingClient:
    """
    Training pipeline client for H1-2 robot.
    
    Implements a state machine that alternates between policy execution
    and human correction phases, with continuous data recording.
    
    Includes full camera support (head camera via ZMQ, wrist cameras via RealSense)
    and hand control matching h1_remote_client.py.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize training client.
        
        Args:
            config_path: Path to training_config.yaml
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info(f"Loaded config from {config_path}")
        
        # State tracking
        self.state = TrainingState.WAITING
        self.epoch_num = 0
        self.running = True
        self.last_policy_epoch = -1  # Track which policy epoch we've seen
        
        # Components (initialized lazily)
        self.robot = None
        self.ik_solver = None
        self.policy_client = None
        self.episode_writer = None
        self.keyboard = KeyboardHandler()
        
        # Camera components
        self.head_img_array = None
        self.head_img_shm = None
        self.head_camera_client = None
        self.head_camera_thread = None
        self.left_wrist_camera = None
        self.right_wrist_camera = None
        self.cameras_ready = False
        
        # Recording state
        self.recording_active = False
        self.current_phase = "policy"
        self.current_advantage_label = None  # True = good, False = bad
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\nReceived interrupt signal...")
        self.running = False
        if self.recording_active and self.episode_writer:
            logger.info("Saving current recording before exit...")
            self.episode_writer.stop_recording()
        self.cleanup()
        sys.exit(0)
    
    def initialize_robot(self):
        """Initialize robot controller, IK solver, and cameras"""
        logger.info("Initializing robot components...")
        
        try:
            from robot_control import H1_2_ArmIK, H1_2_ArmController
            
            robot_config = self.config.get('robot', {})
            
            # Initialize IK solver
            logger.info("  Initializing IK solver...")
            self.ik_solver = H1_2_ArmIK(
                Unit_Test=False,
                Visualization=False,
                Hand_Control=True
            )
            logger.info("  IK solver ready")
            
            # Initialize robot controller (includes hand bridges)
            logger.info("  Initializing robot controller...")
            self.robot = H1_2_ArmController(
                simulation_mode=False,
                hand_control=True,
                left_hand_ip=robot_config.get('left_hand_ip', '192.168.123.211'),
                right_hand_ip=robot_config.get('right_hand_ip', '192.168.123.210'),
                network_interface=robot_config.get('network_interface', 'eno1')
            )
            logger.info("  Robot controller ready")
            
            # Initialize cameras in background
            logger.info("  Initializing cameras (in background)...")
            self._init_cameras_background()
            
            logger.info("Robot initialized successfully")
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
                robot_config.get('head_camera_server_ip', '192.168.123.163'),
                robot_config.get('head_camera_server_port', 5555)
            )
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
                logger.info("  Cameras ready: All cameras working!")
            elif len(working_cameras) > 0:
                logger.info(f"  Cameras ready: {', '.join(working_cameras)} working (others using dummy)")
            else:
                logger.info("  Cameras ready: All using dummy images")
            
            self.cameras_ready = True
        
        camera_thread = threading.Thread(target=init_cameras_thread, daemon=True)
        camera_thread.start()
        logger.info("  Cameras initializing in background...")
    
    def _init_head_camera(self, server_ip: str, server_port: int):
        """Initialize head camera client (receives from robot via ZMQ)"""
        if not CAMERAS_AVAILABLE:
            logger.warning("Camera utilities not available, using dummy for head camera")
            return
        
        try:
            logger.info(f"    Connecting to head camera at {server_ip}:{server_port}...")
            
            # Head camera shape: 480x640 RGB
            self.head_img_shape = (480, 640, 3)
            
            # Create shared memory buffer
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
                server_address=server_ip,
                port=server_port,
                image_show=False
            )
            
            # Start receiving thread
            self.head_camera_thread = threading.Thread(
                target=self.head_camera_client.receive_process,
                daemon=True
            )
            self.head_camera_thread.start()
            
            # Wait briefly to check connection
            for _ in range(30):  # 3 seconds
                time.sleep(0.1)
                if np.any(self.head_img_array != 0):
                    logger.info("    Head camera connected!")
                    return
            
            logger.warning("    Head camera connected but no frames yet")
            
        except Exception as e:
            logger.error(f"    Failed to init head camera: {e}")
            self.head_img_array = None
    
    def _init_wrist_cameras(self):
        """Initialize wrist cameras (RealSense D405 on laptop)"""
        if not CAMERAS_AVAILABLE:
            logger.warning("Camera utilities not available, using dummy for wrist cameras")
            return
        
        # RealSense serial numbers
        left_serial = "218622271789"
        right_serial = "241222076627"
        
        # Left wrist
        try:
            logger.info(f"    Connecting to left wrist camera (serial: {left_serial})...")
            left_cam = RealSenseCamera(
                img_shape=[480, 640],
                fps=30,
                serial_number=left_serial
            )
            test_frame = left_cam.get_frame()
            if test_frame is not None:
                logger.info("    Left wrist camera connected!")
                self.left_wrist_camera = left_cam
            else:
                logger.warning("    Left wrist camera opened but no frames")
        except Exception as e:
            logger.warning(f"    Failed to init left wrist camera: {e}")
        
        # Right wrist
        try:
            logger.info(f"    Connecting to right wrist camera (serial: {right_serial})...")
            right_cam = RealSenseCamera(
                img_shape=[480, 640],
                fps=30,
                serial_number=right_serial
            )
            test_frame = right_cam.get_frame()
            if test_frame is not None:
                logger.info("    Right wrist camera connected!")
                self.right_wrist_camera = right_cam
            else:
                logger.warning("    Right wrist camera opened but no frames")
        except Exception as e:
            logger.warning(f"    Failed to init right wrist camera: {e}")
    
    def initialize_policy_client(self):
        """Initialize connection to policy server"""
        try:
            from openpi_client import websocket_client_policy
            
            server_config = self.config.get('policy_server', {})
            host = server_config.get('host', 'localhost')
            port = server_config.get('port', 8000)
            
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
        """
        Poll the policy server for training status.
        
        Returns:
            Dict with 'ready' (bool) and 'epoch' (int) fields
        """
        server_config = self.config.get('policy_server', {})
        host = server_config.get('host', 'localhost')
        port = server_config.get('port', 8000)
        
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
        """Start recording a new episode"""
        from utils.episode_writer_hdf5 import EpisodeWriterHDF5
        
        # Get recording settings
        recording_config = self.config.get('recording', {})
        fps = recording_config.get('fps', 50)
        
        # Get data save directory (supports both old and new config structure)
        data_config = self.config.get('data', {})
        base_save_dir = data_config.get('save_dir', 
            recording_config.get('save_dir', './data/training_epochs'))
        
        # Get task name (supports both old and new config structure)
        task_config = self.config.get('task', {})
        task_name = task_config.get('name',
            self.config.get('training', {}).get('task_name', 'training_session'))
        
        # Use epoch-based directory structure: {base_dir}/{task_name}/epoch_{N}/raw/
        # This matches the structure expected by integrated_training.sh
        epoch_dir = os.path.join(base_save_dir, task_name, f"epoch_{self.epoch_num}", "raw")
        
        self.episode_writer = EpisodeWriterHDF5(
            save_dir=epoch_dir,
            label_name="",  # Episodes saved directly in epoch_dir
            fps=fps
        )
        self.episode_writer.start_recording()
        self.recording_active = True
        logger.info(f"Started recording: {self.episode_writer.filepath}")
        logger.info(f"  Task: {task_name}, Epoch: {self.epoch_num}")
    
    def stop_recording(self):
        """Stop and save the current recording (emergency/cleanup use)"""
        if self.episode_writer:
            # Check if we have data to save
            if self.episode_writer.get_current_length() > 0:
                filepath = self.episode_writer.filepath
                length = self.episode_writer.get_current_length()
                
                # If no advantage label was set, default to False (interrupted episodes are "bad")
                if self.episode_writer.get_advantage_label() is None:
                    self.episode_writer.set_advantage_label(False)
                    logger.warning("No advantage label set, defaulting to False (interrupted)")
                
                self.episode_writer.stop_recording()
                
                logger.info(f"Saved episode: {filepath}")
                logger.info(f"  Total timesteps: {length}")
            
            self.recording_active = False
    
    def get_observation(self) -> dict:
        """Get current observation from robot with real camera feeds"""
        # Get current joint positions
        current_q = self.robot.get_current_dual_arm_q()
        
        # Dummy image for fallback
        dummy_image = np.full((224, 224, 3), 128, dtype=np.uint8)
        
        images = {}
        
        # Head camera (from robot via ZMQ)
        if self.head_img_array is not None:
            try:
                head_frame = self.head_img_array.copy()
                if np.any(head_frame != 0):
                    head_image = cv2.resize(head_frame, (224, 224))
                    head_image = cv2.cvtColor(head_image, cv2.COLOR_BGR2RGB)
                    images["ego_cam"] = head_image
                else:
                    images["ego_cam"] = dummy_image
            except Exception:
                images["ego_cam"] = dummy_image
        else:
            images["ego_cam"] = dummy_image
        
        # Left wrist camera
        if self.left_wrist_camera is not None:
            try:
                left_frame = self.left_wrist_camera.get_frame()
                if left_frame is not None:
                    left_image = cv2.resize(left_frame, (224, 224))
                    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                    images["cam_left_wrist"] = left_image
                else:
                    images["cam_left_wrist"] = dummy_image
            except Exception:
                images["cam_left_wrist"] = dummy_image
        else:
            images["cam_left_wrist"] = dummy_image
        
        # Right wrist camera
        if self.right_wrist_camera is not None:
            try:
                right_frame = self.right_wrist_camera.get_frame()
                if right_frame is not None:
                    right_image = cv2.resize(right_frame, (224, 224))
                    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                    images["cam_right_wrist"] = right_image
                else:
                    images["cam_right_wrist"] = dummy_image
            except Exception:
                images["cam_right_wrist"] = dummy_image
        else:
            images["cam_right_wrist"] = dummy_image
        
        return {
            "state": current_q,
            "images": images,
        }
    
    def compute_gravity_compensation(self, joint_positions: np.ndarray) -> np.ndarray:
        """Compute gravity compensation torques"""
        import pinocchio as pin
        
        zero_velocity = np.zeros(self.ik_solver.reduced_robot.model.nv)
        zero_acceleration = np.zeros(self.ik_solver.reduced_robot.model.nv)
        
        gravity_torques = pin.rnea(
            self.ik_solver.reduced_robot.model,
            self.ik_solver.reduced_robot.data,
            joint_positions,
            zero_velocity,
            zero_acceleration
        )
        
        return gravity_torques
    
    def execute_policy_step(self) -> np.ndarray:
        """Execute one policy inference step and return the action"""
        obs = self.get_observation()
        
        # Get task description (supports both old and new config structure)
        task_config = self.config.get('task', {})
        task_description = task_config.get('description',
            self.config.get('training', {}).get('default_label', 'manipulation task'))
        
        # Remap image keys to match policy expectations
        # OpenPi model expects (cam_head, cam_left_wrist, cam_right_wrist)
        raw_images = obs["images"]
        obs["images"] = {
            "cam_head": raw_images.get("ego_cam", raw_images.get("cam_head")),
            "cam_left_wrist": raw_images["cam_left_wrist"],
            "cam_right_wrist": raw_images["cam_right_wrist"],
        }
        
        # Add prompt with Advantage=True to sample from good distribution
        obs["prompt"] = f"{task_description}, Advantage=True"
        
        # Get action from policy
        response = self.policy_client.infer(obs)
        action = response["actions"][0]  # Take first action from chunk
        
        return action
    
    def rsync_to_remote(self) -> bool:
        """
        Sync recorded data to remote server using rsync.
        
        Returns:
            True if sync succeeded, False otherwise
        """
        rsync_config = self.config.get('rsync', {})
        
        if not rsync_config.get('enabled', False):
            logger.info("Rsync disabled in config, skipping...")
            return True
        
        target = rsync_config.get('target', '')
        options = rsync_config.get('options', '-avz --progress')
        ssh_key = rsync_config.get('ssh_key', '')
        
        if not target:
            logger.warning("No rsync target configured")
            return False
        
        recording_config = self.config.get('recording', {})
        source_dir = recording_config.get('save_dir', './data/training_epochs')
        
        # Build rsync command
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
    
    def run_waiting_state(self):
        """WAITING state: Poll for new policy weights"""
        print("\n" + "=" * 60)
        print("[WAITING] Polling policy server for new weights...")
        print("=" * 60)
        
        poll_interval = self.config.get('policy_server', {}).get('poll_interval_sec', 5)
        
        while self.running and self.state == TrainingState.WAITING:
            status = self.poll_training_status()
            
            if status.get('ready', False):
                current_epoch = status.get('epoch', 0)
                if current_epoch > self.last_policy_epoch:
                    self.last_policy_epoch = current_epoch
                    logger.info(f"New weights available! Policy epoch: {current_epoch}")
                    self.state = TrainingState.READY
                    return
            
            print(f"  Still waiting... (last check: {time.strftime('%H:%M:%S')})", end='\r')
            time.sleep(poll_interval)
    
    def run_ready_state(self):
        """READY state: Wait for user confirmation to start epoch"""
        print("\n" + "=" * 60)
        print(f"[READY] New weights available!")
        print(f"  Press 'y' to start epoch {self.epoch_num + 1}, 'n' to skip")
        print("=" * 60)
        
        with self.keyboard:
            key = self.keyboard.wait_for_key({'y', 'n'}, "Your choice: ")
            
            if key == 'y':
                self.epoch_num += 1
                self.start_recording()
                self.state = TrainingState.EXECUTING
            else:
                logger.info("Skipped this epoch")
                self.state = TrainingState.WAITING
    
    def run_executing_state(self):
        """EXECUTING state: Run policy inference with recording"""
        print("\n" + "=" * 60)
        print(f"[EXECUTING] Running policy (epoch {self.epoch_num})")
        print("  Press 's' to stop execution and enter damping mode")
        print("=" * 60)
        
        self.current_phase = "policy"
        self.robot.speed_instant_max()
        
        control_rate = 50  # Hz
        control_period = 1.0 / control_rate
        
        with self.keyboard:
            while self.running and self.state == TrainingState.EXECUTING:
                loop_start = time.time()
                
                # Check for stop key
                key = self.keyboard.get_key(timeout=0.001)
                if key and key.lower() == 's':
                    logger.info("Stop signal received")
                    self.state = TrainingState.LABELING
                    break
                
                # Execute policy step
                try:
                    action = self.execute_policy_step()
                    
                    # Compute gravity compensation
                    gravity_torques = self.compute_gravity_compensation(action)
                    
                    # Send command to robot
                    self.robot.ctrl_dual_arm(
                        q_target=action,
                        tauff_target=gravity_torques
                    )
                    
                    # Record timestep
                    current_q = self.robot.get_current_dual_arm_q()
                    obs = self.get_observation()
                    self.episode_writer.add_timestep(
                        qpos=current_q,
                        action=action,
                        images=obs.get('images'),
                        phase="policy"
                    )
                    
                except Exception as e:
                    logger.error(f"Policy execution error: {e}")
                
                # Maintain control rate
                elapsed = time.time() - loop_start
                sleep_time = max(0, control_period - elapsed)
                time.sleep(sleep_time)
    
    def run_labeling_state(self):
        """LABELING state: Stop recording and prompt for advantage label"""
        print("\n" + "=" * 60)
        print("[LABELING] Policy execution complete")
        print("  Was this execution successful?")
        print("    'g' - GOOD (Advantage=True) - Task completed successfully")
        print("    'b' - BAD (Advantage=False) - Needs improvement")
        print("=" * 60)
        
        # Stop recording immediately - we only record during policy execution
        if self.recording_active and self.episode_writer:
            # Don't save yet - just stop adding frames
            self.recording_active = False
            logger.info(f"Stopped recording: {self.episode_writer.get_current_length()} frames captured")
        
        with self.keyboard:
            key = self.keyboard.wait_for_key({'g', 'b'}, "Label this episode (g/b): ")
            
            if key == 'g':
                self.current_advantage_label = True
                logger.info("Episode labeled as GOOD (Advantage=True)")
            else:
                self.current_advantage_label = False
                logger.info("Episode labeled as BAD (Advantage=False)")
            
            # Set the advantage label on the episode writer
            if self.episode_writer:
                self.episode_writer.set_advantage_label(self.current_advantage_label)
        
        self.state = TrainingState.DAMPING
    
    def run_damping_state(self):
        """DAMPING state: Robot in damping mode for safe adjustment (NO recording)"""
        advantage_str = "GOOD" if self.current_advantage_label else "BAD"
        print("\n" + "=" * 60)
        print(f"[DAMPING] Robot in damping mode (Episode labeled: {advantage_str})")
        print("  Adjust robot pose freely - NO data is being recorded")
        print("  Press 'e' to save episode and continue")
        print("=" * 60)
        
        # Enter damping mode
        self.robot.enter_damping_mode()
        
        control_rate = 50  # Hz
        control_period = 1.0 / control_rate
        
        with self.keyboard:
            while self.running and self.state == TrainingState.DAMPING:
                loop_start = time.time()
                
                # Check for end key
                key = self.keyboard.get_key(timeout=0.001)
                if key and key.lower() == 'e':
                    logger.info("End epoch signal received")
                    self.state = TrainingState.SAVING
                    break
                
                # NO recording in damping mode - just maintain gravity compensation
                current_q = self.robot.get_current_dual_arm_q()
                gravity_torques = self.compute_gravity_compensation(current_q)
                self.robot.ctrl_dual_arm(
                    q_target=current_q,  # Target = current (no movement)
                    tauff_target=gravity_torques
                )
                
                # Maintain control rate
                elapsed = time.time() - loop_start
                sleep_time = max(0, control_period - elapsed)
                time.sleep(sleep_time)
        
        # Exit damping mode
        self.robot.exit_damping_mode()
    
    def run_saving_state(self):
        """SAVING state: Save the episode with advantage label"""
        advantage_str = "GOOD (Advantage=True)" if self.current_advantage_label else "BAD (Advantage=False)"
        print("\n" + "=" * 60)
        print(f"[SAVING] Saving episode data (Label: {advantage_str})...")
        print("=" * 60)
        
        # The advantage label was already set in LABELING state
        # Now we just save the episode
        if self.episode_writer:
            filepath = self.episode_writer.filepath
            length = self.episode_writer.get_current_length()
            
            self.episode_writer.stop_recording()
            
            logger.info(f"Saved episode: {filepath}")
            logger.info(f"  Total timesteps: {length}")
            logger.info(f"  Advantage: {advantage_str}")
        
        # Reset for next episode
        self.current_advantage_label = None
        self.state = TrainingState.DECIDING
    
    def run_deciding_state(self):
        """DECIDING state: User decides to continue or end"""
        print("\n" + "=" * 60)
        print("[DECIDING] What's next?")
        print("  Press 'y' to start another epoch")
        print("  Press 'n' to finish training session")
        print("=" * 60)
        
        with self.keyboard:
            key = self.keyboard.wait_for_key({'y', 'n'}, "Your choice: ")
            
            if key == 'y':
                self.state = TrainingState.READY
            else:
                # Confirm end
                print("\nAre you sure you want to end this training session?")
                confirm = self.keyboard.wait_for_key({'y', 'n'}, "Confirm (y/n): ")
                
                if confirm == 'y':
                    self.state = TrainingState.SYNCING
                else:
                    self.state = TrainingState.READY
    
    def run_syncing_state(self):
        """SYNCING state: Sync data to remote server"""
        print("\n" + "=" * 60)
        print("[SYNCING] Uploading data to remote server...")
        print("=" * 60)
        
        training_config = self.config.get('training', {})
        if training_config.get('auto_sync', True):
            success = self.rsync_to_remote()
            if success:
                logger.info("Data synced successfully")
            else:
                logger.warning("Data sync failed - data remains local")
        else:
            logger.info("Auto-sync disabled, skipping...")
        
        self.state = TrainingState.FINISHED
    
    def run(self):
        """Main training loop"""
        print("\n" + "=" * 70)
        print("  H1-2 Training Pipeline Client")
        print("=" * 70)
        
        # Initialize robot
        if not self.initialize_robot():
            logger.error("Failed to initialize robot, exiting")
            return 1
        
        # Initialize policy client
        if not self.initialize_policy_client():
            logger.error("Failed to connect to policy server, exiting")
            return 1
        
        # Move to home position
        logger.info("Moving robot to home position...")
        self.robot.ctrl_dual_arm_go_home()
        
        print("\n" + "=" * 70)
        print("  Training pipeline ready!")
        print("  Controls:")
        print("    's' - Stop policy execution")
        print("    'g' - Label episode as GOOD (Advantage=True)")
        print("    'b' - Label episode as BAD (Advantage=False)")
        print("    'e' - End damping mode, save episode")
        print("    'y' - Yes/confirm")
        print("    'n' - No/decline")
        print("    Ctrl+C - Emergency exit")
        print("=" * 70)
        
        # State machine loop
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
        print("=" * 70)
        
        return 0
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up...")
        
        # Save any remaining recording
        if self.recording_active and self.episode_writer:
            logger.info("Saving any remaining recording...")
            self.episode_writer.stop_recording()
        
        # Stop head camera client
        if self.head_camera_client is not None:
            try:
                self.head_camera_client.running = False
                if self.head_camera_thread and self.head_camera_thread.is_alive():
                    self.head_camera_thread.join(timeout=2.0)
            except Exception as e:
                logger.warning(f"Error stopping head camera: {e}")
        
        # Cleanup shared memory
        if self.head_img_shm is not None:
            try:
                self.head_img_shm.close()
                self.head_img_shm.unlink()
            except Exception as e:
                logger.warning(f"Error cleaning shared memory: {e}")
        
        # Release wrist cameras
        if self.left_wrist_camera is not None:
            try:
                self.left_wrist_camera.release()
            except Exception:
                pass
        
        if self.right_wrist_camera is not None:
            try:
                self.right_wrist_camera.release()
            except Exception:
                pass
        
        # Move robot to home
        if self.robot:
            logger.info("Moving robot to home position...")
            try:
                if self.robot.is_damping_mode():
                    self.robot.exit_damping_mode()
                self.robot.ctrl_dual_arm_go_home()
            except Exception as e:
                logger.warning(f"Error moving robot to home: {e}")
        
        logger.info("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description="H1-2 Training Pipeline Client")
    parser.add_argument(
        "--config",
        type=str,
        default="training_config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Check config exists
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return 1
    
    client = H1TrainingClient(args.config)
    return client.run()


if __name__ == "__main__":
    sys.exit(main())

