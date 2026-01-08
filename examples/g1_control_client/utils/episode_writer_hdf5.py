"""
HDF5 Episode Writer for G1 Policy Execution Recording

Saves policy execution data in HDF5 format compatible with the OpenPI training pipeline.
Includes locomotion data (loco_state, loco_action) for hybrid locomotion control.

File structure:
/observations/
    qpos: [T, 28] - arm (14) + hand (14) joint positions
    loco_state: [T, 17] - locomotion state from lowstate
        [0]     mode_machine    - FSM state (0=zero torque, 1=damp, etc.)
        [1:4]   rpy             - roll, pitch, yaw (radians)
        [4:8]   quaternion      - orientation quaternion (w, x, y, z)
        [8:11]  accelerometer   - linear acceleration (x, y, z)
        [11:14] gyroscope       - angular velocity (x, y, z)
        [14:17] leg_joints      - knee joint positions (left, right, avg) as height proxy
    images/
        cam_head: [T,] - JPEG compressed images (ego camera)
/action: [T, 28] - upper body joint targets (arm + hand)
/loco_action: [T, 20] - controller inputs (joysticks + buttons)
    [0:4]   joysticks       - Lx, Ly, Rx, Ry (float, -1 to 1)
    [4:20]  buttons         - L1, L2, R1, R2, A, B, X, Y, Up, Down, Left, Right, Select, Start, F1, F3
/phase: [T,] - string labels ("policy" or "human") for training pipeline

Metadata attributes:
    advantage: bool - True if episode was labeled as good, False if bad
    episode_length: int - number of timesteps
    fps: int - recording frequency
    robot_name: str - "G1_29"
    timestamp: str - ISO format timestamp
    has_loco_data: bool - whether locomotion data is present
    policy_frames: int - count of policy execution frames
    human_frames: int - count of human adjustment frames
"""

import h5py
import numpy as np
import os
from datetime import datetime
from PIL import Image
import io
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class EpisodeWriterHDF5:
    def __init__(self, save_dir='./g1_data_auto', label_name='', fps=30):
        """
        Initialize HDF5 episode writer for G1 policy execution recording
        
        Args:
            save_dir: Base directory to save episodes (default: './g1_data_auto')
            label_name: Label/task name for organizing episodes (default: '')
            fps: Recording frequency in Hz (default: 30)
        """
        self.base_dir = save_dir
        self.label_name = label_name
        self.fps = fps
        self.robot_name = "G1_29"
        
        # Create subdirectory for this label (or use save_dir directly if label_name is empty)
        if label_name:
            self.save_dir = os.path.join(save_dir, label_name)
        else:
            self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Find next episode number by scanning the directory
        self.episode_idx = 0
        while os.path.exists(os.path.join(self.save_dir, f'episode_{self.episode_idx}.hdf5')):
            self.episode_idx += 1
        
        self.filepath = os.path.join(self.save_dir, f'episode_{self.episode_idx}.hdf5')
        
        # Data buffers
        self.qpos_buffer = []       # [T, 28] - arm (14) + hand (14) joint positions
        self.action_buffer = []     # [T, 28] - upper body joint targets
        self.image_buffers = {}     # {camera_name: [frames]}
        self.phase_buffer = []      # Phase labels ("policy" or "human")
        
        # Locomotion data buffers
        self.loco_state_buffer = []   # [T, 17] - locomotion state
        self.loco_action_buffer = []  # [T, 20] - joysticks (4) + buttons (16)
        
        # Advantage labeling (per-episode)
        self.advantage_label = None  # True = good, False = bad, None = unlabeled
        
        self.recording = False
        
        logger.info(f"G1 EpisodeWriterHDF5 initialized: {self.filepath}")
    
    def start_recording(self):
        """Start a new episode recording"""
        self.recording = True
        self.qpos_buffer = []
        self.action_buffer = []
        self.image_buffers = {}
        self.phase_buffer = []
        self.loco_state_buffer = []
        self.loco_action_buffer = []
        self.advantage_label = None  # Reset advantage label for new episode
        logger.info(f"Started recording episode {self.episode_idx}")
    
    def set_advantage_label(self, is_good: bool):
        """
        Set the advantage label for the current episode.
        
        Args:
            is_good: True if episode was successful/good, False if it needs correction
        """
        self.advantage_label = is_good
        label_str = "GOOD (Advantage=True)" if is_good else "BAD (Advantage=False)"
        logger.info(f"Episode {self.episode_idx} labeled as: {label_str}")
    
    def add_timestep(
        self, 
        qpos, 
        action, 
        images=None, 
        loco_state=None, 
        loco_action=None, 
        phase="policy"
    ):
        """
        Add a single timestep to the episode
        
        Args:
            qpos: Joint positions array [28] - arm (14) + hand (14)
            action: Target joint positions array [28] - upper body targets
            images: Dict of {camera_name: image_array} where image is [H, W, 3] RGB uint8
            loco_state: Locomotion state array [17] (optional)
                [mode_machine, rpy(3), quaternion(4), accel(3), gyro(3), leg_joints(3)]
            loco_action: Controller input array [20] (optional)
                [joysticks(4): Lx,Ly,Rx,Ry, buttons(16)]
                During policy execution, this can be zeros or the policy's vx,vy,vyaw
            phase: Phase label - "policy" (robot executing) or "human" (operator adjusting)
        """
        if not self.recording:
            return
        
        self.qpos_buffer.append(np.array(qpos, dtype=np.float32))
        self.action_buffer.append(np.array(action, dtype=np.float32))
        self.phase_buffer.append(phase)
        
        # Store locomotion data if provided
        if loco_state is not None:
            self.loco_state_buffer.append(np.array(loco_state, dtype=np.float32))
        
        if loco_action is not None:
            self.loco_action_buffer.append(np.array(loco_action, dtype=np.float32))
        
        if images is not None:
            for camera_name, image in images.items():
                if camera_name not in self.image_buffers:
                    self.image_buffers[camera_name] = []
                # Ensure uint8 RGB format
                if image.dtype != np.uint8:
                    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
                self.image_buffers[camera_name].append(image)
    
    def stop_recording(self):
        """Stop recording and save to HDF5 file with JPEG compression"""
        if not self.recording:
            logger.warning("Not currently recording")
            return
        
        self.recording = False
        
        if len(self.qpos_buffer) == 0:
            logger.warning("No data to save")
            return
        
        try:
            # Convert buffers to numpy arrays
            qpos_data = np.array(self.qpos_buffer, dtype=np.float32)      # [T, 28]
            action_data = np.array(self.action_buffer, dtype=np.float32)  # [T, 28]
            
            episode_length = len(self.qpos_buffer)
            
            # Check if we have locomotion data
            has_loco_data = len(self.loco_state_buffer) > 0
            
            logger.info(f"Saving episode {self.episode_idx}: {episode_length} timesteps")
            
            # Save to HDF5
            with h5py.File(self.filepath, 'w') as f:
                # Create observations group
                obs_group = f.create_group('observations')
                obs_group.create_dataset('qpos', data=qpos_data, compression='gzip')
                
                # Save locomotion state if available
                if has_loco_data and self.loco_state_buffer:
                    loco_state_data = np.array(self.loco_state_buffer, dtype=np.float32)
                    obs_group.create_dataset('loco_state', data=loco_state_data, compression='gzip')
                    logger.info(f"  Saved loco_state: {loco_state_data.shape}")
                
                # Create images subgroup with JPEG compression
                if self.image_buffers:
                    logger.info(f"Compressing and saving {len(self.image_buffers)} camera streams...")
                    images_group = obs_group.create_group('images')
                    
                    for camera_name, frames in self.image_buffers.items():
                        logger.info(f"  Compressing {camera_name} ({len(frames)} frames)...")
                        
                        # Compress each frame to JPEG
                        compressed_frames = []
                        for frame in tqdm(frames, desc=f"  {camera_name}", unit="frame"):
                            # Convert to PIL Image
                            pil_img = Image.fromarray(frame)
                            # Compress to JPEG
                            buf = io.BytesIO()
                            pil_img.save(buf, format='JPEG', quality=85)
                            compressed_frames.append(np.frombuffer(buf.getvalue(), dtype=np.uint8))
                        
                        # Save as variable-length dataset
                        dt = h5py.vlen_dtype(np.dtype('uint8'))
                        images_group.create_dataset(
                            camera_name,
                            (len(compressed_frames),),
                            dtype=dt
                        )
                        
                        for i, compressed_frame in enumerate(compressed_frames):
                            images_group[camera_name][i] = compressed_frame
                        
                        logger.info(f"  Saved {camera_name}: {len(frames)} JPEG frames")
                else:
                    logger.warning("No images to save")
                
                # Save actions (upper body joint targets)
                f.create_dataset('action', data=action_data, compression='gzip')
                
                # Save locomotion actions if available
                if has_loco_data and self.loco_action_buffer:
                    loco_action_data = np.array(self.loco_action_buffer, dtype=np.float32)
                    f.create_dataset('loco_action', data=loco_action_data, compression='gzip')
                    logger.info(f"  Saved loco_action: {loco_action_data.shape}")
                
                # Save phase labels as fixed-length strings
                phase_data = np.array(self.phase_buffer, dtype='S10')
                f.create_dataset('phase', data=phase_data, compression='gzip')
                
                # Save metadata as attributes
                f.attrs['episode_length'] = episode_length
                f.attrs['fps'] = self.fps
                f.attrs['robot_name'] = self.robot_name
                f.attrs['timestamp'] = datetime.now().isoformat()
                f.attrs['has_loco_data'] = has_loco_data
                
                # Save advantage label (per-episode)
                if self.advantage_label is not None:
                    f.attrs['advantage'] = self.advantage_label
                else:
                    # Default to False if not labeled
                    f.attrs['advantage'] = False
                    logger.warning(f"Episode {self.episode_idx} has no advantage label, defaulting to False")
                
                # Count phase statistics
                policy_frames = sum(1 for p in self.phase_buffer if p == "policy")
                human_frames = sum(1 for p in self.phase_buffer if p == "human")
                f.attrs['policy_frames'] = policy_frames
                f.attrs['human_frames'] = human_frames
            
            advantage_str = "True (good)" if self.advantage_label else "False (needs improvement)"
            logger.info(f"Episode saved successfully: {self.filepath}")
            logger.info(f"  Robot: {self.robot_name}")
            logger.info(f"  Advantage: {advantage_str}")
            logger.info(f"  qpos: {qpos_data.shape}, action: {action_data.shape}")
            logger.info(f"  Phase breakdown: {policy_frames} policy, {human_frames} human")
            if has_loco_data:
                logger.info(f"  Has locomotion data: True")
            
            # Prepare for next episode (increment and update filepath)
            self.episode_idx += 1
            self.filepath = os.path.join(self.save_dir, f'episode_{self.episode_idx}.hdf5')
            
        except Exception as e:
            logger.error(f"Error saving episode: {e}")
            import traceback
            traceback.print_exc()
    
    def is_recording(self):
        """Check if currently recording"""
        return self.recording
    
    def get_current_length(self):
        """Get number of timesteps in current recording"""
        return len(self.qpos_buffer)
    
    def get_phase_counts(self):
        """Get count of frames per phase in current recording"""
        policy_count = sum(1 for p in self.phase_buffer if p == "policy")
        human_count = sum(1 for p in self.phase_buffer if p == "human")
        return {"policy": policy_count, "human": human_count}
    
    def get_advantage_label(self):
        """Get the current advantage label (True/False/None)"""
        return self.advantage_label
