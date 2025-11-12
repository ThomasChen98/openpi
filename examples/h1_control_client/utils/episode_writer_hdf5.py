"""
HDF5 Episode Writer for H1 Policy Execution Recording

Saves policy execution data in HDF5 format compatible with the OpenPI training pipeline.

File structure:
/observations/
    qpos: [T, 14] - arm joint positions
    images/
        ego_cam: [T,] - JPEG compressed images
        cam_left_wrist: [T,] - JPEG compressed images
        cam_right_wrist: [T,] - JPEG compressed images
/action: [T, 14] - predicted joint positions from policy
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
    def __init__(self, save_dir='./data/policy_episodes', label_name='default', fps=50):
        """
        Initialize HDF5 episode writer for policy execution recording
        
        Args:
            save_dir: Base directory to save episodes (default: './data/policy_episodes')
            label_name: Label/task name for organizing episodes (default: 'default')
            fps: Recording frequency in Hz (default: 50)
        """
        self.base_dir = save_dir
        self.label_name = label_name
        self.fps = fps
        
        # Create subdirectory for this label
        self.save_dir = os.path.join(save_dir, label_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Find next episode number by scanning the directory
        self.episode_idx = 0
        while os.path.exists(os.path.join(self.save_dir, f'episode_{self.episode_idx}.hdf5')):
            self.episode_idx += 1
        
        self.filepath = os.path.join(self.save_dir, f'episode_{self.episode_idx}.hdf5')
        
        # Data buffers
        self.qpos_buffer = []  # Executed joint positions
        self.action_buffer = []  # Predicted joint positions
        self.image_buffers = {}  # {camera_name: [frames]}
        
        self.recording = False
        
        logger.info(f"EpisodeWriterHDF5 initialized for label '{label_name}': {self.filepath}")
    
    def start_recording(self):
        """Start a new episode recording"""
        self.recording = True
        self.qpos_buffer = []
        self.action_buffer = []
        self.image_buffers = {}
        logger.info(f"Started recording episode {self.episode_idx}")
    
    def add_timestep(self, qpos, action, images=None):
        """
        Add a single timestep to the episode
        
        Args:
            qpos: Executed joint positions array [14] (arm joints)
            action: Predicted joint positions array [14] (policy output)
            images: Dict of {camera_name: image_array} where image is [H, W, 3] RGB uint8
        """
        if not self.recording:
            return
        
        self.qpos_buffer.append(np.array(qpos, dtype=np.float32))
        self.action_buffer.append(np.array(action, dtype=np.float32))
        
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
            qpos_data = np.array(self.qpos_buffer, dtype=np.float32)  # [T, 14]
            action_data = np.array(self.action_buffer, dtype=np.float32)  # [T, 14]
            
            episode_length = len(self.qpos_buffer)
            
            logger.info(f"Saving episode {self.episode_idx}: {episode_length} timesteps")
            
            # Save to HDF5
            with h5py.File(self.filepath, 'w') as f:
                # Create observations group
                obs_group = f.create_group('observations')
                obs_group.create_dataset('qpos', data=qpos_data, compression='gzip')
                
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
                
                # Save actions (predicted joint positions)
                f.create_dataset('action', data=action_data, compression='gzip')
                
                # Save metadata as attributes
                f.attrs['episode_length'] = episode_length
                f.attrs['fps'] = self.fps
                f.attrs['timestamp'] = datetime.now().isoformat()
            
            logger.info(f"Episode saved successfully: {self.filepath}")
            logger.info(f"  Label: {self.label_name}")
            logger.info(f"  qpos: {qpos_data.shape}, action: {action_data.shape}")
            
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

