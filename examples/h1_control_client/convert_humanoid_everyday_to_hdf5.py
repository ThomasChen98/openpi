"""
Convert Humanoid Everyday dataset to H1 HDF5 format.

This script converts the Humanoid Everyday dataset (https://github.com/ausbxuse/Humanoid-Everyday)
to the HDF5 format used by the H1 control pipeline for visualization and training.

Dataset source: https://humanoideveryday.github.io/

Usage:
    # Convert a single task
    python convert_humanoid_everyday_to_hdf5.py --input_dir ~/Downloads/push_a_button --output_dir ./h1_data_processed/humanoid_everyday/

    # Convert with specific episodes
    python convert_humanoid_everyday_to_hdf5.py --input_dir ~/Downloads/push_a_button --output_dir ./output/ --max_episodes 10

Requirements:
    pip install humanoid-everyday  # or pip install -e /path/to/Humanoid-Everyday
"""

import json
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
import tyro
from tqdm import tqdm


def validate_humanoid_everyday_format(data_dir: Path) -> tuple[bool, str]:
    """Validate that the directory contains Humanoid Everyday format data.
    
    Args:
        data_dir: Path to the dataset directory
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not data_dir.exists():
        return False, f"Directory does not exist: {data_dir}"
    
    # Check for episode directories
    episode_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])
    if not episode_dirs:
        return False, "No episode directories found (expected 'episode_0', 'episode_1', etc.)"
    
    # Check first episode structure
    first_episode = episode_dirs[0]
    data_json = first_episode / "data.json"
    if not data_json.exists():
        return False, f"Missing data.json in {first_episode}"
    
    # Check for image/depth/lidar directories
    color_dir = first_episode / "color"
    if not color_dir.exists():
        return False, f"Missing color directory in {first_episode}"
    
    return True, f"Valid Humanoid Everyday dataset with {len(episode_dirs)} episodes"


def load_humanoid_everyday_episode(episode_dir: Path) -> dict:
    """Load a single episode from Humanoid Everyday format.
    
    Args:
        episode_dir: Path to episode directory
        
    Returns:
        Dictionary with keys:
            - actions: (N, 14) array of arm joint commands
            - qpos: (N, 14) array of arm joint states
            - ego_cam: (N, H, W, 3) array of RGB images
            - depth: (N, H, W) array of depth maps (optional)
            - has_depth: bool indicating if depth data exists
    """
    # Load JSON data
    with open(episode_dir / "data.json", "r") as f:
        data = json.load(f)
    
    num_steps = len(data)
    
    # Extract arm states and actions (14 DoF)
    arm_states = []
    arm_actions = []
    
    for i, step in enumerate(data):
        # Humanoid Everyday arm_state is 14 dimensions
        # This is the sensor reading - actual measured joint positions
        arm_state = np.array(step["states"]["arm_state"], dtype=np.float32)
        arm_states.append(arm_state)
        
        # FIX: Use next timestep's arm_state as the action target
        # This ensures action and state are in the same coordinate frame.
        # Previously we used sol_q (IK motor commands) which has motor offsets
        # that don't match sensor readings, causing ~1.3 rad systematic offset.
        if i < len(data) - 1:
            next_arm_state = np.array(data[i + 1]["states"]["arm_state"], dtype=np.float32)
            arm_actions.append(next_arm_state)
        else:
            # For last timestep, use current state (no further motion)
            arm_actions.append(arm_state.copy())
    
    qpos = np.array(arm_states, dtype=np.float32)  # (N, 14)
    actions = np.array(arm_actions, dtype=np.float32)  # (N, 14)
    
    # Load images
    color_dir = episode_dir / "color"
    # Try .jpg first, then .png
    image_files = sorted(color_dir.glob("*.jpg"))
    if not image_files:
        image_files = sorted(color_dir.glob("*.png"))
    
    if len(image_files) != num_steps:
        print(f"Warning: Found {len(image_files)} images but {num_steps} timesteps in {episode_dir.name}")
    
    images = []
    for img_path in tqdm(image_files, desc=f"Loading images from {episode_dir.name}", leave=False):
        img = cv2.imread(str(img_path))
        if img is not None:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        else:
            print(f"Warning: Failed to load image {img_path}")
    
    ego_cam = np.array(images, dtype=np.uint8)  # (N, 480, 640, 3)
    
    # Load depth maps if available
    depth_dir = episode_dir / "depth"
    has_depth = depth_dir.exists()
    depth_maps = None
    
    if has_depth:
        # Try .jpg first, then .png
        depth_files = sorted(depth_dir.glob("*.jpg"))
        if not depth_files:
            depth_files = sorted(depth_dir.glob("*.png"))
        if len(depth_files) == num_steps:
            depth_maps = []
            for depth_path in tqdm(depth_files, desc=f"Loading depth from {episode_dir.name}", leave=False):
                depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                if depth is not None:
                    depth_maps.append(depth)
            depth_maps = np.array(depth_maps, dtype=np.uint16)  # (N, 480, 640)
        else:
            print(f"Warning: Depth file count mismatch in {episode_dir.name}")
            has_depth = False
    
    return {
        "actions": actions,
        "qpos": qpos,
        "ego_cam": ego_cam,
        "depth": depth_maps,
        "has_depth": has_depth,
    }


def convert_episode_to_hdf5(episode_data: dict, output_path: Path) -> None:
    """Convert a single episode to HDF5 format.
    
    Args:
        episode_data: Episode data dictionary from load_humanoid_everyday_episode
        output_path: Path to output HDF5 file
    """
    with h5py.File(output_path, "w") as f:
        # Store actions and observations
        f.create_dataset("action", data=episode_data["actions"], compression="gzip")
        
        # Create observations group
        obs_group = f.create_group("observations")
        obs_group.create_dataset("qpos", data=episode_data["qpos"], compression="gzip")
        
        # Create images group
        images_group = obs_group.create_group("images")
        images_group.create_dataset("ego_cam", data=episode_data["ego_cam"], compression="gzip")
        
        # Optionally store depth
        if episode_data["has_depth"] and episode_data["depth"] is not None:
            images_group.create_dataset("depth", data=episode_data["depth"], compression="gzip")
        
        # Store metadata
        f.attrs["fps"] = 30
        f.attrs["num_frames"] = len(episode_data["actions"])
        f.attrs["source"] = "humanoid_everyday"
        f.attrs["arm_dof"] = 14


def main(
    input_dir: str,
    output_dir: str,
    *,
    max_episodes: Optional[int] = None,
    skip_depth: bool = False,
) -> None:
    """Convert Humanoid Everyday dataset to H1 HDF5 format.
    
    Args:
        input_dir: Path to Humanoid Everyday task directory (e.g., ~/Downloads/push_a_button)
        output_dir: Output directory for HDF5 files
        max_episodes: Maximum number of episodes to convert (None = all)
        skip_depth: Skip loading depth data (faster conversion)
    """
    input_path = Path(input_dir).expanduser()
    output_path = Path(output_dir).expanduser()
    
    # Validate input format
    print("Validating Humanoid Everyday dataset format...")
    is_valid, message = validate_humanoid_everyday_format(input_path)
    if not is_valid:
        print(f"❌ Invalid dataset format: {message}")
        return
    print(f"✓ {message}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all episode directories
    episode_dirs = sorted([d for d in input_path.iterdir() if d.is_dir() and d.name.startswith("episode_")])
    
    if max_episodes is not None:
        episode_dirs = episode_dirs[:max_episodes]
    
    print(f"\nConverting {len(episode_dirs)} episodes to HDF5 format...")
    print(f"Output directory: {output_path}")
    
    # Convert each episode
    for episode_dir in tqdm(episode_dirs, desc="Converting episodes"):
        episode_num = int(episode_dir.name.split("_")[1])
        output_file = output_path / f"episode_{episode_num}.hdf5"
        
        try:
            # Load episode data
            episode_data = load_humanoid_everyday_episode(episode_dir)
            
            # Convert to HDF5
            convert_episode_to_hdf5(episode_data, output_file)
            
            print(f"  ✓ Converted {episode_dir.name}: {len(episode_data['actions'])} frames → {output_file.name}")
            
        except Exception as e:
            print(f"  ❌ Failed to convert {episode_dir.name}: {e}")
            continue
    
    print(f"\n✓ Conversion complete! Converted {len(episode_dirs)} episodes")
    print(f"\nYou can now visualize with:")
    print(f"  python utils/data_replay.py --hdf5-path {output_path}/episode_0.hdf5")
    print(f"\nOr convert to LeRobot format:")
    print(f"  python convert_h1_data_to_lerobot.py --data_dir {output_path} --repo_id username/task_name")


if __name__ == "__main__":
    tyro.cli(main)



