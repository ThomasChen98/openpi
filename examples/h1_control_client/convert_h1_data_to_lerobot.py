"""
Script for converting H1 HDF5 dataset to LeRobot format.

Supports advantage labeling for training with advantage-augmented prompts:
  - human_labeling: Read advantage label from HDF5 metadata (set during data collection)
  - reward_labeling: Use a reward function to determine advantage (not implemented)
  - none: No advantage labeling (original behavior)

The prompt format with advantage labeling is:
  "task_description, Advantage=True" or "task_description, Advantage=False"

Usage:
uv run examples/h1_control_client/convert_h1_data_to_lerobot.py --data_dir training_data/h1/circular.hdf5 --task_description "pick up the cube" --num_repeats 10

For a directory with multiple HDF5 files:
uv run examples/h1_control_client/convert_h1_data_to_lerobot.py --data_dir examples/h1_control_client/h1_data_processed/box_action/good/ --task_description "move the lid" --num_repeats 4

With advantage labeling:
uv run examples/h1_control_client/convert_h1_data_to_lerobot.py --data_dir ./data/ --task_description "press button" --labeling_mode human_labeling

Note: Install h5py if needed: `uv pip install h5py`
Note: Install opencv-python for image resizing: `uv pip install opencv-python`

The resulting dataset will be saved to the $HF_LEROBOT_HOME directory.
"""

import shutil
from pathlib import Path
from typing import Literal

import os
import cv2
import h5py
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

# Supported labeling modes
LabelingMode = Literal["none", "human_labeling", "reward_labeling"]


def resize_image(image: np.ndarray, target_height: int = 224, target_width: int = 224) -> np.ndarray:
    """Resize image to target dimensions."""
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def decompress_jpeg_images(compressed_data) -> np.ndarray:
    """Decompress JPEG-compressed images from HDF5 variable-length arrays.
    
    Args:
        compressed_data: Array of variable-length uint8 arrays (JPEG bytes)
        
    Returns:
        numpy array of shape (num_frames, height, width, 3)
    """
    from PIL import Image
    import io
    
    frames = []
    for jpeg_bytes in compressed_data:
        # Decompress JPEG
        img = Image.open(io.BytesIO(bytes(jpeg_bytes)))
        frames.append(np.array(img))
    
    return np.array(frames)


def load_episode_from_hdf5(hdf5_path: str, read_advantage: bool = False) -> dict:
    """Load a single episode from an HDF5 file.
    
    Handles both old format (raw numpy arrays with ego_cam) and new format
    (JPEG-compressed with cam_head from h1_training_client).
    
    Args:
        hdf5_path: Path to the HDF5 file
        read_advantage: Whether to read the advantage label from metadata
        
    Returns:
        Dictionary containing episode data with keys:
        - actions: (num_steps, 14) array
        - qpos: (num_steps, 14) array
        - ego_cam: (num_steps, height, width, 3) array
        - cam_left_wrist: (num_steps, height, width, 3) array or None
        - cam_right_wrist: (num_steps, height, width, 3) array or None
        - advantage: bool or None (if read_advantage is True)
    """
    with h5py.File(hdf5_path, "r") as f:
        # Extract data from HDF5
        actions = f["action"][:]  # Shape: (num_steps, 14 or 26)
        qpos = f["observations"]["qpos"][:]  # Shape: (num_steps, 14 or 26)
        
        images_group = f["observations"]["images"]
        
        # Determine which camera names are present (cam_head or ego_cam)
        if "cam_head" in images_group:
            # New format from h1_training_client (JPEG compressed)
            head_cam_data = images_group["cam_head"][:]
            # Check if data is JPEG compressed (variable-length uint8) or raw
            if head_cam_data.dtype == object or len(head_cam_data.shape) == 1:
                ego_cam = decompress_jpeg_images(head_cam_data)
            else:
                ego_cam = head_cam_data
        elif "ego_cam" in images_group:
            # Old format (raw numpy arrays)
            ego_cam = images_group["ego_cam"][:]
        else:
            raise KeyError(f"No head camera found in {hdf5_path}. Expected 'cam_head' or 'ego_cam'")
        
        # Try to load wrist cameras, use None if not present
        cam_left_wrist = None
        cam_right_wrist = None
        
        if "cam_left_wrist" in images_group:
            left_data = images_group["cam_left_wrist"][:]
            if left_data.dtype == object or len(left_data.shape) == 1:
                cam_left_wrist = decompress_jpeg_images(left_data)
            else:
                cam_left_wrist = left_data
        
        if "cam_right_wrist" in images_group:
            right_data = images_group["cam_right_wrist"][:]
            if right_data.dtype == object or len(right_data.shape) == 1:
                cam_right_wrist = decompress_jpeg_images(right_data)
            else:
                cam_right_wrist = right_data
        
        # Use only first 14 dimensions for state and actions (as per h1_policy.py)
        if actions.shape[1] > 14:
            actions = actions[:, :14]
        if qpos.shape[1] > 14:
            qpos = qpos[:, :14]
        
        # Read advantage label from metadata if requested
        advantage = None
        if read_advantage:
            if "advantage" in f.attrs:
                advantage = bool(f.attrs["advantage"])
            else:
                # Default to False if no advantage label (older data or unlabeled)
                print(f"  Warning: No advantage label in {hdf5_path}, defaulting to False")
                advantage = False
        
        return {
            "actions": actions,
            "qpos": qpos,
            "ego_cam": ego_cam,
            "cam_left_wrist": cam_left_wrist,
            "cam_right_wrist": cam_right_wrist,
            "advantage": advantage,
        }


def main(
    data_dir: str,
    task_description: str,
    repo_id: str = "your_hf_username/h1_circular",
    *,
    num_repeats: int = 256,
    push_to_hub: bool = False,
    save_dir: str = None,
    labeling_mode: LabelingMode = "none",
):
    """Convert H1 HDF5 data to LeRobot format.
    
    Args:
        data_dir: Path to the HDF5 file or directory containing HDF5 files
        repo_id: Repository ID for the output dataset
        num_repeats: Number of times to repeat the episode sequence (to create more training data)
        push_to_hub: Whether to push the dataset to Hugging Face Hub
        save_dir: Name of the directory to save the dataset
        labeling_mode: How to handle advantage labeling:
            - "none": No advantage labeling (task description only)
            - "human_labeling": Read advantage from HDF5 metadata
            - "reward_labeling": Use reward function (not implemented)
    """
    # Validate labeling mode
    if labeling_mode == "reward_labeling":
        raise NotImplementedError("reward_labeling mode is not yet implemented")
    
    use_advantage = labeling_mode == "human_labeling"
    if use_advantage:
        print(f"Using advantage labeling mode: {labeling_mode}")
        print("  Prompts will be formatted as: '{task_description}, Advantage=True/False'")
    # Determine if data_dir is a file or directory
    data_path = Path(data_dir)
    
    if data_path.is_file():
        # Single HDF5 file
        hdf5_files = [data_path]
    elif data_path.is_dir():
        # Directory containing multiple HDF5 files
        hdf5_files = sorted(data_path.glob("*.hdf5"))
        if not hdf5_files:
            raise ValueError(f"No HDF5 files found in directory: {data_dir}")
    else:
        raise ValueError(f"Invalid path: {data_dir}. Must be a file or directory.")
    
    print(f"Found {len(hdf5_files)} HDF5 file(s):")
    for hdf5_file in hdf5_files:
        print(f"  - {hdf5_file}")
    
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / repo_id
    if save_dir is not None:
        repo_id = save_dir
        print(f"Using directory name: {save_dir}")
        # Get the directory where this file is located
        current_dir = Path(__file__).parent.resolve()
        output_path = Path(current_dir) / 'h1_data_lerobot' / save_dir
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # Based on h1_policy.py, we use 14-dim state and 14-dim actions
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        robot_type="h1",
        fps=50,  # Adjust based on your data collection frequency
        features={
            "ego_cam": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "cam_left_wrist": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "cam_right_wrist": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "qpos": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["qpos"],
            },
            "action": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["action"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Load all episodes from HDF5 files
    print("\nLoading episodes from HDF5 files...")
    episodes_data = []
    advantage_stats = {"true": 0, "false": 0}
    
    for hdf5_file in hdf5_files:
        print(f"Loading {hdf5_file.name}...")
        episode_data = load_episode_from_hdf5(str(hdf5_file), read_advantage=use_advantage)
        
        # Track advantage statistics
        if use_advantage:
            if episode_data["advantage"]:
                advantage_stats["true"] += 1
            else:
                advantage_stats["false"] += 1
        episodes_data.append(episode_data)
        
        print(f"  Data shapes:")
        print(f"    actions: {episode_data['actions'].shape}")
        print(f"    qpos: {episode_data['qpos'].shape}")
        print(f"    ego_cam: {episode_data['ego_cam'].shape}")
        print(f"    cam_left_wrist: {episode_data['cam_left_wrist'].shape if episode_data['cam_left_wrist'] is not None else 'None (will use zero padding)'}")
        print(f"    cam_right_wrist: {episode_data['cam_right_wrist'].shape if episode_data['cam_right_wrist'] is not None else 'None (will use zero padding)'}")
        if use_advantage:
            advantage_str = "True (good)" if episode_data["advantage"] else "False (needs improvement)"
            print(f"    advantage: {advantage_str}")
    
    if use_advantage:
        print(f"\nAdvantage label statistics:")
        print(f"  Good episodes (Advantage=True): {advantage_stats['true']}")
        print(f"  Bad episodes (Advantage=False): {advantage_stats['false']}")

    # Create zero-padded images for missing cameras
    zero_image = np.zeros((224, 224, 3), dtype=np.uint8)

    # Process episodes in the order: [1,2,...,N, 1,2,...,N, ...] (repeated num_repeats times)
    total_episodes = len(episodes_data) * num_repeats
    print(f"\nProcessing {total_episodes} episodes ({len(episodes_data)} files × {num_repeats} repeats)...")
    
    episode_counter = 0
    for repeat_idx in range(num_repeats):
        for file_idx, episode_data in enumerate(episodes_data):
            episode_counter += 1
            print(f"Processing episode {episode_counter}/{total_episodes} (file {file_idx + 1}, repeat {repeat_idx + 1})")
            
            actions = episode_data["actions"]
            qpos = episode_data["qpos"]
            ego_cam = episode_data["ego_cam"]
            cam_left_wrist = episode_data["cam_left_wrist"]
            cam_right_wrist = episode_data["cam_right_wrist"]
            
            # Format task description with advantage label if using advantage labeling
            if use_advantage:
                advantage = episode_data["advantage"]
                advantage_str = "True" if advantage else "False"
                episode_task = f"{task_description}, Advantage={advantage_str}"
            else:
                episode_task = task_description
            
            # Iterate through each timestep in the episode
            for step_idx in range(len(actions)):
                # Resize ego camera
                ego_cam_resized = resize_image(ego_cam[step_idx])
                
                # Resize or use zero padding for wrist cameras
                if cam_left_wrist is not None:
                    cam_left_wrist_resized = resize_image(cam_left_wrist[step_idx])
                else:
                    cam_left_wrist_resized = zero_image.copy()
                
                if cam_right_wrist is not None:
                    cam_right_wrist_resized = resize_image(cam_right_wrist[step_idx])
                else:
                    cam_right_wrist_resized = zero_image.copy()

                # Add frame to dataset (all frames in episode get same task/advantage)
                dataset.add_frame(
                    {
                        "ego_cam": ego_cam_resized,
                        "cam_left_wrist": cam_left_wrist_resized,
                        "cam_right_wrist": cam_right_wrist_resized,
                        "qpos": qpos[step_idx].astype(np.float32),
                        "action": actions[step_idx].astype(np.float32),
                        "task": episode_task,
                    }
                )
            
            # Save the episode
            dataset.save_episode()
            advantage_info = f" (Advantage={'True' if episode_data.get('advantage') else 'False'})" if use_advantage else ""
            print(f"  Saved episode {episode_counter} with {len(actions)} frames{advantage_info}")

    print(f"\nDataset created successfully!")
    print(f"Total episodes: {total_episodes}")
    print(f"Total frames: {episode_counter * len(actions)}")
    print(f"Dataset saved to: {output_path}")

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        print("\nPushing dataset to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["h1", "humanoid", "robot"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print("Dataset pushed successfully!")


if __name__ == "__main__":
    tyro.cli(main)

