"""
Script for converting H1 HDF5 dataset to LeRobot format.

Usage:
uv run training_data/h1/convert_h1_data_to_lerobot.py --data_dir training_data/h1/circular.hdf5 --repo_id your_hf_username/h1_circular --num_repeats 10

For a directory with multiple HDF5 files:
uv run examples/h1_control_client/convert_h1_data_to_lerobot.py --data_dir examples/h1_control_client/h1_data_processed/box_action/good/ --repo_id ThomasChen98/h1_box_action_with_lang --num_repeats 4

If you want to push your dataset to the Hugging Face Hub:
uv run training_data/h1/convert_h1_data_to_lerobot.py --data_dir training_data/h1/circular.hdf5 --repo_id your_hf_username/h1_circular --num_repeats 10 --push_to_hub

Note: Install h5py if needed: `uv pip install h5py`
Note: Install opencv-python for image resizing: `uv pip install opencv-python`

The resulting dataset will be saved to the $HF_LEROBOT_HOME directory.
"""

import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


def resize_image(image: np.ndarray, target_height: int = 224, target_width: int = 224) -> np.ndarray:
    """Resize image to target dimensions."""
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def load_episode_from_hdf5(hdf5_path: str) -> dict:
    """Load a single episode from an HDF5 file.
    
    Args:
        hdf5_path: Path to the HDF5 file
        
    Returns:
        Dictionary containing episode data with keys:
        - actions: (num_steps, 14) array
        - qpos: (num_steps, 14) array
        - ego_cam: (num_steps, height, width, 3) array
        - cam_left_wrist: (num_steps, height, width, 3) array or None
        - cam_right_wrist: (num_steps, height, width, 3) array or None
    """
    with h5py.File(hdf5_path, "r") as f:
        # Extract data from HDF5
        actions = f["action"][:]  # Shape: (num_steps, 26)
        qpos = f["observations"]["qpos"][:]  # Shape: (num_steps, 26)
        
        # Load ego camera (required)
        ego_cam = f["observations"]["images"]["ego_cam"][:]  # Shape: (num_steps, height, width, 3)
        
        # Try to load wrist cameras, use None if not present
        cam_left_wrist = None
        cam_right_wrist = None
        
        if "cam_left_wrist" in f["observations"]["images"]:
            cam_left_wrist = f["observations"]["images"]["cam_left_wrist"][:]
        
        if "cam_right_wrist" in f["observations"]["images"]:
            cam_right_wrist = f["observations"]["images"]["cam_right_wrist"][:]
        
        # Use only first 14 dimensions for state and actions (as per h1_policy.py)
        actions = actions[:, :14]
        qpos = qpos[:, :14]
        
        return {
            "actions": actions,
            "qpos": qpos,
            "ego_cam": ego_cam,
            "cam_left_wrist": cam_left_wrist,
            "cam_right_wrist": cam_right_wrist,
        }


def main(
    data_dir: str,
    repo_id: str = "your_hf_username/h1_circular",
    *,
    num_repeats: int = 256,
    push_to_hub: bool = False,
):
    """Convert H1 HDF5 data to LeRobot format.
    
    Args:
        data_dir: Path to the HDF5 file or directory containing HDF5 files
        repo_id: Repository ID for the output dataset
        num_repeats: Number of times to repeat the episode sequence (to create more training data)
        push_to_hub: Whether to push the dataset to Hugging Face Hub
    """
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
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # Based on h1_policy.py, we use 14-dim state and 14-dim actions
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="h1",
        fps=30,  # Adjust based on your data collection frequency
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
    for hdf5_file in hdf5_files:
        print(f"Loading {hdf5_file.name}...")
        episode_data = load_episode_from_hdf5(str(hdf5_file))
        episodes_data.append(episode_data)
        
        print(f"  Data shapes:")
        print(f"    actions: {episode_data['actions'].shape}")
        print(f"    qpos: {episode_data['qpos'].shape}")
        print(f"    ego_cam: {episode_data['ego_cam'].shape}")
        print(f"    cam_left_wrist: {episode_data['cam_left_wrist'].shape if episode_data['cam_left_wrist'] is not None else 'None (will use zero padding)'}")
        print(f"    cam_right_wrist: {episode_data['cam_right_wrist'].shape if episode_data['cam_right_wrist'] is not None else 'None (will use zero padding)'}")

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

                # Add frame to dataset
                dataset.add_frame(
                    {
                        "ego_cam": ego_cam_resized,
                        "cam_left_wrist": cam_left_wrist_resized,
                        "cam_right_wrist": cam_right_wrist_resized,
                        "qpos": qpos[step_idx].astype(np.float32),
                        "action": actions[step_idx].astype(np.float32),
                        "task": "pour the water into the cup",
                    }
                )
            
            # Save the episode
            dataset.save_episode()
            print(f"  Saved episode {episode_counter} with {len(actions)} frames")

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

