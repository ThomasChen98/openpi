"""
Script for converting H1 HDF5 dataset to LeRobot format.

Usage:
uv run training_data/h1/convert_h1_data_to_lerobot.py --data_dir training_data/h1/circular.hdf5 --repo_id your_hf_username/h1_circular --num_repeats 10

If you want to push your dataset to the Hugging Face Hub:
uv run training_data/h1/convert_h1_data_to_lerobot.py --data_dir training_data/h1/circular.hdf5 --repo_id your_hf_username/h1_circular --num_repeats 10 --push_to_hub

Note: Install h5py if needed: `uv pip install h5py`
Note: Install opencv-python for image resizing: `uv pip install opencv-python`

The resulting dataset will be saved to the $HF_LEROBOT_HOME directory.
"""

import shutil

import cv2
import h5py
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


def resize_image(image: np.ndarray, target_height: int = 224, target_width: int = 224) -> np.ndarray:
    """Resize image to target dimensions."""
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def main(
    data_dir: str,
    repo_id: str = "your_hf_username/h1_circular",
    *,
    num_repeats: int = 256,
    push_to_hub: bool = False,
):
    """Convert H1 HDF5 data to LeRobot format.
    
    Args:
        data_dir: Path to the HDF5 file containing the H1 data
        repo_id: Repository ID for the output dataset
        num_repeats: Number of times to repeat the single episode (to create more training data)
        push_to_hub: Whether to push the dataset to Hugging Face Hub
    """
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

    # Load the HDF5 file and convert to LeRobot format
    print(f"Loading data from {data_dir}")
    with h5py.File(data_dir, "r") as f:
        # Extract data from HDF5
        actions = f["action"][:]  # Shape: (600, 26)
        qpos = f["observations"]["qpos"][:]  # Shape: (600, 26)
        ego_cam = f["observations"]["images"]["ego_cam"][:]  # Shape: (600, 480, 640, 3)
        cam_left_wrist = f["observations"]["images"]["cam_left_wrist"][:]  # Shape: (600, 480, 640, 3)
        cam_right_wrist = f["observations"]["images"]["cam_right_wrist"][:]  # Shape: (600, 480, 640, 3)

        # Use only first 14 dimensions for state and actions (as per h1_policy.py)
        actions = actions[:, :14]
        qpos = qpos[:, :14]

        print(f"Data shapes:")
        print(f"  actions: {actions.shape}")
        print(f"  qpos: {qpos.shape}")
        print(f"  ego_cam: {ego_cam.shape}")
        print(f"  cam_left_wrist: {cam_left_wrist.shape}")
        print(f"  cam_right_wrist: {cam_right_wrist.shape}")

        # Repeat the episode num_repeats times
        print(f"\nRepeating episode {num_repeats} times...")
        for episode_idx in range(num_repeats):
            print(f"Processing episode {episode_idx + 1}/{num_repeats}")
            
            # Iterate through each timestep in the episode
            for step_idx in range(len(actions)):
                # Resize images to 224x224
                ego_cam_resized = resize_image(ego_cam[step_idx])
                cam_left_wrist_resized = resize_image(cam_left_wrist[step_idx])
                cam_right_wrist_resized = resize_image(cam_right_wrist[step_idx])

                # Add frame to dataset
                dataset.add_frame(
                    {
                        "ego_cam": ego_cam_resized,
                        "cam_left_wrist": cam_left_wrist_resized,
                        "cam_right_wrist": cam_right_wrist_resized,
                        "qpos": qpos[step_idx].astype(np.float32),
                        "action": actions[step_idx].astype(np.float32),
                        "task": "move arm circularly",
                    }
                )
            
            # Save the episode
            dataset.save_episode()
            print(f"  Saved episode {episode_idx + 1} with {len(actions)} frames")

    print(f"\nDataset created successfully with {num_repeats} episodes!")
    print(f"Total frames: {num_repeats * len(actions)}")
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

