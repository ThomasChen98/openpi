"""
Direct converter from Humanoid Everyday dataset to LeRobot format.

This script directly converts Humanoid Everyday dataset to LeRobot format
without the intermediate HDF5 step, optimized for training.

Dataset source: https://humanoideveryday.github.io/
Dataset repo: https://github.com/ausbxuse/Humanoid-Everyday

Usage:
    # Install the humanoid_everyday dataloader first:
    # pip install humanoid-everyday  # or pip install -e /path/to/Humanoid-Everyday

    # Convert a single task
    python convert_humanoid_everyday_to_lerobot.py \
        --data_path ~/Downloads/push_a_button.zip \
        --repo_id username/humanoid_everyday_push_button \
        --task_name "push a button"

    # Convert and push to hub
    python convert_humanoid_everyday_to_lerobot.py \
        --data_path ~/Downloads/push_a_button.zip \
        --repo_id username/humanoid_everyday_push_button \
        --task_name "push a button" \
        --push_to_hub

    # Convert with fewer repeats (default is 10 for data augmentation)
    python convert_humanoid_everyday_to_lerobot.py \
        --data_path ~/Downloads/push_a_button.zip \
        --repo_id username/task \
        --task_name "task description" \
        --num_repeats 5
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


def resize_image(image: np.ndarray, target_height: int = 224, target_width: int = 224) -> np.ndarray:
    """Resize image to target dimensions."""
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def main(
    data_path: str,
    repo_id: str,
    task_name: str,
    *,
    num_repeats: int = 10,
    max_episodes: int | None = None,
    push_to_hub: bool = False,
) -> None:
    """Convert Humanoid Everyday dataset to LeRobot format.
    
    Args:
        data_path: Path to the downloaded .zip file or extracted directory
        repo_id: Repository ID for the output dataset (e.g., 'username/task_name')
        task_name: Human-readable task description (e.g., "push a button")
        num_repeats: Number of times to repeat episodes for data augmentation
        max_episodes: Maximum number of episodes to include (None = all)
        push_to_hub: Whether to push the dataset to Hugging Face Hub
    """
    try:
        from humanoid_everyday import Dataloader
    except ImportError:
        print("❌ Error: humanoid_everyday package not found!")
        print("\nPlease install it:")
        print("  git clone https://github.com/ausbxuse/Humanoid-Everyday")
        print("  cd Humanoid-Everyday")
        print("  pip install -e .")
        return
    
    # Clean up any existing dataset
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)
    
    # Load Humanoid Everyday dataset
    print(f"\nLoading Humanoid Everyday dataset from: {data_path}")
    ds = Dataloader(data_path)
    print(f"✓ Loaded dataset with {len(ds)} episodes")
    
    # Limit episodes if specified
    num_episodes = len(ds) if max_episodes is None else min(max_episodes, len(ds))
    print(f"Converting {num_episodes} episodes (repeated {num_repeats} times = {num_episodes * num_repeats} total)")
    
    # Create LeRobot dataset
    print("\nCreating LeRobot dataset...")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="h1",  # Can be "h1" or "g1" depending on robot_type field
        fps=30,
        features={
            "ego_cam": {
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
    
    # Convert episodes
    print("\nConverting episodes...")
    episode_counter = 0
    total_frames = 0
    
    for repeat_idx in range(num_repeats):
        for ep_idx in range(num_episodes):
            episode_counter += 1
            print(f"Processing episode {episode_counter}/{num_episodes * num_repeats} "
                  f"(source ep {ep_idx}, repeat {repeat_idx + 1}/{num_repeats})")
            
            episode = ds[ep_idx]
            num_frames = len(episode)
            
            for step_idx in range(num_frames):
                step = episode[step_idx]
                
                # Extract arm state (14 DoF)
                arm_state = np.array(step["states"]["arm_state"], dtype=np.float32)  # (14,)
                
                # Extract actions (left 7 + right 7 = 14)
                left_angles = np.array(step["actions"]["left_angles"], dtype=np.float32)  # (7,)
                right_angles = np.array(step["actions"]["right_angles"], dtype=np.float32)  # (7,)
                arm_action = np.concatenate([left_angles, right_angles])  # (14,)
                
                # Get and resize image
                ego_cam = step["image"]  # (480, 640, 3) uint8
                ego_cam_resized = resize_image(ego_cam)  # (224, 224, 3)
                
                # Add frame to dataset
                dataset.add_frame({
                    "ego_cam": ego_cam_resized,
                    "qpos": arm_state,
                    "action": arm_action,
                    "task": task_name,
                })
            
            # Save episode
            dataset.save_episode()
            total_frames += num_frames
            print(f"  ✓ Saved episode {episode_counter} with {num_frames} frames")
    
    print(f"\n{'='*80}")
    print("✓ Dataset created successfully!")
    print(f"{'='*80}")
    print(f"Total episodes: {episode_counter}")
    print(f"Total frames: {total_frames}")
    print(f"Dataset saved to: {output_path}")
    print(f"Task: {task_name}")
    
    # Push to hub if requested
    if push_to_hub:
        print("\nPushing dataset to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["h1", "g1", "humanoid", "humanoid_everyday"],
            private=False,
            push_videos=True,
            license="mit",
        )
        print("✓ Dataset pushed successfully!")
        print(f"View at: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    tyro.cli(main)




