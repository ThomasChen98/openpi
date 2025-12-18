"""
Direct converter from Humanoid Everyday dataset to LeRobot format.

This script directly converts Humanoid Everyday dataset to LeRobot format
without the intermediate HDF5 step, optimized for training.

Dataset source: https://humanoideveryday.github.io/
Dataset repo: https://github.com/ausbxuse/Humanoid-Everyday

Output format (26 DOF):
  - qpos: 26 DOF (14 arm + 12 hand)
  - action: 26 DOF (14 arm + 12 hand)
  - Format: [left_arm(7), right_arm(7), left_hand(6), right_hand(6)]

Usage:
    # Install the humanoid_everyday dataloader first:
    # pip install humanoid-everyday  # or pip install -e /path/to/Humanoid-Everyday

    # Convert a single task (26 DOF with hands)
    python convert_humanoid_everyday_to_lerobot.py \
        --data_path ~/Downloads/place_kettle.zip \
        --repo_id h1_place_kettle \
        --task_name "place the kettle on its base"

    # Convert with custom output directory
    python convert_humanoid_everyday_to_lerobot.py \
        --data_path ~/Downloads/task.zip \
        --repo_id h1_task \
        --task_name "do the task" \
        --output_dir ./h1_data_lerobot

    # Convert arms only (14 DOF, no hands)
    python convert_humanoid_everyday_to_lerobot.py \
        --data_path ~/Downloads/task.zip \
        --repo_id h1_task \
        --task_name "do the task" \
        --no-include_hands

    # Convert with fewer repeats (default is 10 for data augmentation)
    python convert_humanoid_everyday_to_lerobot.py \
        --data_path ~/Downloads/task.zip \
        --repo_id h1_task \
        --task_name "task description" \
        --num_repeats 5
"""

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import tyro


def get_lerobot_home():
    """Get HF_LEROBOT_HOME, must be called after setting env var."""
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    return HF_LEROBOT_HOME


def resize_image(image: np.ndarray, target_height: int = 224, target_width: int = 224) -> np.ndarray:
    """Resize image to target dimensions."""
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def main(
    data_path: str,
    repo_id: str,
    task_name: str,
    *,
    output_dir: str | None = None,
    include_hands: bool = True,
    num_repeats: int = 10,
    max_episodes: int | None = None,
    push_to_hub: bool = False,
) -> None:
    """Convert Humanoid Everyday dataset to LeRobot format.
    
    Args:
        data_path: Path to the downloaded .zip file or extracted directory
        repo_id: Repository ID for the output dataset (e.g., 'h1_place_kettle')
        task_name: Human-readable task description (e.g., "place the kettle on its base")
        output_dir: Custom output directory (default: ~/.cache/huggingface/lerobot/)
        include_hands: Include hand data for 26 DOF output (default: True). If False, outputs 14 DOF arms only.
        num_repeats: Number of times to repeat episodes for data augmentation
        max_episodes: Maximum number of episodes to include (None = all)
        push_to_hub: Whether to push the dataset to Hugging Face Hub
    """
    try:
        from humanoid_everyday import Dataloader
    except ImportError:
        print("Error: humanoid_everyday package not found!")
        print("\nPlease install it:")
        print("  git clone https://github.com/ausbxuse/Humanoid-Everyday")
        print("  cd Humanoid-Everyday")
        print("  pip install -e .")
        return
    
    # Override HF_LEROBOT_HOME if custom output_dir specified (must be before importing lerobot)
    if output_dir:
        os.environ["HF_LEROBOT_HOME"] = str(Path(output_dir).resolve())
    
    # Import lerobot after setting env var
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    
    # Get output path
    output_path = get_lerobot_home() / repo_id
    
    # Clean up any existing dataset
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)
    
    # Load Humanoid Everyday dataset
    print(f"\nLoading Humanoid Everyday dataset from: {data_path}")
    ds = Dataloader(data_path)
    print(f"Loaded dataset with {len(ds)} episodes")
    
    # Auto-detect data dimensions from first step
    first_step = ds[0][0]
    arm_state = np.array(first_step["states"]["arm_state"])
    
    print(f"\n=== Data Dimensions (auto-detected) ===")
    print(f"  arm_state: {arm_state.shape}")
    
    # Check for hand state
    hand_state = first_step["states"].get("hand_state")
    has_hand_state = hand_state is not None
    if has_hand_state:
        hand_state = np.array(hand_state)
        print(f"  hand_state: {hand_state.shape}")
    
    # Determine output dimensions
    if include_hands and has_hand_state:
        state_dim = arm_state.shape[0] + hand_state.shape[0]  # 14 + 12 = 26
        action_dim = state_dim  # Same: 26 DOF
        print(f"\n  -> Mode: Arms + Hands (26 DOF)")
        print(f"  -> Format: [left_arm(7), right_arm(7), left_hand(6), right_hand(6)]")
    else:
        state_dim = arm_state.shape[0]  # 14
        action_dim = state_dim  # Same: 14 DOF
        if include_hands and not has_hand_state:
            print(f"\n  -> Warning: --include_hands specified but dataset has no hand_state!")
        print(f"\n  -> Mode: Arms only (14 DOF)")
    
    print(f"  -> Final state_dim: {state_dim}")
    print(f"  -> Final action_dim: {action_dim}")
    
    # Limit episodes if specified
    num_episodes = len(ds) if max_episodes is None else min(max_episodes, len(ds))
    print(f"\nConverting {num_episodes} episodes (repeated {num_repeats} times = {num_episodes * num_repeats} total)")
    
    # Create LeRobot dataset
    print(f"\nCreating LeRobot dataset at: {output_path}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="h1",
        fps=30,
        features={
            "ego_cam": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "qpos": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["qpos"],
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
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
                
                # Extract arm state (14 DOF)
                arm_state = np.array(step["states"]["arm_state"], dtype=np.float32)
                
                if include_hands and has_hand_state:
                    # Extract hand state (12 DOF)
                    hand_state = np.array(step["states"]["hand_state"], dtype=np.float32)
                    
                    # 26 DOF: [arm_state(14), hand_state(12)]
                    # Use state as both qpos and action (action = next state target)
                    qpos = np.concatenate([arm_state, hand_state])
                    action = qpos.copy()  # Action targets the same state format
                else:
                    # 14 DOF: arms only
                    qpos = arm_state
                    action = arm_state.copy()
                
                # Get and resize image
                ego_cam = step["image"]
                ego_cam_resized = resize_image(ego_cam)
                
                # Add frame to dataset
                dataset.add_frame({
                    "ego_cam": ego_cam_resized,
                    "qpos": qpos,
                    "action": action,
                    "task": task_name,
                })
            
            # Save episode
            dataset.save_episode()
            total_frames += num_frames
            print(f"  Saved episode {episode_counter} with {num_frames} frames")
    
    print(f"\n{'='*80}")
    print("Dataset created successfully!")
    print(f"{'='*80}")
    print(f"Total episodes: {episode_counter}")
    print(f"Total frames: {total_frames}")
    print(f"Dataset saved to: {output_path}")
    print(f"Task: {task_name}")
    print(f"DOF: {action_dim} ({'arms + hands' if include_hands and has_hand_state else 'arms only'})")
    print(f"Format: [left_arm(7), right_arm(7)" + (", left_hand(6), right_hand(6)]" if include_hands and has_hand_state else "]"))
    
    # Push to hub if requested
    if push_to_hub:
        print("\nPushing dataset to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["h1", "humanoid", "humanoid_everyday"] + (["dexterous", "inspire_hand"] if include_hands and has_hand_state else []),
            private=False,
            push_videos=True,
            license="mit",
        )
        print("Dataset pushed successfully!")
        print(f"View at: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    tyro.cli(main)
