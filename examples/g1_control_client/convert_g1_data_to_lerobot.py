"""
Script for converting G1 HDF5 dataset to LeRobot format.

Supports advantage labeling for training with advantage-augmented prompts:
  - human_labeling: Read advantage label from HDF5 metadata (set during data collection)
  - reward_labeling: Use a reward function to determine advantage (not implemented)
  - none: No advantage labeling (original behavior)

The prompt format with advantage labeling is:
  "task_description, Advantage=True" or "task_description, Advantage=False"

Usage:
uv run examples/g1_control_client/convert_g1_data_to_lerobot.py --data_dir training_data/g1/cabinet_bottle/episode_03.hdf5 --task_description "pick up the bottle" --num_repeats 10

For a directory with multiple HDF5 files:
uv run examples/g1_control_client/convert_g1_data_to_lerobot.py --data_dir examples/g1_control_client/g1_data_processed/cabinet_bottle/ --task_description "pick up the bottle" --num_repeats 4

With advantage labeling:
uv run examples/g1_control_client/convert_g1_data_to_lerobot.py --data_dir ./data/ --task_description "pick up the bottle" --labeling_mode human_labeling

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
    
    G1 HDF5 format:
    - action: [T, 28] - arm + hand joint targets
    - loco_action: [T, 20] - joystick/button inputs
    - observations/qpos: [T, 28] - arm + hand joint positions
    - observations/loco_state: [T, 17] - locomotion state (rpy at [1:4], gyro at [11:14])
    - observations/images/ego_cam: [T, H, W, 3] - RGB images
    
    Args:
        hdf5_path: Path to the HDF5 file
        read_advantage: Whether to read the advantage label from metadata
        
    Returns:
        Dictionary containing episode data with keys:
        - actions: (num_steps, 28) array - upper body actions
        - loco_action: (num_steps, 20) array - locomotion actions
        - qpos: (num_steps, 28) array - joint positions
        - loco_state: (num_steps, 17) array - locomotion state
        - ego_cam: (num_steps, height, width, 3) array
        - advantage: bool or None (if read_advantage is True)
    """
    with h5py.File(hdf5_path, "r") as f:
        # Extract data from HDF5
        actions = f["action"][:]  # Shape: (num_steps, 28)
        loco_action = f["loco_action"][:]  # Shape: (num_steps, 20)
        qpos = f["observations"]["qpos"][:]  # Shape: (num_steps, 28)
        loco_state = f["observations"]["loco_state"][:]  # Shape: (num_steps, 17)
        
        images_group = f["observations"]["images"]
        
        # Load ego camera (G1 only has head camera)
        if "ego_cam" in images_group:
            ego_cam_data = images_group["ego_cam"][:]
            # Check if data is JPEG compressed (variable-length uint8) or raw
            if ego_cam_data.dtype == object or len(ego_cam_data.shape) == 1:
                ego_cam = decompress_jpeg_images(ego_cam_data)
            else:
                ego_cam = ego_cam_data
        elif "cam_head" in images_group:
            # Alternative name for head camera
            head_cam_data = images_group["cam_head"][:]
            if head_cam_data.dtype == object or len(head_cam_data.shape) == 1:
                ego_cam = decompress_jpeg_images(head_cam_data)
            else:
                ego_cam = head_cam_data
        else:
            raise KeyError(f"No head camera found in {hdf5_path}. Expected 'ego_cam' or 'cam_head'")
        
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
            "loco_action": loco_action,
            "qpos": qpos,
            "loco_state": loco_state,
            "ego_cam": ego_cam,
            "advantage": advantage,
        }


def main(
    data_dir: str,
    task_description: str,
    repo_id: str = "your_hf_username/g1_cabinet_bottle",
    *,
    num_repeats: int = 256,
    push_to_hub: bool = False,
    save_dir: str = None,
    labeling_mode: LabelingMode = "none",
    reward_task_instruction: str = None,
    reward_max_frames: int = 30,
    reward_image_rotation: int = 0,
    reward_advantage_threshold: float = 0.3,
    reward_ranking_frames: int = 5,
    filter_good_only: bool = False,
):
    """Convert G1 HDF5 data to LeRobot format.
    
    Args:
        data_dir: Path to the HDF5 file or directory containing HDF5 files
        task_description: Task description for the dataset
        repo_id: Repository ID for the output dataset
        num_repeats: Number of times to repeat the episode sequence (to create more training data)
        push_to_hub: Whether to push the dataset to Hugging Face Hub
        save_dir: Name of the directory to save the dataset
        labeling_mode: How to handle advantage labeling:
            - "none": No advantage labeling (task description only)
            - "human_labeling": Read advantage from HDF5 metadata
            - "reward_labeling": Use embodied reward model to label advantage
        reward_task_instruction: Detailed task instruction for reward model (required for reward_labeling)
        reward_max_frames: Maximum frames to sample for reward labeling
        reward_image_rotation: Image rotation angle for reward labeling (0, 90, 180, 270)
        reward_advantage_threshold: Percentile threshold for advantage labeling (0.0-1.0)
                                   e.g., 0.3 means top 30% episodes get Advantage=True
        reward_ranking_frames: Number of frames from the end to use for ranking (default: 5, use 0 for all frames)
        filter_good_only: If True, only keep episodes with Advantage=True (for epoch 0 training from warmup checkpoint)
    """
    # Validate labeling mode
    if labeling_mode == "reward_labeling":
        if not reward_task_instruction:
            print("Warning: reward_task_instruction not provided, using task_description")
            reward_task_instruction = task_description
    
    use_advantage = labeling_mode in ["human_labeling", "reward_labeling"]
    if use_advantage:
        print(f"Using advantage labeling mode: {labeling_mode}")
        print("  Prompts will be formatted as: '{task_description}, Advantage=True/False'")
    
    # If filter_good_only is enabled, force reading advantage labels
    if filter_good_only:
        if not use_advantage:
            print("Warning: filter_good_only requires advantage labels, enabling human_labeling mode")
            labeling_mode = "human_labeling"
            use_advantage = True
        print(f"Filtering: Only episodes with Advantage=True will be included in the dataset")
        
    # Run reward labeling if needed
    reward_labels = {}
    if labeling_mode == "reward_labeling":
        print("\nRunning reward labeling...")
        print(f"  Task instruction: {reward_task_instruction}")
        print(f"  Max frames: {reward_max_frames}")
        print(f"  Image rotation: {reward_image_rotation}")
        print(f"  Advantage threshold: {reward_advantage_threshold:.1%} (top {reward_advantage_threshold:.1%} episodes)")
        print(f"  Ranking frames: {reward_ranking_frames if reward_ranking_frames > 0 else 'all'}")
        
        # Import Qwen-based reward labeling
        try:
            from qwen_reward_labeling import label_episodes
        except ImportError:
            print("Error: Could not import qwen_reward_labeling module")
            print("Make sure qwen_reward_labeling.py is in the same directory")
            raise
        
        # Determine the directory to label
        data_path = Path(data_dir)
        if data_path.is_file():
            label_dir = data_path.parent
        else:
            label_dir = data_path
        
        # Get checkpoint path from environment variable
        checkpoint_path = os.environ.get("QWEN_REWARD_CHECKPOINT_PATH")
        if not checkpoint_path:
            print("Error: QWEN_REWARD_CHECKPOINT_PATH environment variable not set!")
            print("Set it with: export QWEN_REWARD_CHECKPOINT_PATH='/path/to/checkpoint'")
            raise ValueError("QWEN_REWARD_CHECKPOINT_PATH not set")
        
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint path does not exist: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"  Checkpoint path: {checkpoint_path}")
        
        # Run labeling
        reward_labels = label_episodes(
            data_dir=str(label_dir),
            task_instruction=reward_task_instruction,
            checkpoint_path=checkpoint_path,
            max_frames=reward_max_frames,
            image_rotation=reward_image_rotation,
            advantage_threshold=reward_advantage_threshold,
            inference_batch_size=30,
            ranking_frames=reward_ranking_frames,
        )
        
        print(f"\nReward labeling complete. Labeled {len(reward_labels)} episodes.")
    
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
    
    # Auto-detect dimensions from first HDF5 file
    first_file = hdf5_files[0]
    with h5py.File(first_file, "r") as f:
        action_dim = int(f["action"].shape[1])  # Should be 28
        state_dim = int(f["observations"]["qpos"].shape[1])  # Should be 28
        loco_state_dim = int(f["observations"]["loco_state"].shape[1])  # Should be 17
        loco_action_dim = int(f["loco_action"].shape[1])  # Should be 20
        fps = int(f.attrs.get("fps", 30))  # Default to 30 if not specified
    
    print(f"\nData dimensions:")
    print(f"  Action (upper body): {action_dim} (arm + hand joints)")
    print(f"  State (qpos): {state_dim} (arm + hand joints)")
    print(f"  Loco state: {loco_state_dim}")
    print(f"  Loco action: {loco_action_dim}")
    print(f"  FPS: {fps}")
    
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / repo_id
    if save_dir is not None:
        repo_id = save_dir
        print(f"Using directory name: {save_dir}")
        # Get the directory where this file is located
        current_dir = Path(__file__).parent.resolve()
        output_path = Path(current_dir) / 'g1_data_lerobot' / save_dir
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # Match H1 format exactly: ego_cam, cam_left_wrist, cam_right_wrist, qpos, action
    # Also include loco_state and loco_action for G1 transforms (processed during G1Inputs/G1Outputs)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        robot_type="g1",
        fps=fps,
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
                "shape": (state_dim,),
                "names": ["qpos"],
            },
            "loco_state": {
                "dtype": "float32",
                "shape": (loco_state_dim,),
                "names": ["loco_state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["action"],
            },
            "loco_action": {
                "dtype": "float32",
                "shape": (loco_action_dim,),
                "names": ["loco_action"],
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
        
        # For reward labeling, override advantage with reward label
        if labeling_mode == "reward_labeling":
            episode_data = load_episode_from_hdf5(str(hdf5_file), read_advantage=False)
            
            # Get advantage from reward labels
            if hdf5_file.name in reward_labels:
                episode_data["advantage"] = reward_labels[hdf5_file.name].get("advantage", False)
                print(f"  Reward label: Advantage={episode_data['advantage']} "
                      f"(score={reward_labels[hdf5_file.name].get('corrected_total_reward', 0):.1f})")
            else:
                print(f"  Warning: No reward label found, defaulting to False")
                episode_data["advantage"] = False
        else:
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
        print(f"    loco_action: {episode_data['loco_action'].shape}")
        print(f"    qpos: {episode_data['qpos'].shape}")
        print(f"    loco_state: {episode_data['loco_state'].shape}")
        print(f"    ego_cam: {episode_data['ego_cam'].shape}")
        if use_advantage:
            advantage_str = "True (good)" if episode_data["advantage"] else "False (needs improvement)"
            print(f"    advantage: {advantage_str}")
    
    if use_advantage:
        print(f"\nAdvantage label statistics:")
        print(f"  Good episodes (Advantage=True): {advantage_stats['true']}")
        print(f"  Bad episodes (Advantage=False): {advantage_stats['false']}")
    
    # Filter to keep only good episodes if requested
    if filter_good_only:
        original_count = len(episodes_data)
        episodes_data = [ep for ep in episodes_data if ep.get("advantage", False)]
        filtered_count = len(episodes_data)
        removed_count = original_count - filtered_count
        
        print(f"\nFiltering episodes (filter_good_only=True):")
        print(f"  Original episodes: {original_count}")
        print(f"  Kept (Advantage=True): {filtered_count}")
        print(f"  Removed (Advantage=False): {removed_count}")
        
        if filtered_count == 0:
            raise ValueError("No episodes with Advantage=True found after filtering! Cannot create dataset.")

    # Create zero-padded images for missing cameras (G1 doesn't have wrist cameras)
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
            loco_action = episode_data["loco_action"]
            qpos = episode_data["qpos"]
            loco_state = episode_data["loco_state"]
            ego_cam = episode_data["ego_cam"]
            
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
                
                # G1 doesn't have wrist cameras - use zero padding to match H1 format
                cam_left_wrist_resized = zero_image.copy()
                cam_right_wrist_resized = zero_image.copy()

                # Add frame to dataset (all frames in episode get same task/advantage)
                dataset.add_frame(
                    {
                        "ego_cam": ego_cam_resized,
                        "cam_left_wrist": cam_left_wrist_resized,
                        "cam_right_wrist": cam_right_wrist_resized,
                        "qpos": qpos[step_idx].astype(np.float32),
                        "loco_state": loco_state[step_idx].astype(np.float32),
                        "action": actions[step_idx].astype(np.float32),
                        "loco_action": loco_action[step_idx].astype(np.float32),
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
            tags=["g1", "humanoid", "robot"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print("Dataset pushed successfully!")
    
    # Generate alignment visualization if reward labeling was used and human labels are available
    if labeling_mode == "reward_labeling" and 'reward_labels' in locals():
        try:
            print(f"\n{'='*80}")
            print("GENERATING REWARD ALIGNMENT VISUALIZATION")
            print(f"{'='*80}")
            
            from reward_alignment_viz import generate_alignment_visualization
            
            # Output path for visualization (in the dataset directory)
            viz_output_path = output_path / "reward_alignment.png"
            print(f"Output path: {viz_output_path}")
            print(f"Data dir: {label_dir}")
            print(f"Number of reward labels: {len(reward_labels)}")
            
            generate_alignment_visualization(
                data_dir=str(label_dir),
                reward_labels=reward_labels,
                output_path=str(viz_output_path),
                checkpoint_path=checkpoint_path,
                task_instruction=reward_task_instruction,
            )
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"\n{'!'*80}")
            print(f"ERROR: Could not generate alignment visualization")
            print(f"{'!'*80}")
            print(f"Error: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print(f"{'!'*80}\n")


if __name__ == "__main__":
    tyro.cli(main)

