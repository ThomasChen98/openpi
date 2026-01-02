"""
Qwen-based Embodied Reward Labeling for H1 Robot Data

This script uses a locally trained Qwen3-VL model to automatically label
episodes with advantage labels (True/False) based on task completion quality.

Unlike the OpenAI API-based approach, this runs locally and is much faster
and more cost-effective.

Usage:
    # Test on a directory of HDF5 files
    uv run examples/h1_control_client/qwen_reward_labeling.py \
        --data_dir examples/h1_control_client/h1_data_auto/lift_lid_dec7/epoch_5/raw \
        --task_instruction "Lift the lid off the bowl" \
        --checkpoint_path /path/to/qwen/checkpoint \
        --advantage_threshold 0.3 \
        --ranking_frames 5

    # Label episodes (called by convert script)
    python qwen_reward_labeling.py \
        --data_dir /path/to/raw/hdf5s \
        --task_instruction "task description" \
        --checkpoint_path /path/to/checkpoint \
        --advantage_threshold 0.3 \
        --ranking_frames 5 \
        --output_json labels.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

# Add the third_party directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "third_party" / "emboided_reward"))

try:
    from embodied_reward_util import (
        sample_even_frames,
        Frame,
        print_reward_labeling_header,
        print_percentile_calculation,
        print_episode_ranking,
        print_reward_summary,
        format_reward_progress,
        save_reward_visualization,
        export_reward_labels_csv
    )
    from Qwen_reward_inference_util import (
        load_model_and_tokenizer,
        build_qwen3vl_sft_sample,
        run_inference
    )
except ImportError as e:
    print(f"Error importing reward labeling utilities: {e}")
    print("Make sure embodied_reward_util.py and Qwen_reward_inference_util.py are in the correct location.")
    print(f"sys.path: {sys.path}")
    sys.exit(1)


def hdf5_to_video(hdf5_path: str, fps: int = 10) -> str:
    """
    Convert HDF5 episode to temporary video file for reward inference.
    
    Args:
        hdf5_path: Path to HDF5 file
        fps: Frame rate for video
    
    Returns:
        Path to temporary video file
    """
    # Create temporary video file
    temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_video_path = temp_video.name
    temp_video.close()
    
    with h5py.File(hdf5_path, 'r') as f:
        images_group = f['observations']['images']
        
        # Get camera frames (prefer cam_head, fallback to ego_cam)
        if 'cam_head' in images_group:
            frames_data = images_group['cam_head'][:]
        elif 'ego_cam' in images_group:
            frames_data = images_group['ego_cam'][:]
        else:
            raise KeyError(f"No camera found in {hdf5_path}")
        
        # Decompress if JPEG compressed
        if frames_data.dtype == object or len(frames_data.shape) == 1:
            frames = []
            for jpeg_bytes in frames_data:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(bytes(jpeg_bytes)))
                frames.append(np.array(img))
            frames_data = np.array(frames)
        
        # Write video using OpenCV
        if len(frames_data) == 0:
            raise ValueError(f"No frames in {hdf5_path}")
        
        height, width = frames_data[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        
        for frame in frames_data:
            # Convert RGB to BGR for OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)
        
        out.release()
    
    return temp_video_path


def reward_inference_qwen(
    hdf5_path: str,
    task_instruction: str,
    model,
    tokenizer,
    max_frames: int = 30,
    image_rotation: int = 0,
    inference_batch_size: int = 30,
) -> Dict[str, Any]:
    """
    Run reward inference on an HDF5 file using Qwen model.
    
    Args:
        hdf5_path: Path to HDF5 file
        task_instruction: Detailed task description for the model
        model: Loaded Qwen model
        tokenizer: Loaded tokenizer
        max_frames: Maximum number of frames to sample
        image_rotation: Rotation angle (0, 90, 180, 270)
        inference_batch_size: Batch size for inference
    
    Returns:
        Dictionary with 'reward_pred', 'corrected_total_reward'
    """
    # Convert HDF5 to temporary video
    video_path = hdf5_to_video(hdf5_path)
    
    try:
        # Sample frames from video
        video_frames = sample_even_frames(video_path, max_frames=max_frames, rotate_angle=image_rotation)
        
        # Add fake GT reward to the frames (required by build_qwen3vl_sft_sample)
        for i, fr in enumerate(video_frames):
            fr.gt_reward = 0
        
        # Build the inference samples (one sample per frame)
        test_video_data = [
            build_qwen3vl_sft_sample(
                frames=video_frames,
                task=task_instruction,
                anchor_gt_idx=1,
                shuffle_frames=False,
                target_frame_idx=i + 1,
            )
            for i in range(len(video_frames))
        ]
        
        # Run inference
        reward_pred = run_inference(model, tokenizer, test_video_data, inference_batch_size=inference_batch_size)
        
        # Calculate total reward
        corrected_total_reward = sum(reward_pred)
        
        return {
            'reward_pred': reward_pred,
            'corrected_reward_pred': reward_pred,  # For compatibility with old API
            'total_reward': corrected_total_reward,
            'corrected_total_reward': corrected_total_reward
        }
    
    finally:
        # Clean up temporary video
        if os.path.exists(video_path):
            os.remove(video_path)


def label_episodes(
    data_dir: str,
    task_instruction: str,
    checkpoint_path: str,
    max_frames: int = 30,
    image_rotation: int = 0,
    advantage_threshold: float = 0.3,
    inference_batch_size: int = 30,
    base_model: str = None,
    dtype: str = "bf16",
    ranking_frames: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    Label all HDF5 episodes in a directory with advantage labels using Qwen model.
    
    Args:
        data_dir: Directory containing HDF5 files
        task_instruction: Task description for model
        checkpoint_path: Path to Qwen checkpoint directory
        max_frames: Maximum frames to sample per episode
        image_rotation: Rotation angle for images
        advantage_threshold: Percentile threshold for advantage labeling (0.0-1.0)
                           e.g., 0.3 means top 30% episodes get Advantage=True
        inference_batch_size: Batch size for inference (30 works well)
        base_model: Base model name (default: unsloth/Qwen3-VL-8B-Instruct)
        dtype: Data type for model (bf16, fp16)
        ranking_frames: Number of frames from the end to use for ranking (default: 5, use 0 for all frames)
    
    Returns:
        Dictionary mapping filename to labeling results
    """
    # Get all HDF5 files
    data_path = Path(data_dir)
    hdf5_files = sorted(data_path.glob("*.hdf5"))
    
    if not hdf5_files:
        print(f"No HDF5 files found in {data_dir}")
        return {}
    
    # Print header
    print_reward_labeling_header(len(hdf5_files))
    
    # Load model and tokenizer
    print(f"\nLoading Qwen model from: {checkpoint_path}")
    print(f"  Base model: {base_model or 'unsloth/Qwen3-VL-8B-Instruct'}")
    print(f"  Dtype: {dtype}")
    
    model, tokenizer = load_model_and_tokenizer(
        checkpoint_path,
        base_model=base_model,
        dtype=dtype
    )
    
    print(f"Model loaded successfully on device: {model.device}")
    print()
    
    # Process files sequentially (model is already on GPU, no need for parallel)
    results = {}
    
    for hdf5_file in tqdm(hdf5_files, desc="Computing rewards"):
        try:
            result = reward_inference_qwen(
                str(hdf5_file),
                task_instruction,
                model,
                tokenizer,
                max_frames,
                image_rotation,
                inference_batch_size,
            )
            
            # Store results without advantage label yet
            results[hdf5_file.name] = {
                'advantage': None,  # Will be set after percentile calculation
                'reward_pred': result['reward_pred'],
                'corrected_reward_pred': result['corrected_reward_pred'],
                'total_reward': result['total_reward'],
                'corrected_total_reward': result['corrected_total_reward'],
                'ranking_reward': None,  # Will be calculated below
            }
            
            # Calculate ranking reward based on last N frames
            reward_pred = result['reward_pred']
            if ranking_frames > 0 and len(reward_pred) > 0:
                # Use sum of last N frames for ranking
                last_n_frames = reward_pred[-ranking_frames:]
                ranking_reward = sum(last_n_frames)
            else:
                # Use total reward (all frames)
                ranking_reward = result['corrected_total_reward']
            
            results[hdf5_file.name]['ranking_reward'] = ranking_reward
            
            print(format_reward_progress(hdf5_file.name, result['corrected_total_reward']))
            
        except Exception as e:
            print(f"Error processing {hdf5_file.name}: {e}")
            import traceback
            traceback.print_exc()
            results[hdf5_file.name] = {
                'advantage': False,
                'error': str(e),
                'corrected_total_reward': 0,
            }
    
    # Calculate percentile threshold for advantage labeling
    valid_results = [(name, data) for name, data in results.items() 
                     if 'error' not in data and data['corrected_total_reward'] is not None]
    
    if not valid_results:
        print("\nNo valid results to calculate percentile")
        return results
    
    # Sort by ranking_reward (descending) instead of corrected_total_reward
    sorted_results = sorted(valid_results, key=lambda x: x[1]['ranking_reward'], reverse=True)
    
    # Calculate cutoff index for top X%
    num_good = max(1, int(len(sorted_results) * advantage_threshold))
    cutoff_reward = sorted_results[num_good - 1][1]['ranking_reward'] if num_good > 0 else float('inf')
    
    # Print ranking method info
    if ranking_frames > 0:
        print(f"\nRanking method: Sum of last {ranking_frames} frames")
    else:
        print(f"\nRanking method: Sum of all {max_frames} frames")
    
    # Print percentile calculation info
    print_percentile_calculation(advantage_threshold, len(sorted_results), num_good, cutoff_reward)
    
    # Assign advantage labels based on percentile
    for i, (name, data) in enumerate(sorted_results):
        is_good = i < num_good
        results[name]['advantage'] = is_good
        results[name]['percentile_rank'] = i + 1
    
    # Print episode rankings
    print_episode_ranking(sorted_results, advantage_threshold)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Label H1 robot episodes with Qwen reward model"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing HDF5 files to label"
    )
    parser.add_argument(
        "--task_instruction",
        type=str,
        required=True,
        help="Detailed task instruction for the reward model"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to Qwen checkpoint directory"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=30,
        help="Maximum number of frames to sample per episode"
    )
    parser.add_argument(
        "--image_rotation",
        type=int,
        default=0,
        choices=[0, 90, 180, 270, -90, -180, -270],
        help="Image rotation angle"
    )
    parser.add_argument(
        "--advantage_threshold",
        type=float,
        default=0.3,
        help="Percentile threshold for advantage labeling (0.0-1.0). "
             "e.g., 0.3 means top 30%% episodes get Advantage=True"
    )
    parser.add_argument(
        "--ranking_frames",
        type=int,
        default=5,
        help="Number of frames from the end to use for ranking (default: 5, use 0 for all frames)"
    )
    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=30,
        help="Batch size for inference (default: 30)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name (default: unsloth/Qwen3-VL-8B-Instruct)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Model data type"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID to use (default: 0)"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save results as JSON"
    )
    
    args = parser.parse_args()
    
    # Set GPU device
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint path not found: {args.checkpoint_path}")
        sys.exit(1)
    
    # Label episodes
    results = label_episodes(
        data_dir=args.data_dir,
        task_instruction=args.task_instruction,
        checkpoint_path=args.checkpoint_path,
        max_frames=args.max_frames,
        image_rotation=args.image_rotation,
        advantage_threshold=args.advantage_threshold,
        inference_batch_size=args.inference_batch_size,
        base_model=args.base_model,
        dtype=args.dtype,
        ranking_frames=args.ranking_frames,
    )
    
    # Print summary
    print_reward_summary(results, args.advantage_threshold)
    
    # Save outputs if requested
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nJSON results saved to: {output_path}")
        
        # Also save CSV and visualization
        csv_path = output_path.with_suffix('.csv')
        export_reward_labels_csv(results, str(csv_path))
        
        # Create visualization if matplotlib is available
        try:
            valid_results = [(name, data) for name, data in results.items() 
                           if 'error' not in data and data['corrected_total_reward'] is not None]
            sorted_results = sorted(valid_results, key=lambda x: x[1]['corrected_total_reward'], reverse=True)
            
            png_path = output_path.with_suffix('.png')
            save_reward_visualization(sorted_results, args.advantage_threshold, str(png_path))
        except Exception as e:
            print(f"Could not generate visualization: {e}")


if __name__ == "__main__":
    main()

