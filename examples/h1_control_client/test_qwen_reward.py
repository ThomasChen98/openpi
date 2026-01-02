"""
Test script for Qwen-based reward labeling

This script tests the Qwen reward model on a single video or HDF5 file.

Usage:
    # Test on a video
    uv run python examples/h1_control_client/test_qwen_reward.py \
        --video_path third_party/videos/fold_towel_epoch_2_episode_000000_bad.mp4 \
        --task_instruction "Fold the towel into a small square" \
        --checkpoint_path third_party/emboided_reward/checkpoint-1240
    
    # Test on an HDF5 file
    uv run python examples/h1_control_client/test_qwen_reward.py \
        --hdf5_path examples/h1_control_client/h1_data_auto/fold_towel/epoch_5/raw/episode_0.hdf5 \
        --task_instruction "Fold the towel into a small square" \
        --checkpoint_path third_party/emboided_reward/checkpoint-1240
"""

import sys
import argparse
from pathlib import Path

# Add the third_party directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "third_party" / "emboided_reward"))

from qwen_reward_labeling import reward_inference_qwen, hdf5_to_video, load_model_and_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Test Qwen reward model on a video or HDF5 file")
    
    # Input: either video or HDF5
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video_path", type=str, help="Path to video file (.mp4)")
    group.add_argument("--hdf5_path", type=str, help="Path to HDF5 file (.hdf5)")
    
    # Model and task
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to Qwen checkpoint directory"
    )
    parser.add_argument(
        "--task_instruction",
        type=str,
        required=True,
        help="Task instruction for the reward model"
    )
    
    # Optional parameters
    parser.add_argument("--max_frames", type=int, default=30, help="Maximum frames to sample")
    parser.add_argument("--image_rotation", type=int, default=0, help="Image rotation angle")
    parser.add_argument("--inference_batch_size", type=int, default=30, help="Batch size for inference")
    parser.add_argument("--base_model", type=str, default=None, help="Base model name")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="Model dtype")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use (default: 0)")
    
    args = parser.parse_args()
    
    # Set GPU device
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Load model
    print(f"\nLoading Qwen model from: {args.checkpoint_path}")
    print(f"  Base model: {args.base_model or 'unsloth/Qwen3-VL-8B-Instruct'}")
    print(f"  Dtype: {args.dtype}")
    print(f"  GPU: {args.gpu}")
    
    model, tokenizer = load_model_and_tokenizer(
        args.checkpoint_path,
        base_model=args.base_model,
        dtype=args.dtype
    )
    
    print(f"Model loaded successfully on device: {model.device}\n")
    
    # Prepare input path
    if args.video_path:
        input_path = args.video_path
        print(f"Testing on video: {input_path}")
    else:
        input_path = args.hdf5_path
        print(f"Testing on HDF5: {input_path}")
    
    # Run inference
    print(f"\nRunning inference...")
    print(f"  Task: {args.task_instruction}")
    print(f"  Max frames: {args.max_frames}")
    print(f"  Image rotation: {args.image_rotation}")
    
    # If video path, convert to temporary HDF5-like structure
    if args.video_path:
        # For video files, we need a different approach
        # Import the necessary functions
        from embodied_reward_util import sample_even_frames, Frame
        from Qwen_reward_inference_util import build_qwen3vl_sft_sample, run_inference
        
        # Sample frames from video
        video_frames = sample_even_frames(args.video_path, max_frames=args.max_frames, rotate_angle=args.image_rotation)
        
        # Add fake GT reward to the frames (required by build_qwen3vl_sft_sample)
        for i, fr in enumerate(video_frames):
            fr.gt_reward = 0
        
        # Build the inference samples (one sample per frame)
        test_video_data = [
            build_qwen3vl_sft_sample(
                frames=video_frames,
                task=args.task_instruction,
                anchor_gt_idx=1,
                shuffle_frames=False,
                target_frame_idx=i + 1,
            )
            for i in range(len(video_frames))
        ]
        
        # Run inference
        reward_pred = run_inference(model, tokenizer, test_video_data, inference_batch_size=args.inference_batch_size)
        
        result = {
            'reward_pred': reward_pred,
            'total_reward': sum(reward_pred),
            'corrected_total_reward': sum(reward_pred)
        }
    else:
        # Use the HDF5 inference function
        result = reward_inference_qwen(
            args.hdf5_path,
            args.task_instruction,
            model,
            tokenizer,
            args.max_frames,
            args.image_rotation,
            args.inference_batch_size,
        )
    
    # Print results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Total reward: {result['corrected_total_reward']:.1f}")
    print(f"\nFrame-by-frame rewards:")
    for i, reward in enumerate(result['reward_pred']):
        print(f"  Frame {i+1:2d}: {reward:6.1f}")
    print(f"{'='*80}")
    
    # Simple interpretation
    avg_reward = result['corrected_total_reward'] / len(result['reward_pred'])
    if avg_reward > 70:
        print("✓ High quality episode (average reward > 70)")
    elif avg_reward > 40:
        print("~ Medium quality episode (average reward 40-70)")
    else:
        print("✗ Low quality episode (average reward < 40)")


if __name__ == "__main__":
    main()

