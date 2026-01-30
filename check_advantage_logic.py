"""
Check action chunk advantage logic by computing advantages and visualizing results.

This script:
1. Runs compute_action_chunk_advantages.py with config parameters
2. Loads and analyzes the results
3. Creates MP4 visualizations for advantage=True and advantage=False chunks
4. Prints detailed statistics

Usage:
    python check_advantage_logic.py
"""

import os
import sys
import glob
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess

import numpy as np
import pandas as pd
from PIL import Image
import io
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def run_advantage_computation(config_params: Dict) -> str:
    """Run compute_action_chunk_advantages.py with given parameters."""
    print("\n" + "="*80)
    print("RUNNING ACTION CHUNK ADVANTAGE COMPUTATION")
    print("="*80)
    
    data_dir = config_params['data_dir']
    
    # Build command
    cmd = [
        "/home/yuxin/miniconda/bin/python",
        "examples/h1_control_client/compute_action_chunk_advantages.py",
        "--data-dir", data_dir,
        "--task-instruction", config_params['task_instruction'],
        "--checkpoint-path", config_params['checkpoint_path'],
        "--reward-method", config_params['reward_method'],
        "--max-frames", str(config_params['max_frames']),
        "--look-ahead-window", str(config_params['look_ahead_window']),
        "--advantage-threshold", str(config_params['advantage_threshold']),
        "--distance-threshold", str(config_params['distance_threshold']),
        "--device", "cuda",
    ]
    
    print(f"\nRunning command:")
    print(" ".join(cmd))
    print()
    
    # Set environment variables (fix cuBLAS version mismatch)
    env = os.environ.copy()
    nvidia_lib_path = "/home/yuxin/miniconda/lib/python3.13/site-packages/nvidia/cublas/lib:/home/yuxin/miniconda/lib/python3.13/site-packages/nvidia/cu13/lib"
    env['LD_LIBRARY_PATH'] = f"{nvidia_lib_path}:{env.get('LD_LIBRARY_PATH', '')}"
    env['CUDA_VISIBLE_DEVICES'] = "0"
    env['QWEN_REWARD_CHECKPOINT_PATH'] = config_params['checkpoint_path']
    
    # Run the command
    result = subprocess.run(cmd, capture_output=False, text=True, env=env)
    
    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        sys.exit(1)
    
    return data_dir


def load_episode_images(parquet_path: str) -> List[Image.Image]:
    """Load ego camera images from a parquet episode file."""
    df = pd.read_parquet(parquet_path)
    df_sorted = df.sort_values('frame_index', kind='stable')
    ego_images = [
        Image.open(io.BytesIO(x['bytes'])).convert('RGB')
        for x in df_sorted['ego_cam'].tolist()
    ]
    return ego_images


def load_episode_data(parquet_path: str) -> Tuple[List[Image.Image], pd.DataFrame]:
    """Load images and dataframe from episode."""
    df = pd.read_parquet(parquet_path)
    df_sorted = df.sort_values('frame_index', kind='stable')
    ego_images = [
        Image.open(io.BytesIO(x['bytes'])).convert('RGB')
        for x in df_sorted['ego_cam'].tolist()
    ]
    return ego_images, df_sorted


def save_video(images: List[Image.Image], output_path: str, fps: int = 10):
    """Save a list of PIL images as an MP4 video."""
    if not images:
        print(f"Warning: No images to save for {output_path}")
        return
    
    # Get dimensions from first image
    width, height = images[0].size
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for img in images:
        # Convert PIL to OpenCV format (RGB to BGR)
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    out.release()
    print(f"Saved video: {output_path}")


def analyze_advantages(data_dir: str, output_dir: str, action_chunk_size: int = 50):
    """Analyze advantage labels and create visualizations of actual action chunks."""
    print("\n" + "="*80)
    print("ANALYZING ADVANTAGE LABELS - ACTUAL ACTION CHUNKS")
    print("="*80)
    
    # Find all episode files
    episode_files = sorted(glob.glob(str(Path(data_dir) / "episode_*.parquet")))
    
    if not episode_files:
        print(f"Error: No episode files found in {data_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(episode_files)} episodes")
    print(f"Action chunk size: {action_chunk_size} frames")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all advantage data and valid action chunks
    all_episodes_data = []
    total_frames_with_adv = 0
    total_frames_without_adv = 0
    all_good_chunks = []  # (episode_name, images, start_frame, episode_length, normalized_position)
    all_bad_chunks = []   # (episode_name, images, start_frame, episode_length, normalized_position)
    
    for ep_file in tqdm(episode_files, desc="Loading episodes"):
        advantage_file = ep_file.replace('.parquet', '_action_chunk_advantages.pkl')
        
        if not os.path.exists(advantage_file):
            print(f"Warning: No advantage file found for {ep_file}")
            continue
        
        # Load advantage labels
        with open(advantage_file, 'rb') as f:
            advantages = pickle.load(f)
        
        # Load episode data
        images, df = load_episode_data(ep_file)
        
        episode_name = Path(ep_file).stem
        
        # Collect frame information and valid action chunks
        frames_with_adv = []
        frames_without_adv = []
        episode_length = len(images)
        
        for frame_idx in range(len(advantages)):
            # Check if we have enough frames for a full action chunk
            if frame_idx + action_chunk_size <= len(images):
                # Compute normalized position (0.0 = start, 1.0 = end)
                normalized_pos = frame_idx / max(episode_length - 1, 1)
                
                if advantages[frame_idx]:
                    frames_with_adv.append(frame_idx)
                    total_frames_with_adv += 1
                    all_good_chunks.append((episode_name, images, frame_idx, episode_length, normalized_pos))
                else:
                    frames_without_adv.append(frame_idx)
                    total_frames_without_adv += 1
                    all_bad_chunks.append((episode_name, images, frame_idx, episode_length, normalized_pos))
            else:
                # Frame too close to end, can't form full action chunk
                if advantages[frame_idx]:
                    total_frames_with_adv += 1
                else:
                    total_frames_without_adv += 1
        
        all_episodes_data.append({
            'episode_name': episode_name,
            'episode_file': ep_file,
            'images': images,
            'df': df,
            'advantages': advantages,
            'frames_with_adv': frames_with_adv,
            'frames_without_adv': frames_without_adv,
        })
    
    # Print detailed statistics
    print("\n" + "="*80)
    print("DETAILED STATISTICS")
    print("="*80)
    
    print(f"\nTotal episodes: {len(all_episodes_data)}")
    print(f"Total frames with Advantage=True: {total_frames_with_adv}")
    print(f"Total frames with Advantage=False: {total_frames_without_adv}")
    print(f"Valid action chunks (Adv=True): {len(all_good_chunks)}")
    print(f"Valid action chunks (Adv=False): {len(all_bad_chunks)}")
    print(f"Percentage with Advantage=True: {100*total_frames_with_adv/(total_frames_with_adv+total_frames_without_adv):.1f}%")
    print(f"Percentage with Advantage=False: {100*total_frames_without_adv/(total_frames_with_adv+total_frames_without_adv):.1f}%")
    
    print("\n" + "-"*80)
    print("PER-EPISODE BREAKDOWN")
    print("-"*80)
    print(f"{'Episode':<25} {'Adv=True':<15} {'Adv=False':<15} {'Total':<10} {'% Adv=True':<15}")
    print("-"*80)
    
    for ep_data in all_episodes_data:
        n_true = len(ep_data['frames_with_adv'])
        n_false = len(ep_data['frames_without_adv'])
        total = n_true + n_false
        pct = 100 * n_true / total if total > 0 else 0
        
        print(f"{ep_data['episode_name']:<25} {n_true:<15} {n_false:<15} {total:<10} {pct:<15.1f}%")
    
    # Save sample ACTION CHUNK videos (50 frames each)
    print("\n" + "="*80)
    print("CREATING SAMPLE ACTION CHUNK VIDEOS")
    print("="*80)
    print(f"\nAction chunk size: {action_chunk_size} frames")
    print(f"Total good chunks available: {len(all_good_chunks)}")
    print(f"Total bad chunks available: {len(all_bad_chunks)}")
    
    # Randomly sample chunks
    num_samples_per_type = 100
    import random
    random.seed(42)  # For reproducibility
    
    good_samples = random.sample(all_good_chunks, min(num_samples_per_type, len(all_good_chunks)))
    bad_samples = random.sample(all_bad_chunks, min(num_samples_per_type, len(all_bad_chunks)))
    
    print(f"\nSaving {len(good_samples)} good action chunks (Advantage=True)...")
    print(f"Saving {len(bad_samples)} bad action chunks (Advantage=False)...")
    
    # Save good chunks
    for i, (ep_name, images, start_frame, _, _) in enumerate(tqdm(good_samples, desc="Saving good chunks")):
        end_frame = start_frame + action_chunk_size
        chunk_images = images[start_frame:end_frame]
        video_path = os.path.join(output_dir, f"TRUE_{i:03d}_{ep_name}_frame{start_frame:04d}-{end_frame:04d}.mp4")
        save_video(chunk_images, video_path, fps=10)
    
    # Save bad chunks
    for i, (ep_name, images, start_frame, _, _) in enumerate(tqdm(bad_samples, desc="Saving bad chunks")):
        end_frame = start_frame + action_chunk_size
        chunk_images = images[start_frame:end_frame]
        video_path = os.path.join(output_dir, f"FALSE_{i:03d}_{ep_name}_frame{start_frame:04d}-{end_frame:04d}.mp4")
        save_video(chunk_images, video_path, fps=10)
    
    # Analyze temporal distribution
    print("\n" + "="*80)
    print("TEMPORAL DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Extract normalized positions
    good_positions = [chunk[4] for chunk in all_good_chunks]
    bad_positions = [chunk[4] for chunk in all_bad_chunks]
    
    print(f"\nAnalyzing temporal distribution of {len(all_good_chunks)} good and {len(all_bad_chunks)} bad chunks...")
    
    # Create histograms
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Histogram of True chunks
    ax1 = axes[0]
    ax1.hist(good_positions, bins=20, color='green', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Normalized Position in Episode (0=start, 1=end)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax1.set_title(f'Distribution of Advantage=TRUE Action Chunks (n={len(good_positions)})', 
                  fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=12)
    
    # Plot 2: Histogram of False chunks
    ax2 = axes[1]
    ax2.hist(bad_positions, bins=20, color='red', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Normalized Position in Episode (0=start, 1=end)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax2.set_title(f'Distribution of Advantage=FALSE Action Chunks (n={len(bad_positions)})', 
                  fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=12)
    
    # Plot 3: Overlay comparison
    ax3 = axes[2]
    ax3.hist(good_positions, bins=20, color='green', alpha=0.5, label='Advantage=TRUE', edgecolor='black')
    ax3.hist(bad_positions, bins=20, color='red', alpha=0.5, label='Advantage=FALSE', edgecolor='black')
    ax3.set_xlabel('Normalized Position in Episode (0=start, 1=end)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax3.set_title('Temporal Distribution Comparison', fontsize=16, fontweight='bold')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'temporal_distribution.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved temporal distribution plot: {plot_path}")
    plt.close()
    
    # Compute statistics by quartile
    print("\n" + "-"*80)
    print("TEMPORAL DISTRIBUTION BY QUARTILE")
    print("-"*80)
    print(f"{'Quartile':<20} {'True Count':<15} {'False Count':<15} {'True %':<15} {'False %':<15}")
    print("-"*80)
    
    quartiles = [(0.0, 0.25, "Q1 (0-25%)"), 
                 (0.25, 0.5, "Q2 (25-50%)"),
                 (0.5, 0.75, "Q3 (50-75%)"),
                 (0.75, 1.0, "Q4 (75-100%)")]
    
    for q_min, q_max, q_name in quartiles:
        true_in_q = sum(1 for pos in good_positions if q_min <= pos < q_max or (q_max == 1.0 and pos == 1.0))
        false_in_q = sum(1 for pos in bad_positions if q_min <= pos < q_max or (q_max == 1.0 and pos == 1.0))
        true_pct = 100 * true_in_q / len(good_positions) if good_positions else 0
        false_pct = 100 * false_in_q / len(bad_positions) if bad_positions else 0
        print(f"{q_name:<20} {true_in_q:<15} {false_in_q:<15} {true_pct:<15.1f} {false_pct:<15.1f}")
    
    # Compute mean and std
    print("\n" + "-"*80)
    print("STATISTICAL SUMMARY")
    print("-"*80)
    good_mean = np.mean(good_positions) if good_positions else 0
    good_std = np.std(good_positions) if good_positions else 0
    bad_mean = np.mean(bad_positions) if bad_positions else 0
    bad_std = np.std(bad_positions) if bad_positions else 0
    
    print(f"\nAdvantage=TRUE chunks:")
    print(f"  Mean position: {good_mean:.3f} (0=start, 1=end)")
    print(f"  Std deviation: {good_std:.3f}")
    print(f"  Interpretation: {'Balanced' if 0.4 <= good_mean <= 0.6 else 'Early bias' if good_mean < 0.4 else 'Late bias'}")
    
    print(f"\nAdvantage=FALSE chunks:")
    print(f"  Mean position: {bad_mean:.3f} (0=start, 1=end)")
    print(f"  Std deviation: {bad_std:.3f}")
    print(f"  Interpretation: {'Balanced' if 0.4 <= bad_mean <= 0.6 else 'Early bias' if bad_mean < 0.4 else 'Late bias'}")
    
    # Check for imbalance
    print("\n" + "-"*80)
    print("BALANCE ASSESSMENT")
    print("-"*80)
    
    if abs(good_mean - 0.5) < 0.1:
        print("✓ Good chunks are WELL-BALANCED across the task timeline")
    elif good_mean < 0.5:
        print("⚠ Good chunks are CONCENTRATED in EARLY stages of episodes")
    else:
        print("⚠ Good chunks are CONCENTRATED in LATE stages of episodes")
    
    if abs(bad_mean - 0.5) < 0.1:
        print("✓ Bad chunks are WELL-BALANCED across the task timeline")
    elif bad_mean < 0.5:
        print("⚠ Bad chunks are CONCENTRATED in EARLY stages of episodes")
    else:
        print("⚠ Bad chunks are CONCENTRATED in LATE stages of episodes")
    
    # Create detailed frame-by-frame analysis for first episode
    print("\n" + "="*80)
    print("DETAILED FRAME-BY-FRAME ANALYSIS (First Episode)")
    print("="*80)
    
    if all_episodes_data:
        ep_data = all_episodes_data[0]
        advantages = ep_data['advantages']
        
        print(f"\nEpisode: {ep_data['episode_name']}")
        print(f"Total frames: {len(advantages)}")
        print("\nFrame-by-frame advantage labels (showing first 100 frames):")
        print("Frame index: [Advantage label]")
        
        for i in range(min(100, len(advantages))):
            label = "TRUE " if advantages[i] else "FALSE"
            if i % 10 == 0:
                print()
                print(f"Frames {i:3d}-{min(i+9, len(advantages)-1):3d}: ", end="")
            print(f"{label} ", end="")
        print("\n")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Action chunk videos saved:")
    print(f"  - {len(good_samples)} good chunks (Advantage=True, 50 frames each)")
    print(f"  - {len(bad_samples)} bad chunks (Advantage=False, 50 frames each)")


def main():
    # Configuration parameters from training_config_insert_bottle_jan22_Ours.yaml
    config_params = {
        'task_name': 'insert_bottle_jan22_Ours_H300',
        'task_instruction': 'Pick up the bottle and insert it into the tray. Keep the bottle perfectly upright (vertical) and aligned straight.',
        'reward_method': 'Ours',
        'checkpoint_path': '/home/yuxin/Projects/openpi/third_party/emboided_reward/IB-checkpoint-1275',
        'max_frames': 30,
        'look_ahead_window': 80,
        'advantage_threshold': 0.33,
        'distance_threshold': 0.45,
        'data_dir': '/home/yuxin/Projects/openpi/examples/h1_control_client/h1_data_lerobot/insert_bottle_jan23_Ourstest/epoch_1/data/chunk-000',
    }
    
    print("="*80)
    print("ACTION CHUNK ADVANTAGE LOGIC CHECKER")
    print("="*80)
    print("\nConfiguration:")
    for key, value in config_params.items():
        print(f"  {key}: {value}")
    
    # Output directory for visualizations
    output_dir = f"/home/yuxin/Projects/openpi/advantage_analysis_{config_params['task_name']}"
    
    # Check if advantage files already exist
    data_dir = config_params['data_dir']
    advantage_files = glob.glob(str(Path(data_dir) / "*_action_chunk_advantages.pkl"))
    
    if advantage_files:
        print(f"\n✓ Found {len(advantage_files)} existing advantage files")
        print("Skipping computation and going directly to analysis...")
    else:
        # Run advantage computation
        run_advantage_computation(config_params)
    
    # Analyze results and create visualizations
    analyze_advantages(data_dir, output_dir)
    
    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nYou can now:")
    print("  1. Review the printed statistics above")
    print("  2. Watch the MP4 files to see advantage=True vs advantage=False chunks")
    print("  3. Verify that the advantage labeling logic is working correctly")


if __name__ == "__main__":
    main()
