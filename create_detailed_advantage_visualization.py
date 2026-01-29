"""
Create detailed visualizations with annotations showing advantage labels, 
reward values, and visual similarities.

This script creates annotated MP4 videos showing:
- Frame-by-frame advantage labels (True/False)
- Reward values over time
- Visual similarity information
"""

import os
import sys
import glob
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def load_episode_data(parquet_path: str) -> Tuple[List[Image.Image], pd.DataFrame]:
    """Load images and dataframe from episode."""
    df = pd.read_parquet(parquet_path)
    df_sorted = df.sort_values('frame_index', kind='stable')
    ego_images = [
        Image.open(io.BytesIO(x['bytes'])).convert('RGB')
        for x in df_sorted['ego_cam'].tolist()
    ]
    return ego_images, df_sorted


def add_text_overlay(image: Image.Image, frame_idx: int, adv_label: bool, 
                      reward_val: float, total_frames: int) -> Image.Image:
    """Add text overlay to image with proper layout."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    img_width, img_height = img_copy.size
    
    # Try to use a nice font, fallback to default if not available
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 50)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 35)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Create background bar across full width
    bar_height = 120
    bg_color = (0, 150, 0) if adv_label else (150, 0, 0)
    draw.rectangle((0, 0, img_width, bar_height), fill=bg_color)
    
    # Add text in multiple lines
    padding = 15
    y_pos = padding
    
    # Line 1: Frame number
    text1 = f"Frame {frame_idx}/{total_frames}"
    draw.text((padding, y_pos), text1, font=font_small, fill=(255, 255, 255))
    y_pos += 40
    
    # Line 2: Advantage and Reward
    adv_text = "TRUE " if adv_label else "FALSE"
    text2 = f"Advantage={adv_text} | Reward={reward_val:.3f}"
    draw.text((padding, y_pos), text2, font=font_small, fill=(255, 255, 255))
    
    return img_copy


def create_reward_plot(rewards: np.ndarray, advantages: np.ndarray, 
                       current_frame: int, window_size: int = 200) -> Image.Image:
    """Create a plot showing rewards and advantages over time."""
    fig, ax = plt.subplots(figsize=(16, 5))
    
    # Determine window
    start_idx = max(0, current_frame - window_size // 2)
    end_idx = min(len(rewards), current_frame + window_size // 2)
    
    # Plot rewards
    x = np.arange(start_idx, end_idx)
    ax.plot(x, rewards[start_idx:end_idx], 'b-', linewidth=3, label='Reward', zorder=2)
    
    # Highlight advantage=True regions
    for i in range(start_idx, end_idx):
        if advantages[i]:
            ax.axvspan(i-0.5, i+0.5, alpha=0.3, color='green', zorder=1)
    
    # Mark current frame
    ax.axvline(current_frame, color='red', linestyle='--', linewidth=3, label='Current Frame', zorder=3)
    
    # Larger fonts
    ax.set_xlabel('Frame Index', fontsize=20, fontweight='bold')
    ax.set_ylabel('Reward Value', fontsize=20, fontweight='bold')
    ax.set_title(f'Reward Trajectory (Frame {current_frame}/{len(rewards)})', fontsize=24, fontweight='bold')
    ax.legend(fontsize=18, loc='upper right')
    ax.grid(True, alpha=0.3, linewidth=1.5)
    
    # Larger tick labels
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # Tighter layout
    plt.tight_layout()
    
    # Convert to image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    plot_img = Image.frombytes('RGBA', canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1)
    plot_img = plot_img.convert('RGB')
    plt.close(fig)
    
    return plot_img


def create_annotated_video(episode_file: str, advantage_file: str, 
                           reward_file: str, output_path: str,
                           start_frame: int = 0, end_frame: int = None,
                           fps: int = 10, max_frames: int = None):
    """Create annotated video with advantage labels and reward plots.
    
    Args:
        max_frames: If set, limit to this many frames. If None, show all frames.
    """
    # Load data
    images, df = load_episode_data(episode_file)
    
    with open(advantage_file, 'rb') as f:
        advantages = pickle.load(f)
    
    with open(reward_file, 'rb') as f:
        rewards = pickle.load(f)
    
    if end_frame is None:
        end_frame = len(images)
    
    # Apply max_frames limit if specified
    if max_frames is not None:
        end_frame = min(end_frame, start_frame + max_frames)
    
    # Process frames
    annotated_frames = []
    
    # Target resolution for upscaling (HD resolution)
    TARGET_WIDTH = 1280
    TARGET_HEIGHT = 720
    
    for frame_idx in tqdm(range(start_frame, min(end_frame, len(images))), 
                          desc="Creating annotated video"):
        # Get original image and upscale it first
        img_original = images[frame_idx]
        
        # Upscale the robot image to HD resolution for better readability
        img_upscaled = img_original.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
        
        # Add advantage label
        adv_label = advantages[frame_idx]
        reward_val = rewards[frame_idx]
        
        # Add text overlay to the upscaled image
        img_annotated = add_text_overlay(img_upscaled, frame_idx, adv_label, reward_val, len(images))
        
        # Create reward plot (will be resized to match upscaled image width)
        plot_img = create_reward_plot(rewards, advantages, frame_idx, window_size=200)
        
        # Resize plot to match upscaled image width
        img_width = img_annotated.width
        plot_height = int(plot_img.height * img_width / plot_img.width)
        plot_img_resized = plot_img.resize((img_width, plot_height), Image.LANCZOS)
        
        # Combine image and plot vertically
        combined = Image.new('RGB', (img_width, img_annotated.height + plot_height))
        combined.paste(img_annotated, (0, 0))
        combined.paste(plot_img_resized, (0, img_annotated.height))
        
        annotated_frames.append(combined)
    
    if not annotated_frames:
        print(f"Warning: No frames to save for {output_path}")
        return
    
    # Save video
    width, height = annotated_frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in annotated_frames:
        frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        out.write(frame_cv)
    
    out.release()
    print(f"Saved annotated video: {output_path}")


def main():
    data_dir = "/home/yuxin/Projects/openpi/examples/h1_control_client/h1_data_lerobot/insert_bottle_jan22_Ours/epoch_1/data/chunk-000"
    output_dir = "/home/yuxin/Projects/openpi/advantage_analysis_insert_bottle_jan22_Ours/detailed_videos"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all episode files
    episode_files = sorted(glob.glob(str(Path(data_dir) / "episode_*.parquet")))
    
    print("="*80)
    print("CREATING DETAILED ANNOTATED VIDEOS - ALL FRAMES")
    print("="*80)
    print(f"\nNote: Robot images will be upscaled from 224x224 to 1280x720 (HD)")
    print(f"      This ensures text and plots are clearly readable")
    print(f"\nProcessing first 3 episodes with ALL frames...")
    print(f"Output directory: {output_dir}")
    print(f"\n⚠️  Warning: This will take longer and create larger files!\n")
    
    # Process first 3 episodes with ALL frames
    for ep_file in episode_files[:3]:
        episode_name = Path(ep_file).stem
        advantage_file = ep_file.replace('.parquet', '_action_chunk_advantages.pkl')
        reward_file = ep_file.replace('.parquet', '_Ours_reward.pkl')
        
        if not os.path.exists(advantage_file) or not os.path.exists(reward_file):
            print(f"Warning: Missing data files for {episode_name}")
            continue
        
        # Get total frame count
        images, _ = load_episode_data(ep_file)
        total_frames = len(images)
        
        print(f"\nProcessing {episode_name} ({total_frames} frames)...")
        
        # Create full annotated video with ALL frames (no limit)
        output_path = os.path.join(output_dir, f"{episode_name}_annotated_full.mp4")
        create_annotated_video(ep_file, advantage_file, reward_file, output_path, 
                              start_frame=0, end_frame=None, fps=10, max_frames=None)
    
    print("\n" + "="*80)
    print("DETAILED VIDEOS CREATED")
    print("="*80)
    print(f"\nVideos saved to: {output_dir}")
    print("\nThese videos show:")
    print("  - Frame-by-frame advantage labels (GREEN=True, RED=False)")
    print("  - Reward values at each frame")
    print("  - Reward trajectory over time with advantage regions highlighted")
    print("\nThis helps you understand:")
    print("  - Which frames are labeled as having advantage")
    print("  - How rewards correlate with advantage labels")
    print("  - The temporal structure of advantages within episodes")


if __name__ == "__main__":
    main()
