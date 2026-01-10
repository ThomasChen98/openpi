#!/usr/bin/env python3
"""
Analyze Alignment Between Human Labels and Qwen Reward Model

This script:
1. Loads human-labeled HDF5 episodes (advantage=True/False)
2. Runs Qwen reward model inference on the same episodes
3. Compares human labels vs model predictions
4. Generates alignment metrics and visualizations
5. Optionally converts HDF5 episodes to MP4 videos

Usage:
    python examples/h1_control_client/utils/analyze_reward_alignment.py \
        --data_dir examples/h1_control_client/h1_data_auto/fold_towel_reward_2/epoch_2/raw \
        --task_instruction "Fold the towel into a small square" \
        --checkpoint_path third_party/emboided_reward/checkpoint-1240 \
        --output_dir results/analysis \
        --ranking_frames 5 \
        --save_videos \
        --gpu 1
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import h5py
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Optional: seaborn for nicer heatmaps
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Add paths for imports
# Script is in examples/h1_control_client/utils/, need to go up to project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "third_party" / "emboided_reward"))
sys.path.insert(0, str(project_root / "examples" / "h1_control_client"))

# NOTE: Don't import qwen_reward_labeling here! It imports torch/CUDA.
# Import will be done in main() after setting CUDA_VISIBLE_DEVICES


def natural_sort_key(path):
    """
    Extract numeric parts from filename for natural sorting.
    E.g., episode_7.hdf5 -> (7,), episode_07.hdf5 -> (7,)
    This ensures episode_1, episode_2, ..., episode_10 order instead of lexicographic.
    """
    import re
    filename = path.name if hasattr(path, 'name') else str(path)
    # Extract all numbers from the filename
    numbers = re.findall(r'\d+', filename)
    # Convert to integers for proper numeric sorting
    return tuple(int(n) for n in numbers) if numbers else (0,)


def parse_dataset_info(data_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract dataset name and epoch from data directory path.
    
    Example:
        'examples/h1_control_client/h1_data_auto/fold_towel_reward/epoch_2/raw'
        -> ('fold_towel_reward', '2')
    
    Returns:
        (dataset_name, epoch_number) or (None, None) if not found
    """
    import re
    path = Path(data_dir)
    parts = path.parts
    
    dataset_name = None
    epoch = None
    
    # Look for epoch pattern (e.g., 'epoch_2')
    for i, part in enumerate(parts):
        if re.match(r'epoch_\d+', part):
            # Extract epoch number
            epoch_match = re.search(r'epoch_(\d+)', part)
            if epoch_match:
                epoch = epoch_match.group(1)
            # Dataset name is typically the parent of epoch folder
            if i > 0:
                dataset_name = parts[i - 1]
            break
    
    return dataset_name, epoch


@dataclass
class EpisodeResult:
    """Results for a single episode"""
    filename: str
    human_label: bool  # True=good, False=bad
    qwen_reward: float  # Total reward from Qwen model
    qwen_label: Optional[bool] = None  # Will be set after thresholding
    frame_rewards: List[float] = None
    episode_length: int = 0
    ranking_reward: Optional[float] = None  # Reward used for ranking (last N frames or total)
    
    def to_dict(self):
        return asdict(self)


def decompress_jpeg_images(compressed_data) -> np.ndarray:
    """Decompress JPEG-compressed images from HDF5."""
    from PIL import Image
    import io
    
    frames = []
    for jpeg_bytes in compressed_data:
        img = Image.open(io.BytesIO(bytes(jpeg_bytes)))
        frames.append(np.array(img))
    
    return np.array(frames)


def hdf5_to_mp4(hdf5_path: str, output_path: str, fps: int = 30):
    """
    Convert HDF5 episode to MP4 video.
    
    Args:
        hdf5_path: Path to HDF5 file
        output_path: Output MP4 path
        fps: Frames per second
    """
    import imageio
    import cv2
    
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
            frames_data = decompress_jpeg_images(frames_data)
        
        if len(frames_data) == 0:
            raise ValueError(f"No frames in {hdf5_path}")
        
        # Save as MP4 using imageio (preferred) or opencv fallback
        try:
            with imageio.get_writer(output_path, fps=fps, codec="libx264", pixelformat="yuv420p", quality=8) as writer:
                for frame in frames_data:
                    # Convert BGR to RGB if needed
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        writer.append_data(frame)
                    else:
                        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return "imageio-libx264"
        except Exception:
            try:
                with imageio.get_writer(output_path, fps=fps, codec="mpeg4", quality=8) as writer:
                    for frame in frames_data:
                        if len(frame.shape) == 3 and frame.shape[2] == 3:
                            writer.append_data(frame)
                        else:
                            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                return "imageio-mpeg4"
            except Exception:
                # OpenCV fallback
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                height, width = frames_data[0].shape[:2]
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                for frame in frames_data:
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = frame
                    out.write(frame_bgr)
                
                out.release()
                return "opencv-mp4v"


def load_human_labels(data_dir: str) -> Dict[str, Tuple[bool, int]]:
    """
    Load human labels from HDF5 files.
    
    Returns:
        Dict mapping filename to (advantage_label, episode_length)
    """
    data_path = Path(data_dir)
    hdf5_files = sorted(data_path.glob("*.hdf5"), key=natural_sort_key)
    
    labels = {}
    for filepath in hdf5_files:
        try:
            with h5py.File(filepath, 'r') as f:
                if 'advantage' in f.attrs:
                    advantage = bool(f.attrs['advantage'])
                    episode_length = int(f.attrs.get('episode_length', len(f['action'])))
                    labels[filepath.name] = (advantage, episode_length)
        except Exception as e:
            print(f"Warning: Could not read {filepath.name}: {e}")
    
    return labels


def run_qwen_inference(
    data_dir: str,
    model,
    tokenizer,
    task_instruction: str,
    reward_inference_fn,
    max_frames: int = 30,
    image_rotation: int = 0,
    inference_batch_size: int = 30,
) -> Dict[str, Dict]:
    """
    Run Qwen reward model inference on all HDF5 files.
    
    Returns:
        Dict mapping filename to inference results
    """
    data_path = Path(data_dir)
    hdf5_files = sorted(data_path.glob("*.hdf5"), key=natural_sort_key)
    
    results = {}
    for hdf5_file in tqdm(hdf5_files, desc="Running Qwen inference"):
        try:
            result = reward_inference_fn(
                str(hdf5_file),
                task_instruction,
                model,
                tokenizer,
                max_frames,
                image_rotation,
                inference_batch_size,
            )
            results[hdf5_file.name] = result
        except Exception as e:
            print(f"Error processing {hdf5_file.name}: {e}")
            results[hdf5_file.name] = {
                'error': str(e),
                'corrected_total_reward': 0,
                'reward_pred': []
            }
    
    return results


def calculate_alignment_metrics(results: List[EpisodeResult]) -> Dict:
    """Calculate alignment metrics between human and Qwen labels."""
    
    # Confusion matrix
    true_positive = sum(1 for r in results if r.human_label and r.qwen_label)
    false_positive = sum(1 for r in results if not r.human_label and r.qwen_label)
    true_negative = sum(1 for r in results if not r.human_label and not r.qwen_label)
    false_negative = sum(1 for r in results if r.human_label and not r.qwen_label)
    
    total = len(results)
    accuracy = (true_positive + true_negative) / total if total > 0 else 0
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Get rewards for good and bad episodes
    good_rewards = [r.qwen_reward for r in results if r.human_label]
    bad_rewards = [r.qwen_reward for r in results if not r.human_label]
    
    metrics = {
        'total_episodes': total,
        'confusion_matrix': {
            'true_positive': true_positive,
            'false_positive': false_positive,
            'true_negative': true_negative,
            'false_negative': false_negative,
        },
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'reward_statistics': {
            'good_episodes': {
                'count': len(good_rewards),
                'mean': float(np.mean(good_rewards)) if good_rewards else 0,
                'std': float(np.std(good_rewards)) if good_rewards else 0,
                'min': float(np.min(good_rewards)) if good_rewards else 0,
                'max': float(np.max(good_rewards)) if good_rewards else 0,
            },
            'bad_episodes': {
                'count': len(bad_rewards),
                'mean': float(np.mean(bad_rewards)) if bad_rewards else 0,
                'std': float(np.std(bad_rewards)) if bad_rewards else 0,
                'min': float(np.min(bad_rewards)) if bad_rewards else 0,
                'max': float(np.max(bad_rewards)) if bad_rewards else 0,
            },
            'separation': float(np.mean(good_rewards) - np.mean(bad_rewards)) if (good_rewards and bad_rewards) else 0,
        }
    }
    
    return metrics


def plot_alignment_analysis(results: List[EpisodeResult], output_dir: Path, checkpoint_path: str = None, 
                           dataset_name: str = None, epoch: str = None):
    """Generate visualization plots for alignment analysis."""
    
    # Prepare data
    human_labels = [r.human_label for r in results]
    qwen_rewards = [r.qwen_reward for r in results]
    qwen_labels = [r.qwen_label for r in results]
    
    good_rewards = [r.qwen_reward for r in results if r.human_label]
    bad_rewards = [r.qwen_reward for r in results if not r.human_label]
    
    # Create figure with subplots (expanded to 4x3 for stats panels)
    fig = plt.figure(figsize=(16, 14))
    
    # Add main title with dataset, epoch, and checkpoint info
    title_parts = ['Reward Alignment Analysis']
    if dataset_name:
        title_parts.append(f'Dataset: {dataset_name}')
    if epoch:
        title_parts.append(f'Epoch: {epoch}')
    if checkpoint_path:
        checkpoint_name = Path(checkpoint_path).name
        title_parts.append(f'Checkpoint: {checkpoint_name}')
    
    fig.suptitle(' | '.join(title_parts), fontsize=16, fontweight='bold', y=0.985)
    
    # Add more space between title and figures
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3, top=0.93, bottom=0.05)
    
    # 1. Reward distribution by human label
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist([good_rewards, bad_rewards], label=['Human: Good', 'Human: Bad'], 
             bins=20, alpha=0.7, color=['green', 'red'])
    ax1.set_xlabel('Qwen Total Reward')
    ax1.set_ylabel('Count')
    ax1.set_title('Reward Distribution by Human Label')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.boxplot([good_rewards, bad_rewards], labels=['Human: Good', 'Human: Bad'])
    ax2.set_ylabel('Qwen Total Reward')
    ax2.set_title('Reward Distribution Comparison')
    ax2.grid(True, alpha=0.3)
    
    # 3. Confusion matrix
    ax3 = fig.add_subplot(gs[0, 2])
    tp = sum(1 for r in results if r.human_label and r.qwen_label)
    fp = sum(1 for r in results if not r.human_label and r.qwen_label)
    fn = sum(1 for r in results if r.human_label and not r.qwen_label)
    tn = sum(1 for r in results if not r.human_label and not r.qwen_label)
    
    confusion = np.array([[tn, fp], [fn, tp]])
    
    if HAS_SEABORN:
        import seaborn as sns
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', ax=ax3,
                    xticklabels=['Qwen: Bad', 'Qwen: Good'],
                    yticklabels=['Human: Bad', 'Human: Good'])
    else:
        # Manual heatmap using matplotlib
        im = ax3.imshow(confusion, cmap='Blues', aspect='auto')
        ax3.set_xticks([0, 1])
        ax3.set_yticks([0, 1])
        ax3.set_xticklabels(['Qwen: Bad', 'Qwen: Good'])
        ax3.set_yticklabels(['Human: Bad', 'Human: Good'])
        # Add text annotations
        for i in range(2):
            for j in range(2):
                ax3.text(j, i, str(confusion[i, j]), ha='center', va='center',
                        color='white' if confusion[i, j] > confusion.max()/2 else 'black',
                        fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax3)
    
    ax3.set_title('Confusion Matrix')
    
    # 4. Scatter plot: rewards vs episode index
    ax4 = fig.add_subplot(gs[1, :])
    for i, result in enumerate(results):
        color = 'green' if result.human_label else 'red'
        marker = 'o' if result.qwen_label else 'x'
        ax4.scatter(i, result.qwen_reward, c=color, marker=marker, s=100, alpha=0.7)
    ax4.set_xlabel('Episode Index')
    ax4.set_ylabel('Qwen Total Reward')
    ax4.set_title('Reward by Episode (Color=Human Label, Marker=Qwen Label)\n'
                  'Green=Good, Red=Bad | Circle=Qwen Good, X=Qwen Bad')
    ax4.grid(True, alpha=0.3)
    
    # 5. Agreement analysis
    ax5 = fig.add_subplot(gs[2, 0])
    agree = sum(1 for r in results if r.human_label == r.qwen_label)
    disagree = len(results) - agree
    ax5.pie([agree, disagree], labels=['Agreement', 'Disagreement'], 
            autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    ax5.set_title(f'Overall Agreement\n({agree}/{len(results)} episodes)')
    
    # 6. Misclassification details
    ax6 = fig.add_subplot(gs[2, 1])
    false_positives = [r.qwen_reward for r in results if not r.human_label and r.qwen_label]
    false_negatives = [r.qwen_reward for r in results if r.human_label and not r.qwen_label]
    
    if false_positives or false_negatives:
        data = []
        labels = []
        if false_positives:
            data.append(false_positives)
            labels.append(f'False Positives\n(n={len(false_positives)})')
        if false_negatives:
            data.append(false_negatives)
            labels.append(f'False Negatives\n(n={len(false_negatives)})')
        ax6.boxplot(data, labels=labels)
        ax6.set_ylabel('Qwen Total Reward')
        ax6.set_title('Misclassification Analysis')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No Misclassifications!', 
                ha='center', va='center', fontsize=14, color='green')
        ax6.set_title('Perfect Agreement')
    
    # 7. Threshold analysis
    ax7 = fig.add_subplot(gs[2, 2])
    sorted_rewards = sorted(qwen_rewards)
    # Calculate accuracy at different thresholds
    thresholds = np.linspace(min(sorted_rewards), max(sorted_rewards), 50)
    accuracies = []
    for thresh in thresholds:
        correct = sum(1 for r in results if (r.qwen_reward >= thresh) == r.human_label)
        accuracies.append(correct / len(results))
    
    ax7.plot(thresholds, accuracies, 'b-', linewidth=2)
    # Mark current threshold
    current_thresh = np.percentile(qwen_rewards, 70)  # Top 30%
    current_acc = accuracies[np.argmin(np.abs(thresholds - current_thresh))]
    ax7.axvline(current_thresh, color='r', linestyle='--', label=f'Current (30% threshold)')
    ax7.scatter([current_thresh], [current_acc], color='r', s=100, zorder=5)
    ax7.set_xlabel('Reward Threshold')
    ax7.set_ylabel('Accuracy')
    ax7.set_title('Accuracy vs Threshold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Calculate metrics for stats panels
    total = len(results)
    tp = sum(1 for r in results if r.human_label and r.qwen_label)
    fp = sum(1 for r in results if not r.human_label and r.qwen_label)
    tn = sum(1 for r in results if not r.human_label and not r.qwen_label)
    fn = sum(1 for r in results if r.human_label and not r.qwen_label)
    
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 8. Combined statistics panel (bottom, single centered block)
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('off')
    
    # Create side-by-side text using proper formatting
    combined_text = (
        f"{'Alignment Metrics':^40}{'Reward Statistics':^40}\n"
        f"{'─' * 17:^40}{'─' * 17:^40}\n"
        f"{'Accuracy: ' + f'{accuracy:.1%}':^40}{'Good Episodes:':^40}\n"
        f"{'Precision: ' + f'{precision:.1%}':^40}{'  Mean: ' + f'{np.mean(good_rewards):.1f}':^40}\n"
        f"{'Recall: ' + f'{recall:.1%}':^40}{'  Std:  ' + f'{np.std(good_rewards):.1f}':^40}\n"
        f"{'F1 Score: ' + f'{f1:.3f}':^40}{' ':^40}\n"
        f"{' ':^40}{'Bad Episodes:':^40}\n"
        f"{'Episodes: ' + f'{total}':^40}{'  Mean: ' + f'{np.mean(bad_rewards):.1f}':^40}\n"
        f"{'Good: ' + f'{sum(1 for r in results if r.human_label)}':^40}{'  Std:  ' + f'{np.std(bad_rewards):.1f}':^40}\n"
        f"{'Bad: ' + f'{sum(1 for r in results if not r.human_label)}':^40}{' ':^40}\n"
        f"{' ':^40}{'Separation: ' + f'{np.mean(good_rewards) - np.mean(bad_rewards):.1f}':^40}"
    )
    
    ax8.text(0.5, 0.5, combined_text, fontsize=11, verticalalignment='center', horizontalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.savefig(output_dir / 'alignment_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to: {output_dir / 'alignment_analysis.png'}")


def check_and_setup_dependencies():
    """Check if Qwen dependencies are installed, run setup if needed."""
    import subprocess
    
    try:
        import transformers
        version = transformers.__version__
        
        # Check if transformers is at least version 5.0
        if version.startswith('5.') or version.startswith('6.'):
            print(f"✓ Dependencies OK (transformers {version})")
            return True
        else:
            print(f"⚠️  Old transformers version detected: {version}")
            print("   Need version 5.0+ for Qwen3-VL support")
    except ImportError:
        print("⚠️  Transformers not found")
    
    # Need to run setup
    print("\n" + "="*80)
    print("SETTING UP QWEN DEPENDENCIES")
    print("="*80)
    print("Running setup script to install required dependencies...")
    print("This will take a few minutes on first run.")
    print()
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent.parent
    setup_script = project_root / "scripts" / "setup_qwen_reward.sh"
    
    if not setup_script.exists():
        print(f"ERROR: Setup script not found: {setup_script}")
        print("Please run manually:")
        print("  cd /path/to/project && ./scripts/setup_qwen_reward.sh")
        sys.exit(1)
    
    # Run setup script
    try:
        result = subprocess.run(
            ["bash", str(setup_script)],
            cwd=str(project_root),
            check=True,
            capture_output=False,  # Show output directly
        )
        print("\n✓ Setup complete!")
        print("="*80)
        print()
        
        # Restart script to use new dependencies
        print("🔄 Restarting script with updated dependencies...")
        os.execv(sys.executable, [sys.executable] + sys.argv)
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Setup failed: {e}")
        print("Please run manually:")
        print(f"  cd {project_root} && ./scripts/setup_qwen_reward.sh")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Analyze alignment between human and Qwen reward labels")
    
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing HDF5 episode files")
    parser.add_argument("--task_instruction", type=str, required=True,
                       help="Task instruction for Qwen model")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to Qwen checkpoint")
    parser.add_argument("--output_dir", type=str, default="results/alignment_analysis",
                       help="Output directory for results")
    parser.add_argument("--max_frames", type=int, default=30,
                       help="Max frames to sample per episode")
    parser.add_argument("--image_rotation", type=int, default=0,
                       help="Image rotation angle")
    parser.add_argument("--inference_batch_size", type=int, default=30,
                       help="Batch size for inference")
    parser.add_argument("--advantage_threshold", type=float, default=0.3,
                       help="Percentile threshold for Qwen labeling (0.3 = top 30%%)")
    parser.add_argument("--ranking_frames", type=int, default=5,
                       help="Number of frames from the end to use for ranking (default: 5, use 0 for all frames)")
    parser.add_argument("--save_videos", action="store_true",
                       help="Convert HDF5 files to MP4 videos")
    parser.add_argument("--video_fps", type=int, default=30,
                       help="FPS for output videos")
    parser.add_argument("--skip_setup_check", action="store_true",
                       help="Skip automatic dependency setup check")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU device ID to use (default: 0)")
    
    args = parser.parse_args()
    
    # Set GPU device BEFORE any CUDA operations or imports
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print(f"Set CUDA_VISIBLE_DEVICES={args.gpu}")
    
    # NOW import the Qwen utilities (which import torch/CUDA)
    try:
        from qwen_reward_labeling import reward_inference_qwen, load_model_and_tokenizer
    except ImportError as e:
        print(f"Error importing qwen_reward_labeling: {e}")
        print(f"sys.path: {sys.path}")
        print("\nMake sure you're running with the correct Python environment:")
        print("  Use: python examples/h1_control_client/utils/analyze_reward_alignment.py")
        print("  NOT: uv run python ... (will use wrong environment)")
        sys.exit(1)
    
    # Check and setup dependencies first (unless skipped)
    if not args.skip_setup_check:
        check_and_setup_dependencies()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse dataset name and epoch from data directory
    dataset_name, epoch = parse_dataset_info(args.data_dir)
    
    print("="*80)
    print("REWARD ALIGNMENT ANALYSIS")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    if dataset_name:
        print(f"Dataset: {dataset_name}")
    if epoch:
        print(f"Epoch: {epoch}")
    print(f"Task: {args.task_instruction}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Step 1: Load human labels
    print("Step 1/4: Loading human labels...")
    human_labels = load_human_labels(args.data_dir)
    
    if not human_labels:
        print("Error: No human-labeled episodes found!")
        print("Make sure HDF5 files have 'advantage' attribute set.")
        return
    
    good_count = sum(1 for adv, _ in human_labels.values() if adv)
    bad_count = len(human_labels) - good_count
    print(f"  Found {len(human_labels)} labeled episodes:")
    print(f"    Good (advantage=True):  {good_count}")
    print(f"    Bad (advantage=False):  {bad_count}")
    print()
    
    # Step 2: Load Qwen model
    print("Step 2/4: Loading Qwen model...")
    print(f"  Using GPU: {args.gpu} (visible as cuda:0)")
    
    model, tokenizer = load_model_and_tokenizer(args.checkpoint_path, dtype="bf16")
    print(f"  Model loaded on device: {model.device}")
    print()
    
    # Step 3: Run inference
    print("Step 3/4: Running Qwen inference...")
    qwen_results = run_qwen_inference(
        args.data_dir,
        model,
        tokenizer,
        args.task_instruction,
        reward_inference_qwen,
        args.max_frames,
        args.image_rotation,
        args.inference_batch_size,
    )
    print()
    
    # Step 4: Compare and analyze
    print("Step 4/4: Analyzing alignment...")
    
    # Combine results - preserve the natural sort order for plotting
    combined_results = []
    filenames_sorted = sorted(human_labels.keys(), key=lambda f: natural_sort_key(Path(f)))
    
    for filename in filenames_sorted:
        human_adv, ep_len = human_labels[filename]
        if filename in qwen_results and 'error' not in qwen_results[filename]:
            qwen_data = qwen_results[filename]
            frame_rewards = qwen_data.get('reward_pred', [])
            total_reward = qwen_data['corrected_total_reward']
            
            # Calculate ranking reward based on last N frames
            if args.ranking_frames > 0 and len(frame_rewards) > 0:
                # Use sum of last N frames for ranking
                last_n_frames = frame_rewards[-args.ranking_frames:]
                ranking_reward = sum(last_n_frames)
            else:
                # Use total reward (all frames)
                ranking_reward = total_reward
            
            combined_results.append(EpisodeResult(
                filename=filename,
                human_label=human_adv,
                qwen_reward=total_reward,
                frame_rewards=frame_rewards,
                episode_length=ep_len,
                ranking_reward=ranking_reward,
            ))
    
    if not combined_results:
        print("Error: No valid results to compare!")
        return
    
    # Determine Qwen labels based on threshold using ranking_reward
    sorted_by_reward = sorted(combined_results, key=lambda x: x.ranking_reward, reverse=True)
    num_good = max(1, int(len(sorted_by_reward) * args.advantage_threshold))
    
    for i, result in enumerate(sorted_by_reward):
        result.qwen_label = (i < num_good)
    
    # Calculate metrics
    metrics = calculate_alignment_metrics(combined_results)
    
    # Print results
    print("\n" + "="*80)
    print("ALIGNMENT METRICS")
    print("="*80)
    print(f"Total episodes: {metrics['total_episodes']}")
    if args.ranking_frames > 0:
        print(f"Ranking method: Sum of last {args.ranking_frames} frames")
    else:
        print(f"Ranking method: Sum of all {args.max_frames} frames")
    print()
    print("Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  True Positives:  {cm['true_positive']:3d}  (Human=Good, Qwen=Good)")
    print(f"  False Positives: {cm['false_positive']:3d}  (Human=Bad,  Qwen=Good)")
    print(f"  True Negatives:  {cm['true_negative']:3d}  (Human=Bad,  Qwen=Bad)")
    print(f"  False Negatives: {cm['false_negative']:3d}  (Human=Good, Qwen=Bad)")
    print()
    print(f"Accuracy:  {metrics['accuracy']:.1%}")
    print(f"Precision: {metrics['precision']:.1%}")
    print(f"Recall:    {metrics['recall']:.1%}")
    print(f"F1 Score:  {metrics['f1_score']:.3f}")
    print()
    print("Reward Statistics:")
    good_stats = metrics['reward_statistics']['good_episodes']
    bad_stats = metrics['reward_statistics']['bad_episodes']
    print(f"  Good episodes (n={good_stats['count']}): {good_stats['mean']:.1f} ± {good_stats['std']:.1f}")
    print(f"  Bad episodes  (n={bad_stats['count']}):  {bad_stats['mean']:.1f} ± {bad_stats['std']:.1f}")
    print(f"  Separation: {metrics['reward_statistics']['separation']:.1f}")
    print("="*80)
    
    # Save results
    results_json = {
        'metrics': metrics,
        'episodes': [r.to_dict() for r in combined_results],
        'config': {
            'data_dir': args.data_dir,
            'task_instruction': args.task_instruction,
            'checkpoint_path': args.checkpoint_path,
            'advantage_threshold': args.advantage_threshold,
            'ranking_frames': args.ranking_frames,
        }
    }
    
    json_path = output_dir / 'alignment_results.json'
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved results to: {json_path}")
    
    # Generate visualizations
    plot_alignment_analysis(combined_results, output_dir, args.checkpoint_path, dataset_name, epoch)
    
    # Convert to videos if requested
    if args.save_videos:
        print("\nConverting HDF5 files to MP4 videos...")
        video_dir = output_dir / 'videos'
        video_dir.mkdir(exist_ok=True)
        
        for result in tqdm(combined_results, desc="Converting to video"):
            hdf5_path = Path(args.data_dir) / result.filename
            human_label_str = "good" if result.human_label else "bad"
            qwen_label_str = "good" if result.qwen_label else "bad"
            
            # Build video name with dataset and epoch info
            video_name_parts = []
            if dataset_name:
                video_name_parts.append(dataset_name)
            if epoch:
                video_name_parts.append(f"epoch{epoch}")
            video_name_parts.append(result.filename.replace('.hdf5', ''))
            video_name_parts.append(f"human_{human_label_str}")
            video_name_parts.append(f"qwen_{qwen_label_str}")
            
            video_name = "_".join(video_name_parts) + ".mp4"
            video_path = video_dir / video_name
            
            try:
                codec = hdf5_to_mp4(str(hdf5_path), str(video_path), args.video_fps)
                #print(f"  Saved: {video_name} ({codec})")
            except Exception as e:
                print(f"  Error converting {result.filename}: {e}")
        
        print(f"Videos saved to: {video_dir}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

