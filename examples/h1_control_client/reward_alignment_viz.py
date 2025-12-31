"""
Helper module to generate reward alignment visualization during integrated training.

This reuses logic from analyze_reward_alignment.py but works with pre-computed
reward labels rather than re-running inference.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class EpisodeComparison:
    """Comparison between human and Qwen labels for a single episode"""
    filename: str
    human_label: bool  # True=good, False=bad
    qwen_reward: float
    qwen_label: bool  # True=good, False=bad


def natural_sort_key(path):
    """Extract numeric parts from filename for natural sorting."""
    import re
    filename = path.name if hasattr(path, 'name') else str(path)
    numbers = re.findall(r'\d+', filename)
    return tuple(int(n) for n in numbers) if numbers else (0,)


def parse_dataset_info(data_dir: str):
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


def load_human_labels(data_dir: str) -> Dict[str, bool]:
    """Load human advantage labels from HDF5 files."""
    data_path = Path(data_dir)
    hdf5_files = sorted(data_path.glob("*.hdf5"), key=natural_sort_key)
    
    labels = {}
    for filepath in hdf5_files:
        try:
            with h5py.File(filepath, 'r') as f:
                if 'advantage' in f.attrs:
                    labels[filepath.name] = bool(f.attrs['advantage'])
        except Exception:
            pass
    
    return labels


def generate_alignment_visualization(
    data_dir: str,
    reward_labels: Dict[str, Dict[str, Any]],
    output_path: str,
    checkpoint_path: str = None,
    task_instruction: str = None,
):
    """
    Generate alignment visualization comparing human and Qwen labels.
    
    Args:
        data_dir: Directory containing HDF5 files with human labels
        reward_labels: Dictionary from label_episodes() with Qwen predictions
        output_path: Where to save the PNG
        checkpoint_path: Path to checkpoint (for title)
        task_instruction: Task description (for title)
    """
    # Parse dataset name and epoch from path
    dataset_name, epoch = parse_dataset_info(data_dir)
    
    # Load human labels
    human_labels = load_human_labels(data_dir)
    
    if not human_labels:
        print("Warning: No human labels found, skipping alignment visualization")
        return
    
    # Build comparison list (in natural sort order)
    comparisons = []
    filenames_sorted = sorted(human_labels.keys(), key=lambda f: natural_sort_key(Path(f)))
    
    for filename in filenames_sorted:
        if filename not in reward_labels or 'error' in reward_labels[filename]:
            continue
        
        human_adv = human_labels[filename]
        qwen_data = reward_labels[filename]
        qwen_adv = qwen_data.get('advantage', False)
        qwen_reward = qwen_data.get('corrected_total_reward', 0)
        
        comparisons.append(EpisodeComparison(
            filename=filename,
            human_label=human_adv,
            qwen_reward=qwen_reward,
            qwen_label=qwen_adv,
        ))
    
    if not comparisons:
        print("Warning: No valid comparisons found, skipping visualization")
        return
    
    # Calculate metrics
    tp = sum(1 for c in comparisons if c.human_label and c.qwen_label)
    fp = sum(1 for c in comparisons if not c.human_label and c.qwen_label)
    tn = sum(1 for c in comparisons if not c.human_label and not c.qwen_label)
    fn = sum(1 for c in comparisons if c.human_label and not c.qwen_label)
    
    total = len(comparisons)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    good_rewards = [c.qwen_reward for c in comparisons if c.human_label]
    bad_rewards = [c.qwen_reward for c in comparisons if not c.human_label]
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    
    # Add title with dataset, epoch, checkpoint, and task info
    title_parts = ["Reward Alignment Analysis"]
    if dataset_name:
        title_parts.append(f"Dataset: {dataset_name}")
    if epoch:
        title_parts.append(f"Epoch: {epoch}")
    if checkpoint_path:
        checkpoint_name = Path(checkpoint_path).name
        title_parts.append(f"Checkpoint: {checkpoint_name}")
    if task_instruction:
        title_parts.append(f"Task: {task_instruction}")
    
    fig.suptitle(" | ".join(title_parts), fontsize=14, fontweight='bold', y=0.995)
    
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3, top=0.96)
    
    # 1. Reward distribution by human label
    ax1 = fig.add_subplot(gs[0, 0])
    if good_rewards and bad_rewards:
        ax1.hist([good_rewards, bad_rewards], label=['Human: Good', 'Human: Bad'], 
                 bins=min(20, len(comparisons)//2), alpha=0.7, color=['green', 'red'])
    ax1.set_xlabel('Qwen Total Reward')
    ax1.set_ylabel('Count')
    ax1.set_title('Reward Distribution by Human Label')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    ax2 = fig.add_subplot(gs[0, 1])
    if good_rewards and bad_rewards:
        ax2.boxplot([good_rewards, bad_rewards], labels=['Human: Good', 'Human: Bad'])
    ax2.set_ylabel('Qwen Total Reward')
    ax2.set_title('Reward Distribution Comparison')
    ax2.grid(True, alpha=0.3)
    
    # 3. Confusion matrix
    ax3 = fig.add_subplot(gs[0, 2])
    confusion = np.array([[tn, fp], [fn, tp]])
    im = ax3.imshow(confusion, cmap='Blues', aspect='auto')
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['Qwen: Bad', 'Qwen: Good'])
    ax3.set_yticklabels(['Human: Bad', 'Human: Good'])
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(confusion[i, j]), ha='center', va='center',
                    color='white' if confusion[i, j] > confusion.max()/2 else 'black',
                    fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax3)
    ax3.set_title('Confusion Matrix')
    
    # 4. Scatter plot: rewards vs episode index
    ax4 = fig.add_subplot(gs[1, :])
    for i, comp in enumerate(comparisons):
        color = 'green' if comp.human_label else 'red'
        marker = 'o' if comp.qwen_label else 'x'
        ax4.scatter(i, comp.qwen_reward, c=color, marker=marker, s=100, alpha=0.7)
    ax4.set_xlabel('Episode Index (Natural Order)')
    ax4.set_ylabel('Qwen Total Reward')
    ax4.set_title('Reward by Episode (Color=Human Label, Marker=Qwen Label)\n'
                  'Green=Good, Red=Bad | Circle=Qwen Good, X=Qwen Bad')
    ax4.grid(True, alpha=0.3)
    
    # 5. Metrics summary
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis('off')
    metrics_text = f"""
    Alignment Metrics
    ─────────────────
    Accuracy:  {accuracy:.1%}
    Precision: {precision:.1%}
    Recall:    {recall:.1%}
    F1 Score:  {f1:.3f}
    
    Episodes:  {total}
    Good:      {sum(1 for c in comparisons if c.human_label)}
    Bad:       {sum(1 for c in comparisons if not c.human_label)}
    """
    ax5.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 6. Agreement analysis
    ax6 = fig.add_subplot(gs[2, 1])
    agree = sum(1 for c in comparisons if c.human_label == c.qwen_label)
    disagree = total - agree
    ax6.pie([agree, disagree], labels=['Agreement', 'Disagreement'], 
            autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    ax6.set_title(f'Overall Agreement\n({agree}/{total} episodes)')
    
    # 7. Reward statistics
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    stats_text = f"""
    Reward Statistics
    ─────────────────
    Good Episodes:
      Mean: {np.mean(good_rewards):.1f}
      Std:  {np.std(good_rewards):.1f}
    
    Bad Episodes:
      Mean: {np.mean(bad_rewards):.1f}
      Std:  {np.std(bad_rewards):.1f}
    
    Separation: {np.mean(good_rewards) - np.mean(bad_rewards):.1f}
    """
    ax7.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved alignment visualization: {output_path}")
    print(f"  Accuracy: {accuracy:.1%} | Precision: {precision:.1%} | Recall: {recall:.1%} | F1: {f1:.3f}")

