"""
Embodied Reward Labeling for H1 Robot Data

This script uses an embodied reward model (VLM-based) to automatically label
episodes with advantage labels (True/False) based on task completion quality.

It can be run standalone to test reward labeling effectiveness on raw hdf5 data,
and is also called by convert_h1_data_to_lerobot.py to label episodes during conversion.

Usage:
    # Test on a directory of HDF5 files
    uv run examples/h1_control_client/embodied_reward_labeling.py \\
        --data_dir examples/h1_control_client/h1_data_auto/lift_lid_dec7/epoch_5/raw \\
        --task_instruction "Lift the lid off the bowl" \\
        --advantage_threshold 0.3

    # Label episodes (called by convert script)
    python embodied_reward_labeling.py \\
        --data_dir /path/to/raw/hdf5s \\
        --task_instruction "task description" \\
        --advantage_threshold 0.3 \\
        --output_json labels.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import h5py
import numpy as np
from tqdm import tqdm

# Add the third_party directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "third_party" / "emboided_reward"))

try:
    from embodied_reward_util import (
        sample_even_frames,
        shuffle_frames,
        build_gemini_parts,
        build_openai_responses_input_from_gemini_parts,
        Frame,
        print_reward_labeling_header,
        print_percentile_calculation,
        print_episode_ranking,
        print_reward_summary,
        format_reward_progress,
        save_reward_visualization,
        export_reward_labels_csv
    )
    from openai import OpenAI
except ImportError as e:
    print(f"Error importing reward labeling utilities: {e}")
    print("Make sure the embodied_reward_util.py is in the correct location.")
    sys.exit(1)


def parse_reward_from_result(rollout_frames: List[Frame], raw_result: str) -> List[int]:
    """Parse reward predictions from VLM response."""
    import re
    
    arr_text = raw_result
    try:
        data = json.loads(arr_text)
    except json.JSONDecodeError:
        # Try to repair JSON
        repaired = re.sub(r",\s*([}\]])", r"\1", arr_text)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            # If still fails, return zeros
            print(f"Warning: Failed to parse reward result, using zeros")
            return [0] * len(rollout_frames)
    
    # Sort by shuffled order and extract percentages
    by_shuf = sorted(rollout_frames, key=lambda f: f.shuf_idx)
    reward_pred = []
    
    for i, item in enumerate(data):
        if i >= len(by_shuf):
            break
        percent = int(max(0, min(100, int(item.get("task_completion_percentage", 0)))))
        reward_pred.append(percent)
    
    # Ensure we have the right number of predictions
    while len(reward_pred) < len(rollout_frames):
        reward_pred.append(0)
    
    return reward_pred


def parse_reflection_reward_from_result(rollout_frames: List[Frame], raw_result: str) -> List[int]:
    """Parse corrected reward predictions from reflection VLM response."""
    import re
    
    arr_text = raw_result
    try:
        data = json.loads(arr_text)
    except json.JSONDecodeError:
        repaired = re.sub(r",\s*([}\]])", r"\1", arr_text)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse reflection result, using zeros")
            return [0] * len(rollout_frames)
    
    by_shuf = sorted(rollout_frames, key=lambda f: f.shuf_idx)
    reward_pred = []
    
    for i, item in enumerate(data):
        if i >= len(by_shuf):
            break
        percent = int(max(0, min(100, int(item.get("corrected_task_completion_percentage", 0)))))
        reward_pred.append(percent)
    
    while len(reward_pred) < len(rollout_frames):
        reward_pred.append(0)
    
    return reward_pred


def reward_inference(
    video_or_hdf5_path: str,
    task_instruction: str,
    max_frames: int = 30,
    image_rotation: int = 0,
    use_reflection: bool = True,
    openai_client: Optional[OpenAI] = None
) -> Dict[str, Any]:
    """
    Run reward inference on a video or HDF5 file.
    
    Args:
        video_or_hdf5_path: Path to video file (.mp4) or HDF5 file (.hdf5)
        task_instruction: Detailed task description for the VLM
        max_frames: Maximum number of frames to sample
        image_rotation: Rotation angle (0, 90, 180, 270)
        use_reflection: Whether to use reflection for correction
        openai_client: OpenAI client instance (if None, will create one)
    
    Returns:
        Dictionary with 'reward_pred', 'corrected_reward_pred', 'total_reward', 'corrected_total_reward'
    """
    if openai_client is None:
        # Try to get API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        openai_client = OpenAI(api_key=api_key)
    
    # Check if it's an HDF5 file
    if video_or_hdf5_path.endswith('.hdf5'):
        # Convert HDF5 to temporary video
        video_path = hdf5_to_video(video_or_hdf5_path)
    else:
        video_path = video_or_hdf5_path
    
    # Sample frames from video
    rollout_frames = sample_even_frames(video_path, max_frames=max_frames, rotate_angle=image_rotation)
    shuffle_frames(rollout_frames)
    
    # Build input for VLM
    gemini_input = build_gemini_parts(rollout_frames, task_instruction)
    openai_input = build_openai_responses_input_from_gemini_parts(gemini_input)
    
    # Get initial reward prediction
    try:
        resp = openai_client.responses.create(
            model="gpt-5.1",
            input=openai_input,
        )
        reward_pred_result_raw = resp.output_text
        reward_pred = parse_reward_from_result(rollout_frames, reward_pred_result_raw)
    except Exception as e:
        print(f"Error during initial reward inference: {e}")
        reward_pred = [0] * len(rollout_frames)
    
    # Get corrected reward with reflection
    corrected_reward_pred = reward_pred.copy()
    if use_reflection:
        # Re-sample frames without shuffling and add predicted percentages
        rollout_frames = sample_even_frames(video_path, max_frames=max_frames, rotate_angle=image_rotation)
        for i, frame in enumerate(rollout_frames):
            frame.predicted_percent = reward_pred[i]
        
        # Build reflection input
        from embodied_reward_util import build_gemini_parts_reflection
        gemini_input_reflection = build_gemini_parts_reflection(rollout_frames, task_instruction)
        openai_input_reflection = build_openai_responses_input_from_gemini_parts(gemini_input_reflection)
        
        # Get corrected prediction
        try:
            resp = openai_client.responses.create(
                model="gpt-5.1",
                input=openai_input_reflection,
            )
            reward_pred_result_raw = resp.output_text
            corrected_reward_pred = parse_reflection_reward_from_result(rollout_frames, reward_pred_result_raw)
        except Exception as e:
            print(f"Error during reflection: {e}")
            corrected_reward_pred = reward_pred.copy()
    
    # Clean up temporary video if created
    if video_or_hdf5_path.endswith('.hdf5') and os.path.exists(video_path):
        os.remove(video_path)
    
    return {
        'reward_pred': reward_pred,
        'corrected_reward_pred': corrected_reward_pred,
        'total_reward': sum(reward_pred),
        'corrected_total_reward': sum(corrected_reward_pred)
    }


def hdf5_to_video(hdf5_path: str, fps: int = 10) -> str:
    """
    Convert HDF5 episode to temporary video file for reward inference.
    
    Args:
        hdf5_path: Path to HDF5 file
        fps: Frame rate for video
    
    Returns:
        Path to temporary video file
    """
    import tempfile
    
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
            from embodied_reward_util import bgr_to_data_url
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


def label_episodes(
    data_dir: str,
    task_instruction: str,
    max_frames: int = 30,
    image_rotation: int = 0,
    advantage_threshold: float = 0.3,
    use_reflection: bool = True,
    max_workers: int = 64,
    openai_api_key: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Label all HDF5 episodes in a directory with advantage labels.
    
    Args:
        data_dir: Directory containing HDF5 files
        task_instruction: Task description for VLM
        max_frames: Maximum frames to sample per episode
        image_rotation: Rotation angle for images
        advantage_threshold: Percentile threshold for advantage labeling (0.0-1.0)
                           e.g., 0.3 means top 30% episodes get Advantage=True
        use_reflection: Whether to use reflection
        max_workers: Number of parallel workers
        openai_api_key: OpenAI API key (if not in environment)
    
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
    
    # Set up OpenAI client
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Process files in parallel - first get all rewards
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(
                reward_inference,
                str(hdf5_file),
                task_instruction,
                max_frames,
                image_rotation,
                use_reflection,
                client
            ): hdf5_file
            for hdf5_file in hdf5_files
        }
        
        for future in tqdm(as_completed(future_to_file), total=len(hdf5_files), desc="Computing rewards"):
            hdf5_file = future_to_file[future]
            try:
                result = future.result()
                
                # Store results without advantage label yet
                results[hdf5_file.name] = {
                    'advantage': None,  # Will be set after percentile calculation
                    'reward_pred': result['reward_pred'],
                    'corrected_reward_pred': result['corrected_reward_pred'],
                    'total_reward': result['total_reward'],
                    'corrected_total_reward': result['corrected_total_reward'],
                }
                
                print(format_reward_progress(hdf5_file.name, result['corrected_total_reward']))
                
            except Exception as e:
                print(f"Error processing {hdf5_file.name}: {e}")
                results[hdf5_file.name] = {
                    'advantage': False,
                    'error': str(e),
                    'corrected_total_reward': 0,
                }
    
    # Calculate percentile threshold for advantage labeling
    # Get all valid corrected rewards (excluding errors)
    valid_results = [(name, data) for name, data in results.items() 
                     if 'error' not in data and data['corrected_total_reward'] is not None]
    
    if not valid_results:
        print("\nNo valid results to calculate percentile")
        return results
    
    # Sort by corrected total reward (descending)
    sorted_results = sorted(valid_results, key=lambda x: x[1]['corrected_total_reward'], reverse=True)
    
    # Calculate cutoff index for top X%
    num_good = max(1, int(len(sorted_results) * advantage_threshold))
    cutoff_reward = sorted_results[num_good - 1][1]['corrected_total_reward'] if num_good > 0 else float('inf')
    
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
        description="Label H1 robot episodes with embodied reward model"
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
        "--use_reflection",
        action="store_true",
        default=True,
        help="Use reflection to correct predictions"
    )
    parser.add_argument(
        "--no_reflection",
        action="store_true",
        help="Disable reflection"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save results as JSON"
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key (if not in environment)"
    )
    
    args = parser.parse_args()
    
    # Check for API key
    if not args.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        print("Or pass it with --openai_api_key")
        sys.exit(1)
    
    # Label episodes
    use_reflection = args.use_reflection and not args.no_reflection
    
    results = label_episodes(
        data_dir=args.data_dir,
        task_instruction=args.task_instruction,
        max_frames=args.max_frames,
        image_rotation=args.image_rotation,
        advantage_threshold=args.advantage_threshold,
        use_reflection=use_reflection,
        max_workers=args.max_workers,
        openai_api_key=args.openai_api_key
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
