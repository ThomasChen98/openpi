"""
HDF5 Data Fix Script for Hand Pose Correction

Fixes datasets where hand poses (last 12 DoF) in qpos are incorrectly saved as 0.
The corrected qpos for hands at time t should be the hand action from time t-1.

Usage:
    python data_fix.py --dataset fold_cloth_with_both_hands

This will:
1. Read HDF5 files from h1_data_processed/<dataset>/
2. Fix the hand qpos by using hand actions from previous timestep
3. Save corrected files to h1_data_processed/<dataset>_corrected/
4. Each corrected file will have 1 less frame than the original
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil


def fix_hand_qpos_in_episode(input_path: Path, output_path: Path) -> None:
    """Fix hand qpos in a single episode HDF5 file.
    
    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output HDF5 file
    """
    with h5py.File(input_path, 'r') as f_in:
        # Load data
        actions = f_in['/action'][:]  # Shape: (N, 26)
        qpos = f_in['/observations/qpos'][:]  # Shape: (N, 26) with last 12 as zeros
        
        num_frames = len(actions)
        
        # Check if this is 26 DoF data
        if actions.shape[1] != 26 or qpos.shape[1] != 26:
            raise ValueError(
                f"Expected 26 DoF data, got actions: {actions.shape[1]}, qpos: {qpos.shape[1]}"
            )
        
        # Create corrected qpos: qpos[t] = [arm_qpos[t], hand_action[t-1]]
        # We skip the first frame since it has no previous hand action
        corrected_qpos = np.zeros((num_frames - 1, 26), dtype=qpos.dtype)
        
        for t in range(1, num_frames):
            # Arm qpos (first 14 DoF) from current frame
            corrected_qpos[t-1, :14] = qpos[t, :14]
            # Hand qpos (last 12 DoF) from previous frame's hand action
            corrected_qpos[t-1, 14:] = actions[t-1, 14:]
        
        # Also trim actions to match (skip first frame)
        corrected_actions = actions[1:]
        
        # Load camera data
        camera_topics = []
        camera_data = {}
        image_formats = {}
        
        if '/observations/images' in f_in:
            for key in f_in['/observations/images'].keys():
                camera_topics.append(key)
                
            for topic in camera_topics:
                topic_path = f'/observations/images/{topic}'
                dataset = f_in[topic_path]
                
                # Check format
                if len(dataset.shape) == 4 and dataset.shape[-1] == 3:
                    # Direct array format: (N, H, W, 3)
                    image_formats[topic] = "array"
                    # Trim to match corrected length (skip first frame)
                    camera_data[topic] = dataset[1:]
                else:
                    # JPEG encoded format: (N,)
                    image_formats[topic] = "jpeg"
                    # Load and trim (skip first frame)
                    camera_data[topic] = [dataset[i] for i in range(1, len(dataset))]
        
        # Load phase data if it exists
        phase_data = None
        if '/phase' in f_in:
            phase_data = f_in['/phase'][1:]  # Skip first frame
        
        # Copy attributes
        attrs = dict(f_in.attrs)
    
    # Write corrected data to output file
    with h5py.File(output_path, 'w') as f_out:
        # Save corrected qpos and actions
        obs_group = f_out.create_group('observations')
        obs_group.create_dataset('qpos', data=corrected_qpos, compression='gzip')
        f_out.create_dataset('action', data=corrected_actions, compression='gzip')
        
        # Save camera data
        if camera_topics:
            images_group = obs_group.create_group('images')
            
            for topic in camera_topics:
                if image_formats[topic] == "array":
                    # Direct array format
                    images_group.create_dataset(
                        topic, 
                        data=camera_data[topic], 
                        compression='gzip'
                    )
                else:
                    # JPEG encoded format (variable length)
                    dt = h5py.vlen_dtype(np.dtype('uint8'))
                    images_group.create_dataset(
                        topic,
                        (len(camera_data[topic]),),
                        dtype=dt
                    )
                    for i, compressed_frame in enumerate(camera_data[topic]):
                        images_group[topic][i] = compressed_frame
        
        # Save phase data if it exists
        if phase_data is not None:
            f_out.create_dataset('phase', data=phase_data, compression='gzip')
        
        # Copy and update attributes
        for key, value in attrs.items():
            f_out.attrs[key] = value
        
        # Update frame count
        f_out.attrs['episode_length'] = len(corrected_actions)
        if 'num_frames' in f_out.attrs:
            f_out.attrs['num_frames'] = len(corrected_actions)
        
        # Add a note about the fix
        f_out.attrs['data_fix_applied'] = 'hand_qpos_corrected'


def fix_dataset(dataset_name: str, data_dir: Path = None) -> None:
    """Fix all episodes in a dataset.
    
    Args:
        dataset_name: Name of the dataset folder
        data_dir: Base directory containing datasets (default: h1_data_processed)
    """
    if data_dir is None:
        # Default to h1_data_processed in the same directory as this script
        script_dir = Path(__file__).parent.parent
        data_dir = script_dir / "h1_data_processed"
    
    input_dir = data_dir / dataset_name
    output_dir = data_dir / f"{dataset_name}_corrected"
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {input_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find all HDF5 files
    hdf5_files = sorted(input_dir.glob("*.hdf5"))
    
    if not hdf5_files:
        print(f"No HDF5 files found in {input_dir}")
        return
    
    print(f"Found {len(hdf5_files)} HDF5 files to process")
    
    # Process each file
    for input_path in tqdm(hdf5_files, desc="Processing episodes", unit="file"):
        output_path = output_dir / input_path.name
        
        try:
            fix_hand_qpos_in_episode(input_path, output_path)
        except Exception as e:
            print(f"\nError processing {input_path.name}: {e}")
            continue
    
    print(f"\nProcessing complete!")
    print(f"Corrected files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Fix hand qpos in HDF5 dataset files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python data_fix.py --dataset fold_cloth_with_both_hands
    
This will create a new folder named fold_cloth_with_both_hands_corrected
in the h1_data_processed directory.
        """
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset folder in h1_data_processed/"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Base directory containing datasets (default: h1_data_processed)"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) if args.data_dir else None
    fix_dataset(args.dataset, data_dir)


if __name__ == "__main__":
    main()

