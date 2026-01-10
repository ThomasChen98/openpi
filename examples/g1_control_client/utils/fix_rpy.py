"""
HDF5 Data Fix Script for RPY (Roll, Pitch, Yaw)

Fixes datasets by overwriting the roll, pitch, yaw values in loco_state[1:4]
across all frames. These values correspond to indices [28:31] in the final
32-dim state space (qpos[28] + rpy[3] + yaw_rate[1]).

Usage:
    python fix_rpy.py --dataset approach_cone --roll 0.0 --pitch 0.0 --yaw 0.0

This will:
1. Read HDF5 files from g1_data_processed/<dataset>/
2. Set loco_state[1:4] (rpy) to the specified values for all frames
3. Save corrected files to g1_data_processed/<dataset>_corrected_rpy/
4. Then run convert_g1_data.sh to convert the corrected data to LeRobot format
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


def fix_rpy_in_episode(input_path: Path, output_path: Path, rpy_values: np.ndarray) -> None:
    """Fix RPY values in a single episode HDF5 file.
    
    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output HDF5 file
        rpy_values: Array of shape (3,) containing [roll, pitch, yaw] values in radians
    """
    with h5py.File(input_path, 'r') as f_in:
        # Load data
        actions = f_in['/action'][:]  # Shape: (N, 28)
        qpos = f_in['/observations/qpos'][:]  # Shape: (N, 28)
        
        # Load locomotion state
        if '/observations/loco_state' not in f_in:
            raise ValueError(f"No loco_state found in {input_path}")
        
        loco_state = f_in['/observations/loco_state'][:]  # Shape: (N, 17)
        
        num_frames = len(actions)
        
        # Check dimensions
        if actions.shape[1] != 28 or qpos.shape[1] != 28:
            raise ValueError(
                f"Expected 28 DoF data, got actions: {actions.shape[1]}, qpos: {qpos.shape[1]}"
            )
        
        if loco_state.shape[1] != 17:
            raise ValueError(
                f"Expected loco_state with 17 dims, got {loco_state.shape[1]}"
            )
        
        # Create corrected arrays (copy original data)
        corrected_actions = actions.copy()
        corrected_qpos = qpos.copy()
        corrected_loco_state = loco_state.copy()
        
        # Set RPY (indices 1:4 in loco_state) to specified values for all frames
        corrected_loco_state[:, 1:4] = rpy_values
        
        # Load locomotion action if it exists
        loco_action = None
        if '/loco_action' in f_in:
            loco_action = f_in['/loco_action'][:]
        
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
                    camera_data[topic] = dataset[:]
                else:
                    # JPEG encoded format: (N,)
                    image_formats[topic] = "jpeg"
                    camera_data[topic] = [dataset[i] for i in range(len(dataset))]
        
        # Load phase data if it exists
        phase_data = None
        if '/phase' in f_in:
            phase_data = f_in['/phase'][:]
        
        # Copy attributes
        attrs = dict(f_in.attrs)
    
    # Write corrected data to output file
    with h5py.File(output_path, 'w') as f_out:
        # Save corrected qpos, actions, and loco_state
        obs_group = f_out.create_group('observations')
        obs_group.create_dataset('qpos', data=corrected_qpos, compression='gzip')
        obs_group.create_dataset('loco_state', data=corrected_loco_state, compression='gzip')
        f_out.create_dataset('action', data=corrected_actions, compression='gzip')
        
        # Save locomotion action if it exists
        if loco_action is not None:
            f_out.create_dataset('loco_action', data=loco_action, compression='gzip')
        
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
        
        # Update frame count (same as original)
        f_out.attrs['episode_length'] = len(corrected_actions)
        if 'num_frames' in f_out.attrs:
            f_out.attrs['num_frames'] = len(corrected_actions)
        
        # Add a note about the fix
        f_out.attrs['data_fix_applied'] = f'rpy_fixed_to_{rpy_values[0]:.6f}_{rpy_values[1]:.6f}_{rpy_values[2]:.6f}'


def fix_dataset(dataset_name: str, rpy_values: np.ndarray, data_dir: Path = None) -> None:
    """Fix all episodes in a dataset.
    
    Args:
        dataset_name: Name of the dataset folder
        rpy_values: Array of shape (3,) containing [roll, pitch, yaw] values in radians
        data_dir: Base directory containing datasets (default: g1_data_processed)
    """
    if data_dir is None:
        # Default to g1_data_processed in the same directory as this script
        script_dir = Path(__file__).parent.parent
        data_dir = script_dir / "g1_data_processed"
    
    input_dir = data_dir / dataset_name
    output_dir = data_dir / f"{dataset_name}_corrected_rpy"
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {input_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"RPY values: roll={rpy_values[0]:.6f}, pitch={rpy_values[1]:.6f}, yaw={rpy_values[2]:.6f} (radians)")
    
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
            fix_rpy_in_episode(input_path, output_path, rpy_values)
        except Exception as e:
            print(f"\nError processing {input_path.name}: {e}")
            continue
    
    print(f"\nProcessing complete!")
    print(f"Corrected files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Fix RPY (roll, pitch, yaw) values in G1 HDF5 dataset files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python fix_rpy.py --dataset approach_cone --roll 0.0 --pitch 0.0 --yaw 0.0
    
This will create a new folder named approach_cone_corrected_rpy
in the g1_data_processed directory with RPY values set to the specified values.
Then run convert_g1_data.sh with --data-dir pointing to the corrected folder.

Note: RPY values are in radians. Common values:
    - Zero pose: roll=0.0, pitch=0.0, yaw=0.0
    - 90 degrees: ~1.5708 radians
        """
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset folder in g1_data_processed/"
    )
    parser.add_argument(
        "--roll",
        type=float,
        default=0.0,
        help="Roll value in radians (default: 0.0)"
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=0.0,
        help="Pitch value in radians (default: 0.0)"
    )
    parser.add_argument(
        "--yaw",
        type=float,
        default=0.0,
        help="Yaw value in radians (default: 0.0)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Base directory containing datasets (default: g1_data_processed)"
    )
    
    args = parser.parse_args()
    
    rpy_values = np.array([args.roll, args.pitch, args.yaw], dtype=np.float32)
    data_dir = Path(args.data_dir) if args.data_dir else None
    fix_dataset(args.dataset, rpy_values, data_dir)


if __name__ == "__main__":
    main()

