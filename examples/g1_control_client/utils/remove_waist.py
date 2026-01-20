"""
HDF5 Data Transform Script - Zero Waist Yaw Dimension

Transforms 29 DOF datasets by setting the waist yaw dimension to zero.
This is useful for training policies without waist control while keeping
the same data structure (29 DOF).

Data format (unchanged - stays 29 DOF):
  [0:7] left arm, [7:14] right arm, [14:21] left hand, [21:28] right hand, [28] waist yaw (set to 0)

Usage:
    python remove_waist.py --dataset insert_plate_jan16
    python remove_waist.py --dataset insert_plate_jan16 --data-dir /path/to/data

This will:
1. Read HDF5 files from g1_data_processed/<dataset>/
2. Set waist yaw dimension (index 28) to 0 in qpos and action
3. Save 29 DOF files (with zeroed waist) to g1_data_processed/<dataset>_zero_waist/
4. Then run convert_g1_data.sh to convert the modified data to LeRobot format
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


def zero_waist_in_episode(input_path: Path, output_path: Path) -> dict:
    """Zero out waist yaw dimension in a single episode HDF5 file.
    
    Args:
        input_path: Path to input HDF5 file (29 DOF)
        output_path: Path to output HDF5 file (29 DOF with waist zeroed)
        
    Returns:
        Dictionary with stats about the conversion
    """
    stats = {
        'action_dim': 0,
        'waist_yaw_range': (0.0, 0.0),
        'num_frames': 0,
    }
    
    with h5py.File(input_path, 'r') as f_in:
        # Load data
        actions = f_in['/action'][:]
        qpos = f_in['/observations/qpos'][:]
        
        num_frames = len(actions)
        stats['num_frames'] = num_frames
        stats['action_dim'] = actions.shape[1]
        
        # Check if this is 29 DOF data
        if actions.shape[1] != 29 or qpos.shape[1] != 29:
            raise ValueError(
                f"Expected 29 DOF data, got actions: {actions.shape[1]}, qpos: {qpos.shape[1]}. "
                f"This script is for zeroing waist in 29 DOF data."
            )
        
        # Record waist yaw stats before zeroing
        waist_yaw_actions = actions[:, 28]
        waist_yaw_qpos = qpos[:, 28]
        stats['waist_yaw_range'] = (
            float(min(waist_yaw_actions.min(), waist_yaw_qpos.min())),
            float(max(waist_yaw_actions.max(), waist_yaw_qpos.max()))
        )
        
        # Copy and zero waist yaw dimension (index 28)
        actions_zeroed = actions.copy()
        qpos_zeroed = qpos.copy()
        actions_zeroed[:, 28] = 0.0
        qpos_zeroed[:, 28] = 0.0
        
        # Load qvel if exists and zero waist there too
        qvel_zeroed = None
        if '/observations/qvel' in f_in:
            qvel = f_in['/observations/qvel'][:]
            qvel_zeroed = qvel.copy()
            if qvel.shape[1] == 29:
                qvel_zeroed[:, 28] = 0.0
        
        # Load locomotion data if it exists (pass through unchanged)
        loco_state = None
        loco_action = None
        if '/observations/loco_state' in f_in:
            loco_state = f_in['/observations/loco_state'][:]
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
    
    # Write modified data to output file
    with h5py.File(output_path, 'w') as f_out:
        # Save 29 DOF qpos and actions (with zeroed waist)
        obs_group = f_out.create_group('observations')
        obs_group.create_dataset('qpos', data=qpos_zeroed, compression='gzip')
        f_out.create_dataset('action', data=actions_zeroed, compression='gzip')
        
        # Save qvel if it exists
        if qvel_zeroed is not None:
            obs_group.create_dataset('qvel', data=qvel_zeroed, compression='gzip')
        
        # Save locomotion data if it exists (unchanged)
        if loco_state is not None:
            obs_group.create_dataset('loco_state', data=loco_state, compression='gzip')
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
        
        # Update attributes
        f_out.attrs['episode_length'] = num_frames
        f_out.attrs['num_frames'] = num_frames
        f_out.attrs['data_transform'] = 'waist_zeroed'
        f_out.attrs['action_dim'] = 29  # Still 29 DOF
    
    return stats


def zero_waist_in_dataset(dataset_name: str, data_dir: Path = None) -> None:
    """Zero waist dimension in all episodes of a dataset.
    
    Args:
        dataset_name: Name of the dataset folder
        data_dir: Base directory containing datasets (default: g1_data_processed)
    """
    if data_dir is None:
        # Default to g1_data_processed in the same directory as this script
        script_dir = Path(__file__).parent.parent
        data_dir = script_dir / "g1_data_processed"
    
    input_dir = data_dir / dataset_name
    output_dir = data_dir / f"{dataset_name}_zero_waist"
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {input_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=" * 60)
    print(f"Zero Waist Yaw Dimension (keeps 29 DOF, sets waist to 0)")
    print(f"=" * 60)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Find all HDF5 files
    hdf5_files = sorted(input_dir.glob("*.hdf5"))
    
    if not hdf5_files:
        print(f"No HDF5 files found in {input_dir}")
        return
    
    print(f"Found {len(hdf5_files)} HDF5 files to process")
    print()
    
    # Track overall stats
    all_waist_min = float('inf')
    all_waist_max = float('-inf')
    total_frames = 0
    
    # Process each file
    for input_path in tqdm(hdf5_files, desc="Processing episodes", unit="file"):
        output_path = output_dir / input_path.name
        
        try:
            stats = zero_waist_in_episode(input_path, output_path)
            total_frames += stats['num_frames']
            all_waist_min = min(all_waist_min, stats['waist_yaw_range'][0])
            all_waist_max = max(all_waist_max, stats['waist_yaw_range'][1])
        except Exception as e:
            print(f"\nError processing {input_path.name}: {e}")
            continue
    
    print()
    print(f"=" * 60)
    print(f"Processing complete!")
    print(f"=" * 60)
    print(f"Total frames processed: {total_frames}")
    print(f"Waist yaw range zeroed: [{all_waist_min:.4f}, {all_waist_max:.4f}] rad")
    print(f"                        [{np.degrees(all_waist_min):.2f}°, {np.degrees(all_waist_max):.2f}°]")
    print()
    print(f"Output saved to: {output_dir}")
    print()
    print(f"Next steps:")
    print(f"  1. Convert to LeRobot format:")
    print(f"     ./scripts/convert_g1_data.sh --task-name {dataset_name}_zero_waist")
    print()
    print(f"  2. Train:")
    print(f"     ./scripts/train_g1_local.sh --task-name {dataset_name}_zero_waist")


def main():
    parser = argparse.ArgumentParser(
        description="Zero waist yaw dimension in G1 HDF5 dataset (keeps 29 DOF, sets waist to 0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python remove_waist.py --dataset insert_plate_jan16
    
This will create a new folder named insert_plate_jan16_zero_waist
in the g1_data_processed directory with waist yaw set to 0 (still 29 DOF).

Then convert to LeRobot format and train:
    ./scripts/convert_g1_data.sh --task-name insert_plate_jan16_zero_waist
    ./scripts/train_g1_local.sh --task-name insert_plate_jan16_zero_waist
        """
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset folder in g1_data_processed/"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Base directory containing datasets (default: g1_data_processed)"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) if args.data_dir else None
    zero_waist_in_dataset(args.dataset, data_dir)


if __name__ == "__main__":
    main()
