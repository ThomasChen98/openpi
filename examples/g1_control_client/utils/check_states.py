"""
Check States Script

Prints out the first step of a given HDF5 file to inspect the data structure.

Usage:
    python check_states.py <path_to_hdf5_file>
    
Example:
    python check_states.py ../g1_data_processed/insert_bottle_jan9/episode_05.hdf5
"""

import argparse
import h5py
import numpy as np
from pathlib import Path


def print_first_step(hdf5_path: Path) -> None:
    """Print the first step of an HDF5 episode file.
    
    Args:
        hdf5_path: Path to HDF5 file
    """
    if not hdf5_path.exists():
        raise FileNotFoundError(f"File not found: {hdf5_path}")
    
    print(f"\n{'='*80}")
    print(f"File: {hdf5_path}")
    print(f"{'='*80}\n")
    
    with h5py.File(hdf5_path, 'r') as f:
        # Print file attributes
        print("FILE ATTRIBUTES:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        print()
        
        # Print dataset structure
        print("DATASET STRUCTURE:")
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
        f.visititems(print_structure)
        print()
        
        # Print first step data
        print(f"{'='*80}")
        print("FIRST STEP DATA (index 0):")
        print(f"{'='*80}\n")
        
        # Actions
        if '/action' in f:
            actions = f['/action'][:]
            print(f"Actions (shape={actions.shape}):")
            print(f"  First step: {actions[0]}")
            print()
        
        # Observations - qpos
        if '/observations/qpos' in f:
            qpos = f['/observations/qpos'][:]
            print(f"Observations/qpos (shape={qpos.shape}):")
            print(f"  First step: {qpos[0]}")
            print()
        
        # Phase (if exists)
        if '/phase' in f:
            phase = f['/phase'][:]
            print(f"Phase (shape={phase.shape}):")
            print(f"  First step: {phase[0]}")
            print()
        
        # Camera topics
        if '/observations/images' in f:
            print("Camera Images:")
            for topic in f['/observations/images'].keys():
                dataset = f[f'/observations/images/{topic}']
                if len(dataset.shape) == 4:
                    # Direct array format: (N, H, W, 3)
                    print(f"  {topic}: shape={dataset.shape}, dtype={dataset.dtype}")
                    print(f"    First frame shape: {dataset[0].shape}")
                    print(f"    First frame min/max: {dataset[0].min()}/{dataset[0].max()}")
                else:
                    # JPEG encoded format: (N,)
                    print(f"  {topic}: shape={dataset.shape}, dtype={dataset.dtype} (JPEG encoded)")
                    print(f"    First frame size: {len(dataset[0])} bytes")
            print()
        
        # Print summary statistics
        print(f"{'='*80}")
        print("SUMMARY STATISTICS:")
        print(f"{'='*80}\n")
        
        if '/action' in f:
            actions = f['/action'][:]
            print(f"Actions:")
            print(f"  Number of steps: {len(actions)}")
            print(f"  Dimensions: {actions.shape[1]}")
            print(f"  Min: {actions.min()}")
            print(f"  Max: {actions.max()}")
            print(f"  Mean: {actions.mean():.2f}")
            print(f"  Std: {actions.std():.2f}")
            print()
        
        if '/observations/qpos' in f:
            qpos = f['/observations/qpos'][:]
            print(f"Qpos:")
            print(f"  Number of steps: {len(qpos)}")
            print(f"  Dimensions: {qpos.shape[1]}")
            print(f"  Min: {qpos.min()}")
            print(f"  Max: {qpos.max()}")
            print(f"  Mean: {qpos.mean():.2f}")
            print(f"  Std: {qpos.std():.2f}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Print the first step of an HDF5 episode file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python check_states.py ../g1_data_processed/insert_plate_jan16/episode_05.hdf5
        """
    )
    parser.add_argument(
        "hdf5_file",
        type=str,
        help="Path to HDF5 file"
    )
    
    args = parser.parse_args()
    
    hdf5_path = Path(args.hdf5_file)
    print_first_step(hdf5_path)


if __name__ == "__main__":
    main()

