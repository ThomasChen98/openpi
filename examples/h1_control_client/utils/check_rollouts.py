#!/usr/bin/env python3
"""
Quick script to check HDF5 episode files and count good/bad rollouts
"""

import h5py
import os
import glob
from pathlib import Path

def check_rollouts(directory_path):
    """
    Check all HDF5 files in directory and count good/bad based on advantage attribute
    
    Args:
        directory_path: Path to directory containing HDF5 episode files
    """
    directory_path = os.path.expanduser(directory_path)
    
    if not os.path.exists(directory_path):
        print(f"❌ Directory not found: {directory_path}")
        return
    
    # Find all HDF5 files
    hdf5_files = sorted(glob.glob(os.path.join(directory_path, "*.hdf5")))
    
    if not hdf5_files:
        print(f"❌ No HDF5 files found in: {directory_path}")
        return
    
    print(f"📁 Checking directory: {directory_path}")
    print(f"📊 Found {len(hdf5_files)} HDF5 files\n")
    
    good_count = 0
    bad_count = 0
    unlabeled_count = 0
    
    good_files = []
    bad_files = []
    unlabeled_files = []
    
    for filepath in hdf5_files:
        filename = os.path.basename(filepath)
        try:
            with h5py.File(filepath, 'r') as f:
                # Check if advantage attribute exists
                if 'advantage' in f.attrs:
                    advantage = f.attrs['advantage']
                    episode_length = f.attrs.get('episode_length', 'N/A')
                    
                    if advantage:
                        good_count += 1
                        good_files.append((filename, episode_length))
                        status = "✅ GOOD"
                    else:
                        bad_count += 1
                        bad_files.append((filename, episode_length))
                        status = "❌ BAD"
                else:
                    unlabeled_count += 1
                    unlabeled_files.append((filename, f.attrs.get('episode_length', 'N/A')))
                    status = "⚪ UNLABELED"
                
                print(f"{status} | {filename} | Length: {episode_length}")
        
        except Exception as e:
            print(f"⚠️  ERROR reading {filename}: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("📈 SUMMARY")
    print("="*60)
    print(f"Total episodes: {len(hdf5_files)}")
    print(f"✅ Good rollouts (advantage=True):  {good_count}")
    print(f"❌ Bad rollouts (advantage=False):  {bad_count}")
    if unlabeled_count > 0:
        print(f"⚪ Unlabeled rollouts:              {unlabeled_count}")
    
    if good_count + bad_count > 0:
        good_percentage = (good_count / (good_count + bad_count)) * 100
        print(f"\n📊 Success rate: {good_percentage:.1f}%")
    
    print("="*60)


if __name__ == "__main__":
    # Default directory
    default_dir = "/home/yuxin/Projects/openpi/examples/h1_control_client/h1_data_auto/fold_towel/epoch_0/raw"
    
    import sys
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = default_dir
    
    check_rollouts(directory)

