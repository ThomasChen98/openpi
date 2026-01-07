#!/usr/bin/env python3
"""
Simple HDF5 dataset trimmer.
Trims datasets based on start and end frame indices.
"""

import h5py
import numpy as np
import tyro
from typing import Annotated
from pathlib import Path


def trim_hdf5_dataset(
    input_path: Annotated[str, tyro.conf.arg(aliases=["-i"], help="Path to input HDF5")],
    output_path: Annotated[str, tyro.conf.arg(aliases=["-o"], help="Path to output HDF5")],
    start_frame: Annotated[int, tyro.conf.arg(aliases=["-s"], help="0-based, inclusive")],
    end_frame: Annotated[int, tyro.conf.arg(aliases=["-e"], help="0-based, exclusive; -1 = all")],
) -> None:
    """Trim HDF5 dataset to specified frame range.
    
    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output trimmed HDF5 file
        start_frame: Starting frame index (0-based, inclusive)
        end_frame: Ending frame index (0-based, exclusive). Use -1 for all frames
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    print(f"Trimming dataset: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Frame range: {start_frame} to {end_frame if end_frame != -1 else 'end'}")
    
    with h5py.File(input_path, 'r') as f_in:
        # Get total number of frames from action dataset
        total_frames = f_in['/action'].shape[0]
        print(f"Total frames in dataset: {total_frames}")
        
        # Calculate actual end frame
        if end_frame == -1:
            end_frame = total_frames
        else:
            end_frame = min(end_frame, total_frames)
        
        # Validate frame range
        if start_frame < 0 or start_frame >= total_frames:
            raise ValueError(f"Invalid start_frame: {start_frame}")
        if end_frame <= start_frame:
            raise ValueError(f"Invalid end_frame: {end_frame}")
        
        num_frames = end_frame - start_frame
        print(f"Trimming to {num_frames} frames: {start_frame} to {end_frame-1}")
        
        # Create output file
        with h5py.File(output_path, 'w') as f_out:
            # Copy all datasets with frame trimming
            def copy_dataset(group_in, group_out, path=""):
                for key in group_in.keys():
                    item = group_in[key]
                    current_path = f"{path}/{key}" if path else key
                    
                    if isinstance(item, h5py.Group):
                        # Create group in output
                        new_group = group_out.create_group(key)
                        # Copy attributes
                        for attr_name, attr_value in item.attrs.items():
                            new_group.attrs[attr_name] = attr_value
                        # Recursively copy contents
                        copy_dataset(item, new_group, current_path)
                    else:
                        # Copy dataset with frame trimming
                        if len(item.shape) > 0 and item.shape[0] == total_frames:
                            # This is a time-series dataset, trim it
                            print(f"  Trimming {current_path}: {item.shape} -> ({num_frames}, {item.shape[1:]})")
                            trimmed_data = item[start_frame:end_frame]
                            f_out.create_dataset(current_path, data=trimmed_data, compression='gzip')
                        else:
                            # This is not a time-series dataset, copy as-is
                            print(f"  Copying {current_path}: {item.shape}")
                            f_out.create_dataset(current_path, data=item[:], compression='gzip')
                        
                        # Copy attributes
                        for attr_name, attr_value in item.attrs.items():
                            f_out[current_path].attrs[attr_name] = attr_value
            
            # Start copying from root
            copy_dataset(f_in, f_out)
    
    print(f"Successfully trimmed dataset to {num_frames} frames")
    print(f"Output saved to: {output_path}")


def main():
    """Main function with command line interface."""
    tyro.cli(trim_hdf5_dataset)


if __name__ == "__main__":
    main()
