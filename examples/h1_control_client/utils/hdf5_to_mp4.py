#!/usr/bin/env python3
"""
Extract ego_cam videos from HDF5 datasets and save as MP4 files.
"""

import h5py
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import io
import tyro


def decode_jpeg_image(img_data: bytes) -> Image.Image:
    """Decode JPEG-encoded image data."""
    return Image.open(io.BytesIO(img_data))


def process_hdf5_to_mp4(
    hdf5_path: Path,
    output_path: Path,
    fps: int = 30,
    apply_transforms: bool = True
) -> None:
    """
    Extract ego_cam from HDF5 and save as MP4.
    
    Args:
        hdf5_path: Path to input HDF5 file
        output_path: Path to output MP4 file
        fps: Frames per second for output video
        apply_transforms: Whether to apply flip and rotation transformations
    """
    print(f"\nProcessing: {hdf5_path.name}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # Check if ego_cam exists
        ego_cam_path = '/observations/images/ego_cam'
        if ego_cam_path not in f:
            print(f"  Warning: ego_cam not found in {hdf5_path.name}")
            return
        
        dataset = f[ego_cam_path]
        num_frames = len(dataset)
        print(f"  Found {num_frames} frames")
        
        # Determine image format
        is_jpeg_encoded = len(dataset.shape) == 1
        
        # Get first frame to determine dimensions
        if is_jpeg_encoded:
            # JPEG encoded format
            first_img_data = dataset[0]
            first_img = decode_jpeg_image(first_img_data)
            first_array = np.array(first_img)
        else:
            # Direct array format (N, H, W, 3)
            first_array = dataset[0]
            if first_array.dtype != np.uint8:
                first_array = (first_array * 255).astype(np.uint8)
            # Convert BGR to RGB
            first_array = first_array[:, :, [2, 1, 0]]
        
        # Apply transformations to get final dimensions
        if apply_transforms:
            first_array = np.flipud(first_array)
            first_array = np.rot90(first_array, k=1)  # Counterclockwise
        
        height, width = first_array.shape[:2]
        print(f"  Video dimensions: {width}x{height}")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"  Error: Could not open video writer for {output_path}")
            return
        
        # Process all frames
        for i in tqdm(range(num_frames), desc=f"  Writing {output_path.name}", unit="frames"):
            if is_jpeg_encoded:
                # JPEG encoded
                img_data = dataset[i]
                img = decode_jpeg_image(img_data)
                img_array = np.array(img)
            else:
                # Direct array
                img_array = dataset[i]
                if img_array.dtype != np.uint8:
                    img_array = (img_array * 255).astype(np.uint8)
                # Convert BGR to RGB
                img_array = img_array[:, :, [2, 1, 0]]
            
            # Apply transformations (same as data_replay.py)
            if apply_transforms:
                img_array = np.flipud(img_array)
                img_array = np.rot90(img_array, k=1)  # Counterclockwise
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Write frame
            out.write(img_bgr)
        
        out.release()
        print(f"  ✓ Saved to: {output_path}")


def process_directory(
    input_dir: str,
    output_dir: str | None = None,
    fps: int = 30,
    apply_transforms: bool = True
) -> None:
    """
    Process all HDF5 files in a directory.
    
    Args:
        input_dir: Directory containing HDF5 files
        output_dir: Output directory for MP4 files (defaults to input_dir/videos)
        fps: Frames per second for output videos
        apply_transforms: Whether to apply flip and rotation transformations
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Directory not found: {input_dir}")
        return
    
    # Set output directory
    if output_dir is None:
        output_path = input_path / "videos"
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all HDF5 files
    hdf5_files = sorted(input_path.glob("*.hdf5"))
    
    if not hdf5_files:
        print(f"No HDF5 files found in {input_dir}")
        return
    
    print(f"Found {len(hdf5_files)} HDF5 files in {input_dir}")
    print(f"Output directory: {output_path}")
    
    # Process each file
    for hdf5_file in hdf5_files:
        mp4_file = output_path / f"{hdf5_file.stem}.mp4"
        process_hdf5_to_mp4(hdf5_file, mp4_file, fps, apply_transforms)
    
    print(f"\n✓ Completed! Processed {len(hdf5_files)} files")


def main(
    input_dirs: list[str],
    fps: int = 30,
    apply_transforms: bool = True
) -> None:
    """
    Extract ego_cam videos from HDF5 datasets.
    
    Args:
        input_dirs: List of directories containing HDF5 files
        fps: Frames per second for output videos
        apply_transforms: Whether to apply flip and rotation transformations (same as data_replay.py)
    
    Examples:
        # Process single directory
        python hdf5_to_mp4.py ../h1_data_processed/box_action/bad
        
        # Process multiple directories
        python hdf5_to_mp4.py ../h1_data_processed/box_action/bad ../h1_data_processed/box_action/good
    """
    for input_dir in input_dirs:
        process_directory(input_dir, fps=fps, apply_transforms=apply_transforms)


if __name__ == "__main__":
    tyro.cli(main)

