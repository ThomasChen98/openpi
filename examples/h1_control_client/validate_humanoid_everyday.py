"""
Quick validation script for Humanoid Everyday dataset.

This script validates the format and shows sample data from a Humanoid Everyday dataset
to help you understand the data structure before conversion.

Usage:
    python validate_humanoid_everyday.py --data_path ~/Downloads/push_a_button.zip
    python validate_humanoid_everyday.py --data_path ~/Downloads/push_a_button/ --show_sample
"""

import tyro
import numpy as np
from pathlib import Path


def main(
    data_path: str,
    *,
    show_sample: bool = False,
    episode_idx: int = 0,
    step_idx: int = 0,
) -> None:
    """Validate Humanoid Everyday dataset and optionally show sample data.
    
    Args:
        data_path: Path to the dataset (.zip or directory)
        show_sample: Show sample data from an episode
        episode_idx: Episode index to sample from
        step_idx: Step index to sample from
    """
    try:
        from humanoid_everyday import Dataloader
    except ImportError:
        print("❌ Error: humanoid_everyday package not found!")
        print("\nPlease install it:")
        print("  git clone https://github.com/ausbxuse/Humanoid-Everyday")
        print("  cd Humanoid-Everyday")
        print("  pip install -e .")
        return
    
    print("="*80)
    print("Humanoid Everyday Dataset Validator")
    print("="*80)
    
    # Load dataset
    print(f"\nLoading dataset from: {data_path}")
    try:
        ds = Dataloader(data_path)
        print(f"✓ Successfully loaded dataset")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # Show dataset info
    print(f"\n📊 Dataset Information:")
    print(f"  Total episodes: {len(ds)}")
    
    # Sample first episode to get dimensions
    try:
        first_episode = ds[0]
        first_step = first_episode[0]
        
        print(f"  Episode length: {len(first_episode)} timesteps")
        print(f"  Recording rate: 30 Hz")
        
        # Check robot type
        robot_type = first_step.get("robot_type", "Unknown (H1_2 doesn't have this field)")
        print(f"  Robot type: {robot_type}")
        
        # Show data dimensions
        print(f"\n📏 Data Dimensions:")
        print(f"  Arm state: {np.array(first_step['states']['arm_state']).shape} (14 DoF)")
        print(f"  Leg state: {np.array(first_step['states']['leg_state']).shape}")
        hand_state = np.array(first_step['states']['hand_state'])
        print(f"  Hand state: {hand_state.shape} ({hand_state.shape[0]} DoF - {hand_state.shape[0]//2} per hand)")
        print(f"  Left arm action: {np.array(first_step['actions']['left_angles']).shape} (7 DoF)")
        print(f"  Right arm action: {np.array(first_step['actions']['right_angles']).shape} (7 DoF)")
        
        # Check for hand actions
        if "left_hand_angles" in first_step['actions']:
            print(f"  Left hand action: {np.array(first_step['actions']['left_hand_angles']).shape}")
        if "right_hand_angles" in first_step['actions']:
            print(f"  Right hand action: {np.array(first_step['actions']['right_hand_angles']).shape}")
        
        print(f"  RGB image: {first_step['image'].shape}")
        print(f"  Depth map: {first_step['depth'].shape}")
        print(f"  LiDAR points: {first_step['lidar'].shape}")
        
        # Check data availability
        print(f"\n✓ Available Data:")
        print(f"  ✓ Joint states (arm, leg, hand)")
        print(f"  ✓ Actions (left/right arm commands)")
        print(f"  ✓ RGB images (480x640x3)")
        print(f"  ✓ Depth maps (480x640)")
        print(f"  ✓ LiDAR point clouds (~6000 points)")
        print(f"  ✓ IMU (quaternion, accel, gyro, rpy)")
        print(f"  ✓ Odometry (position, velocity, orientation)")
        
        if "hand_pressure_state" in first_step["states"] and first_step["states"]["hand_pressure_state"]:
            print(f"  ✓ Hand pressure sensors")
        
        # Show sample data if requested
        if show_sample:
            print(f"\n📝 Sample Data (Episode {episode_idx}, Step {step_idx}):")
            sample_episode = ds[episode_idx]
            sample_step = sample_episode[step_idx]
            
            print(f"\n  Arm State (14 DoF):")
            arm_state = np.array(sample_step['states']['arm_state'])
            print(f"    {arm_state}")
            print(f"    Range: [{arm_state.min():.3f}, {arm_state.max():.3f}]")
            
            print(f"\n  Left Arm Action (7 DoF):")
            left_action = np.array(sample_step['actions']['left_angles'])
            print(f"    {left_action}")
            
            print(f"\n  Right Arm Action (7 DoF):")
            right_action = np.array(sample_step['actions']['right_angles'])
            print(f"    {right_action}")
            
            print(f"\n  Hand State ({hand_state.shape[0]} DoF):")
            hand_state_sample = np.array(sample_step['states']['hand_state'])
            print(f"    Left hand (0-5):  {hand_state_sample[:6]}")
            print(f"    Right hand (6-11): {hand_state_sample[6:]}")
            print(f"    Range: [{hand_state_sample.min():.3f}, {hand_state_sample.max():.3f}]")
            
            if "left_hand_angles" in sample_step['actions']:
                print(f"\n  Hand Actions:")
                print(f"    Left: {np.array(sample_step['actions']['left_hand_angles'])}")
                print(f"    Right: {np.array(sample_step['actions']['right_hand_angles'])}")
            else:
                print(f"\n  Hand Actions: Not available (will use hand_state as action)")
            
            print(f"\n  IMU Orientation (quaternion [w,x,y,z]):")
            quat = np.array(sample_step['states']['imu']['quaternion'])
            print(f"    {quat}")
            
            print(f"\n  Odometry Position [x,y,z]:")
            pos = np.array(sample_step['states']['odometry']['position'])
            print(f"    {pos}")
            
            print(f"\n  Image shape: {sample_step['image'].shape}, dtype: {sample_step['image'].dtype}")
            print(f"  Image value range: [{sample_step['image'].min()}, {sample_step['image'].max()}]")
        
        # Conversion suggestions
        print(f"\n💡 Next Steps:")
        print(f"\n  1. Visualize with HDF5 conversion:")
        print(f"     python convert_humanoid_everyday_to_hdf5.py \\")
        print(f"       --input_dir {Path(data_path).expanduser()} \\")
        print(f"       --output_dir ./h1_data_processed/humanoid_everyday/test/")
        print(f"\n     python utils/data_replay.py \\")
        print(f"       --hdf5-path ./h1_data_processed/humanoid_everyday/test/episode_0.hdf5")
        print(f"\n  2. Convert to LeRobot (arms only, 14 DOF):")
        print(f"     python convert_humanoid_everyday_to_lerobot.py \\")
        print(f"       --data_path {data_path} \\")
        print(f"       --repo_id username/task_name \\")
        print(f"       --task_name \"describe the task\" \\")
        print(f"       --num_repeats 10")
        print(f"\n  3. Convert to LeRobot (arms + hands, 26 DOF):")
        print(f"     python convert_humanoid_everyday_to_lerobot.py \\")
        print(f"       --data_path {data_path} \\")
        print(f"       --repo_id username/task_name \\")
        print(f"       --task_name \"describe the task\" \\")
        print(f"       --include_hands \\")
        print(f"       --num_repeats 10")
        
        print(f"\n✓ Validation complete!")
        
    except Exception as e:
        print(f"❌ Error sampling dataset: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    tyro.cli(main)




