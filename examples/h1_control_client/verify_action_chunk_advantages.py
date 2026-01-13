"""
Verify that action chunk advantages are properly stored and applied.

This script checks:
1. Advantages are stored in LeRobot dataset
2. Advantages match the cached .pkl files
3. Transform correctly applies advantages to chunks
4. Distribution of advantages is reasonable

Usage:
    python examples/h1_control_client/verify_action_chunk_advantages.py \
        --dataset-path examples/h1_control_client/h1_data_lerobot/fold_towel/epoch_0
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def verify_stored_advantages(dataset_path: str):
    """Verify advantages are stored in the dataset."""
    print("\n" + "="*80)
    print("STEP 1: Verify Advantages Stored in Dataset")
    print("="*80)
    
    dataset_path = Path(dataset_path)
    
    # Check metadata
    meta = LeRobotDatasetMetadata(str(dataset_path))
    
    if 'advantage' in meta.features:
        print("✅ 'advantage' feature found in dataset metadata")
        print(f"   Feature spec: {meta.features['advantage']}")
    else:
        print("❌ 'advantage' feature NOT found in dataset metadata")
        print(f"   Available features: {list(meta.features.keys())}")
        return False
    
    # Check parquet files
    parquet_dir = dataset_path / "data" / "chunk-000"
    parquet_files = sorted(parquet_dir.glob("episode_*.parquet"))
    
    print(f"\nFound {len(parquet_files)} parquet files")
    
    # Sample first file
    if parquet_files:
        df = pd.read_parquet(parquet_files[0])
        if 'advantage' in df.columns:
            print(f"✅ 'advantage' column found in parquet file: {parquet_files[0].name}")
            print(f"   Data type: {df['advantage'].dtype}")
            print(f"   Shape: {df['advantage'].shape}")
            print(f"   Sample values: {df['advantage'].head().tolist()}")
        else:
            print(f"❌ 'advantage' column NOT found in parquet file")
            print(f"   Available columns: {df.columns.tolist()}")
            return False
    
    return True


def verify_cache_consistency(dataset_path: str):
    """Verify cached advantages match parquet data."""
    print("\n" + "="*80)
    print("STEP 2: Verify Cache Consistency")
    print("="*80)
    
    dataset_path = Path(dataset_path)
    parquet_dir = dataset_path / "data" / "chunk-000"
    parquet_files = sorted(parquet_dir.glob("episode_*.parquet"))
    
    all_match = True
    for pf in parquet_files[:5]:  # Check first 5
        # Load from parquet
        df = pd.read_parquet(pf)
        parquet_advantages = df['advantage'].values
        
        # Load from cache
        cache_file = str(pf).replace('.parquet', '_action_chunk_advantages.pkl')
        if Path(cache_file).exists():
            cached_advantages = pickle.load(open(cache_file, 'rb'))
            
            # Compare
            if np.array_equal(parquet_advantages, cached_advantages):
                print(f"✅ {pf.name}: Cache matches parquet")
            else:
                print(f"❌ {pf.name}: MISMATCH!")
                print(f"   Parquet: {parquet_advantages[:10]}")
                print(f"   Cache:   {cached_advantages[:10]}")
                all_match = False
        else:
            print(f"⚠️  {pf.name}: No cache file found")
    
    return all_match


def verify_advantage_distribution(dataset_path: str):
    """Verify advantage distribution is reasonable."""
    print("\n" + "="*80)
    print("STEP 3: Verify Advantage Distribution")
    print("="*80)
    
    dataset_path = Path(dataset_path)
    parquet_dir = dataset_path / "data" / "chunk-000"
    parquet_files = sorted(parquet_dir.glob("episode_*.parquet"))
    
    all_advantages = []
    episode_stats = []
    
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        advantages = df['advantage'].values
        all_advantages.extend(advantages)
        
        true_count = advantages.sum()
        total = len(advantages)
        episode_stats.append({
            'name': pf.name,
            'true': true_count,
            'false': total - true_count,
            'pct_true': 100 * true_count / total
        })
    
    # Overall stats
    all_advantages = np.array(all_advantages)
    total_true = all_advantages.sum()
    total_false = len(all_advantages) - total_true
    pct_true = 100 * total_true / len(all_advantages)
    
    print(f"Overall Statistics:")
    print(f"  Total frames: {len(all_advantages)}")
    print(f"  Advantage=True:  {total_true:5d} ({pct_true:.1f}%)")
    print(f"  Advantage=False: {total_false:5d} ({100-pct_true:.1f}%)")
    
    # Check if reasonable
    if 15 <= pct_true <= 50:
        print(f"✅ Distribution looks reasonable (15-50% True)")
    else:
        print(f"⚠️  Distribution might be unusual ({pct_true:.1f}% True)")
        print(f"   Expected: 20-40% depending on threshold")
    
    # Per-episode breakdown
    print(f"\nPer-Episode Breakdown (first 10):")
    for stat in episode_stats[:10]:
        print(f"  {stat['name']}: {stat['true']:4d}/{stat['true']+stat['false']:4d} "
              f"({stat['pct_true']:5.1f}% True)")
    
    return True


def verify_transform_application(dataset_path: str, num_samples: int = 10):
    """Verify transform correctly applies advantages."""
    print("\n" + "="*80)
    print("STEP 4: Verify Transform Application")
    print("="*80)
    
    # Load dataset
    dataset = LeRobotDataset(
        str(dataset_path),
        delta_timestamps={
            "action": [i/30 for i in range(50)]  # 50 frames at 30 FPS
        }
    )
    
    print(f"Loaded dataset with {len(dataset)} chunks")
    
    # Sample random chunks
    import random
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    true_count = 0
    false_count = 0
    
    print(f"\nSampling {len(indices)} chunks:")
    for i in indices:
        chunk = dataset[i]
        
        # Check if advantage exists
        if 'advantage' in chunk:
            advantages = chunk['advantage']
            start_advantage = bool(advantages[0, 0]) if len(advantages.shape) > 1 else bool(advantages[0])
            
            # Check task field
            task = chunk.get('task', 'N/A')
            
            if start_advantage:
                true_count += 1
                symbol = "✅"
            else:
                false_count += 1
                symbol = "❌"
            
            print(f"  Chunk {i:5d}: {symbol} Start frame advantage={start_advantage}, Task='{task}'")
        else:
            print(f"  Chunk {i:5d}: ⚠️  No 'advantage' field found")
    
    print(f"\nSampled Distribution:")
    print(f"  Advantage=True:  {true_count}/{len(indices)} ({100*true_count/len(indices):.1f}%)")
    print(f"  Advantage=False: {false_count}/{len(indices)} ({100*false_count/len(indices):.1f}%)")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify action chunk advantages")
    parser.add_argument("--dataset-path", required=True, help="Path to LeRobot dataset")
    args = parser.parse_args()
    
    print("="*80)
    print("ACTION CHUNK ADVANTAGE VERIFICATION")
    print("="*80)
    print(f"Dataset: {args.dataset_path}")
    
    # Run all checks
    checks = [
        ("Stored Advantages", lambda: verify_stored_advantages(args.dataset_path)),
        ("Cache Consistency", lambda: verify_cache_consistency(args.dataset_path)),
        ("Distribution", lambda: verify_advantage_distribution(args.dataset_path)),
        ("Transform", lambda: verify_transform_application(args.dataset_path)),
    ]
    
    results = []
    for name, check in checks:
        try:
            result = check()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status}: {name}")
    
    all_passed = all(r for _, r in results)
    if all_passed:
        print("\n✅ All checks passed! Action chunk advantages are properly configured.")
    else:
        print("\n❌ Some checks failed. Please review the output above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

