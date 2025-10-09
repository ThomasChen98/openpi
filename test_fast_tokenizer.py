#!/usr/bin/env python3
"""
Test script to check if FASTTokenizer can handle the customized collected dataset.
Tests both with and without action normalization.
"""

import h5py
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

MAX_LEN = 512
# Add the src directory to the path so we can import openpi modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openpi.models.tokenizer import FASTTokenizer

def normalize_actions(actions, target_range=(-1, 1)):
    """Normalize actions to target range."""
    min_val, max_val = target_range
    actions_min = actions.min()
    actions_max = actions.max()
    
    # Avoid division by zero
    if actions_max == actions_min:
        return np.zeros_like(actions)
    
    # Normalize to [0, 1] first, then to target range
    normalized = (actions - actions_min) / (actions_max - actions_min)
    normalized = normalized * (max_val - min_val) + min_val
    
    return normalized

def test_dataset_tokenization_with_normalization():
    """Test if FASTTokenizer can tokenize the collected dataset with and without normalization."""
    
    # Load a sample from the dataset
    data_path = "./collected_data/bottle_ep_01.hdf5"
    
    print("Loading dataset...")
    with h5py.File(data_path, 'r') as f:
        # Get action data
        actions = f['action'][:]  # Shape: (2483, 51)
        qpos = f['observations/qpos'][:]  # Shape: (2483, 51)
        
        print(f"Dataset loaded:")
        print(f"  Actions shape: {actions.shape}")
        print(f"  QPOS shape: {qpos.shape}")
        print(f"  Action range: [{actions.min():.4f}, {actions.max():.4f}]")
        print(f"  QPOS range: [{qpos.min():.4f}, {qpos.max():.4f}]")
        
        # Take a small sample for testing
        sample_idx = 100  # Use a middle sample
        sample_actions = actions[sample_idx:sample_idx+1]  # Shape: (1, 51)
        sample_qpos = qpos[sample_idx:sample_idx+1]  # Shape: (1, 51)
        
        print(f"\nUsing sample at index {sample_idx}:")
        print(f"  Sample actions shape: {sample_actions.shape}")
        print(f"  Sample qpos shape: {sample_qpos.shape}")
    
    # Initialize FASTTokenizer with higher max_len to avoid truncation
    print("\nInitializing FASTTokenizer...")
    try:
        tokenizer = FASTTokenizer(max_len=MAX_LEN)  # Increased from 256
        print("✓ FASTTokenizer initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize FASTTokenizer: {e}")
        return False
    
    # Test tokenization without normalization
    print("\n" + "="*60)
    print("TESTING WITHOUT NORMALIZATION")
    print("="*60)
    
    prompt = "pick up the bottle"
    state = sample_qpos[0]  # (51,) - state should be 1D
    actions_reshaped = sample_actions.reshape(1, 51)  # (1, 51)
    
    try:
        tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(
            prompt, state, actions_reshaped
        )

        
        print(f"✓ Tokenization successful!")
        print(f"  Tokens shape: {tokens.shape}")
        print(f"  Token masks shape: {token_masks.shape}")
        print(f"  AR masks shape: {ar_masks.shape}")
        print(f"  Loss masks shape: {loss_masks.shape}")
        
        # Test action extraction
        print("\nTesting action extraction...")
        extracted_actions = tokenizer.extract_actions(tokens, action_horizon=1, action_dim=51)
        print(f"✓ Action extraction successful!")
        print(f"  Extracted actions shape: {extracted_actions.shape}")
        print(f"  Original actions shape: {actions_reshaped.shape}")
        
        # Check if extracted actions are reasonable
        print(f"  Original action range: [{actions_reshaped.min():.4f}, {actions_reshaped.max():.4f}]")
        print(f"  Extracted action range: [{extracted_actions.min():.4f}, {extracted_actions.max():.4f}]")
        
        # Calculate reconstruction error
        mse = np.mean((actions_reshaped - extracted_actions) ** 2)
        mae = np.mean(np.abs(actions_reshaped - extracted_actions))
        print(f"  Reconstruction MSE: {mse:.6f}")
        print(f"  Reconstruction MAE: {mae:.6f}")
        
        if mse < 0.1:  # Arbitrary threshold
            print("✓ Reconstruction quality looks good!")
        else:
            print("⚠ Reconstruction quality is poor")
        
        success_without_norm = True
        
    except Exception as e:
        print(f"✗ Tokenization failed: {e}")
        import traceback
        traceback.print_exc()
        success_without_norm = False
    
    # Test tokenization with normalization
    print("\n" + "="*60)
    print("TESTING WITH NORMALIZATION")
    print("="*60)
    
    # Normalize actions to [-1, 1] range
    actions_normalized = normalize_actions(sample_actions, target_range=(-1, 1))
    state_normalized = normalize_actions(sample_qpos, target_range=(-1, 1))[0]  # (51,)
    
    print(f"Normalized action range: [{actions_normalized.min():.4f}, {actions_normalized.max():.4f}]")
    print(f"Normalized state range: [{state_normalized.min():.4f}, {state_normalized.max():.4f}]")
    
    try:
        actions_reshaped_norm = actions_normalized.reshape(1, 51)  # (1, 51)
        
        tokens_norm, token_masks_norm, ar_masks_norm, loss_masks_norm = tokenizer.tokenize(
            prompt, state_normalized, actions_reshaped_norm
        )
        
        print(f"✓ Tokenization successful!")
        print(f"  Tokens shape: {tokens_norm.shape}")
        print(f"  Token masks shape: {token_masks_norm.shape}")
        print(f"  AR masks shape: {ar_masks_norm.shape}")
        print(f"  Loss masks shape: {loss_masks_norm.shape}")
        
        # Test action extraction
        print("\nTesting action extraction...")
        extracted_actions_norm = tokenizer.extract_actions(tokens_norm, action_horizon=1, action_dim=51)
        print(f"✓ Action extraction successful!")
        print(f"  Extracted actions shape: {extracted_actions_norm.shape}")
        print(f"  Original normalized actions shape: {actions_reshaped_norm.shape}")
        
        # Check if extracted actions are reasonable
        print(f"  Original normalized action range: [{actions_reshaped_norm.min():.4f}, {actions_reshaped_norm.max():.4f}]")
        print(f"  Extracted action range: [{extracted_actions_norm.min():.4f}, {extracted_actions_norm.max():.4f}]")
        
        # Calculate reconstruction error
        mse_norm = np.mean((actions_reshaped_norm - extracted_actions_norm) ** 2)
        mae_norm = np.mean(np.abs(actions_reshaped_norm - extracted_actions_norm))
        print(f"  Reconstruction MSE: {mse_norm:.6f}")
        print(f"  Reconstruction MAE: {mae_norm:.6f}")
        
        if mse_norm < 0.1:  # Arbitrary threshold
            print("✓ Reconstruction quality looks good!")
        else:
            print("⚠ Reconstruction quality is poor")
        
        success_with_norm = True
        
    except Exception as e:
        print(f"✗ Tokenization failed: {e}")
        import traceback
        traceback.print_exc()
        success_with_norm = False
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    if success_without_norm and success_with_norm:
        print("Both tests completed successfully!")
        print(f"Without normalization - MSE: {mse:.6f}, MAE: {mae:.6f}")
        print(f"With normalization - MSE: {mse_norm:.6f}, MAE: {mae_norm:.6f}")
        
        if mse_norm < mse:
            print("✓ Normalization improved reconstruction quality!")
        else:
            print("⚠ Normalization did not improve reconstruction quality")
    
    return success_without_norm or success_with_norm

def test_single_timestep_analysis():
    """Analyze single timestep performance with and without normalization."""
    
    data_path = "./collected_data/bottle_ep_01.hdf5"
    
    print("\n" + "="*60)
    print("SINGLE TIMESTEP ANALYSIS")
    print("="*60)
    
    with h5py.File(data_path, 'r') as f:
        actions = f['action'][:]  # (2483, 51)
        qpos = f['observations/qpos'][:]
    
    tokenizer = FASTTokenizer(max_len=MAX_LEN)
    prompt = "pick up the bottle"
    
    # Use a single sample
    sample_idx = 100
    sample_actions = actions[sample_idx:sample_idx+1]  # (1, 51)
    sample_qpos = qpos[sample_idx:sample_idx+1]  # (1, 51)
    
    results = {}
    
    # Test without normalization
    print("Testing without normalization...")
    state = sample_qpos[0]  # (51,)
    actions_reshaped = sample_actions.reshape(1, 51)  # (1, 51)
    
    try:
        tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(
            prompt, state, actions_reshaped
        )
        extracted_actions = tokenizer.extract_actions(tokens, action_horizon=1, action_dim=51)
        
        mse = np.mean((actions_reshaped - extracted_actions) ** 2)
        mae = np.mean(np.abs(actions_reshaped - extracted_actions))
        
        results['without_norm'] = {'mse': mse, 'mae': mae}
        print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}")
        
    except Exception as e:
        print(f"  Error: {e}")
        results['without_norm'] = {'mse': float('inf'), 'mae': float('inf')}
    
    # Test with normalization
    print("Testing with normalization...")
    actions_normalized = normalize_actions(sample_actions, target_range=(-1, 1))
    state_normalized = normalize_actions(sample_qpos, target_range=(-1, 1))[0]  # (51,)
    
    try:
        actions_reshaped_norm = actions_normalized.reshape(1, 51)
        tokens_norm, token_masks_norm, ar_masks_norm, loss_masks_norm = tokenizer.tokenize(
            prompt, state_normalized, actions_reshaped_norm
        )
        extracted_actions_norm = tokenizer.extract_actions(tokens_norm, action_horizon=1, action_dim=51)
        
        mse_norm = np.mean((actions_reshaped_norm - extracted_actions_norm) ** 2)
        mae_norm = np.mean(np.abs(actions_reshaped_norm - extracted_actions_norm))
        
        results['with_norm'] = {'mse': mse_norm, 'mae': mae_norm}
        print(f"  MSE: {mse_norm:.6f}, MAE: {mae_norm:.6f}")
        
    except Exception as e:
        print(f"  Error: {e}")
        results['with_norm'] = {'mse': float('inf'), 'mae': float('inf')}
    
    return results

def test_50_samples_analysis():
    """Analyze performance across 50 samples."""
    
    data_path = "./collected_data/bottle_ep_01.hdf5"
    
    print("\n" + "="*60)
    print("50 SAMPLES ANALYSIS")
    print("="*60)
    
    with h5py.File(data_path, 'r') as f:
        actions = f['action'][:]  # (2483, 51)
        qpos = f['observations/qpos'][:]
    
    tokenizer = FASTTokenizer(max_len=MAX_LEN)
    prompt = "pick up the bottle"
    
    # Test with 50 different samples
    num_samples = 50
    sample_indices = np.linspace(0, len(actions)-1, num_samples, dtype=int)
    
    results_without_norm = {'mse': [], 'mae': []}
    results_with_norm = {'mse': [], 'mae': []}
    
    for i, sample_idx in enumerate(sample_indices):
        if i % 10 == 0:
            print(f"Processing sample {i+1}/{num_samples} (index {sample_idx})")
        
        sample_actions = actions[sample_idx:sample_idx+1]  # (1, 51)
        sample_qpos = qpos[sample_idx:sample_idx+1]  # (1, 51)
        
        # Test without normalization
        state = sample_qpos[0]  # (51,)
        actions_reshaped = sample_actions.reshape(1, 51)  # (1, 51)
        
        try:
            tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(
                prompt, state, actions_reshaped
            )
            extracted_actions = tokenizer.extract_actions(tokens, action_horizon=1, action_dim=51)
            
            mse = np.mean((actions_reshaped - extracted_actions) ** 2)
            mae = np.mean(np.abs(actions_reshaped - extracted_actions))
            
            results_without_norm['mse'].append(mse)
            results_without_norm['mae'].append(mae)
            
        except Exception as e:
            results_without_norm['mse'].append(float('inf'))
            results_without_norm['mae'].append(float('inf'))
        
        # Test with normalization
        actions_normalized = normalize_actions(sample_actions, target_range=(-1, 1))
        state_normalized = normalize_actions(sample_qpos, target_range=(-1, 1))[0]  # (51,)
        
        try:
            actions_reshaped_norm = actions_normalized.reshape(1, 51)
            tokens_norm, token_masks_norm, ar_masks_norm, loss_masks_norm = tokenizer.tokenize(
                prompt, state_normalized, actions_reshaped_norm
            )
            extracted_actions_norm = tokenizer.extract_actions(tokens_norm, action_horizon=1, action_dim=51)
            
            mse_norm = np.mean((actions_reshaped_norm - extracted_actions_norm) ** 2)
            mae_norm = np.mean(np.abs(actions_reshaped_norm - extracted_actions_norm))
            
            results_with_norm['mse'].append(mse_norm)
            results_with_norm['mae'].append(mae_norm)
            
        except Exception as e:
            results_with_norm['mse'].append(float('inf'))
            results_with_norm['mae'].append(float('inf'))
    
    # Calculate statistics
    def calc_stats(scores):
        valid_scores = [s for s in scores if s != float('inf')]
        if not valid_scores:
            return {'mean': float('inf'), 'std': float('inf'), 'min': float('inf'), 'max': float('inf'), 'valid_count': 0}
        return {
            'mean': np.mean(valid_scores),
            'std': np.std(valid_scores),
            'min': np.min(valid_scores),
            'max': np.max(valid_scores),
            'valid_count': len(valid_scores)
        }
    
    stats_without_norm = {metric: calc_stats(scores) for metric, scores in results_without_norm.items()}
    stats_with_norm = {metric: calc_stats(scores) for metric, scores in results_with_norm.items()}
    
    print(f"\nWithout normalization (valid samples: {stats_without_norm['mse']['valid_count']}/{num_samples}):")
    print(f"  MSE: {stats_without_norm['mse']['mean']:.6f} ± {stats_without_norm['mse']['std']:.6f}")
    print(f"  MAE: {stats_without_norm['mae']['mean']:.6f} ± {stats_without_norm['mae']['std']:.6f}")
    
    print(f"\nWith normalization (valid samples: {stats_with_norm['mse']['valid_count']}/{num_samples}):")
    print(f"  MSE: {stats_with_norm['mse']['mean']:.6f} ± {stats_with_norm['mse']['std']:.6f}")
    print(f"  MAE: {stats_with_norm['mae']['mean']:.6f} ± {stats_with_norm['mae']['std']:.6f}")
    
    return stats_without_norm, stats_with_norm

def test_action_horizon_analysis():
    """Analyze performance across different action horizons."""
    
    data_path = "./collected_data/bottle_ep_01.hdf5"
    
    print("\n" + "="*60)
    print("ACTION HORIZON ANALYSIS")
    print("="*60)
    
    with h5py.File(data_path, 'r') as f:
        actions = f['action'][:]  # (2483, 51)
        qpos = f['observations/qpos'][:]
    
    tokenizer = FASTTokenizer(max_len=MAX_LEN)
    prompt = "pick up the bottle"
    
    # Test different action horizons with fixed action_dim=51
    action_horizons = [1, 5, 20, 50, 100]  # Different horizons
    action_dim = 51  # Fixed action dimension
    
    results = {'horizons': [], 'mse': [], 'mae': []}
    
    # Use 10 samples for this analysis
    num_samples = 10
    sample_indices = np.linspace(1000, len(actions)-200, num_samples, dtype=int)
    
    for horizon in action_horizons:
        print(f"\nTesting action horizon {horizon} (action_dim {action_dim}):")
        
        mse_scores = []
        mae_scores = []
        
        for sample_idx in sample_indices:
            # Get consecutive actions for the horizon
            if sample_idx + horizon > len(actions):
                # If not enough consecutive actions, wrap around or skip
                continue
                
            # Extract consecutive action chunk: (horizon, 51)
            action_chunk = actions[sample_idx:sample_idx+horizon]  # (horizon, 51)
            state = qpos[sample_idx]  # (51,) - use state from first timestep
            
            # Normalize the entire action chunk
            actions_normalized = normalize_actions(action_chunk, target_range=(-1, 1))
            state_normalized = normalize_actions(state.reshape(1, -1), target_range=(-1, 1))[0]  # (51,)
            
            try:
                # Analyze token dimensions before tokenization
                state_str = " ".join(map(str, np.digitize(state_normalized, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1))
                prefix = f"Task: {prompt}, State: {state_str};\n"
                prefix_tokens = tokenizer._paligemma_tokenizer.encode(prefix, add_bos=True)
                
                # Get action tokens from FAST tokenizer
                action_tokens = tokenizer._fast_tokenizer(actions_normalized[None])[0]
                action_tokens_in_pg = tokenizer._act_tokens_to_paligemma_tokens(action_tokens)
                
                # Calculate postfix tokens
                postfix_tokens = (
                    tokenizer._paligemma_tokenizer.encode("Action: ")
                    + action_tokens_in_pg.tolist()
                    + tokenizer._paligemma_tokenizer.encode("|", add_eos=True)
                )
                
                # Print detailed token dimensions
                print(f"    Sample {sample_idx}:")
                print(f"      Action chunk shape: {actions_normalized.shape}")
                print(f"      Prefix tokens: {len(prefix_tokens)}")
                print(f"      FAST action tokens: {len(action_tokens)}")
                print(f"      Mapped action tokens: {len(action_tokens_in_pg)}")
                print(f"      'Action: ' tokens: {len(tokenizer._paligemma_tokenizer.encode('Action: '))}")
                print(f"      '|' tokens: {len(tokenizer._paligemma_tokenizer.encode('|', add_eos=True))}")
                print(f"      Postfix tokens: {len(postfix_tokens)}")
                print(f"      Total tokens: {len(prefix_tokens) + len(postfix_tokens)}")
                
                # Tokenize the entire action chunk at once
                tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(
                    prompt, state_normalized, actions_normalized
                )
                extracted_actions = tokenizer.extract_actions(tokens, action_horizon=horizon, action_dim=action_dim)
                
                print(f"    Extracted actions shape: {extracted_actions.shape}")
                print(f"    Actions normalized shape: {actions_normalized.shape}")
                
                mse = np.mean((actions_normalized - extracted_actions) ** 2)
                mae = np.mean(np.abs(actions_normalized - extracted_actions))
                
                mse_scores.append(mse)
                mae_scores.append(mae)
                
            except Exception as e:
                print(f"    Error: {e}")
                mse_scores.append(float('inf'))
                mae_scores.append(float('inf'))
        
        # Calculate average for this horizon
        valid_mse = [mse for mse in mse_scores if mse != float('inf')]
        valid_mae = [mae for mae in mae_scores if mae != float('inf')]
        
        if valid_mse:
            avg_mse = np.mean(valid_mse)
            avg_mae = np.mean(valid_mae)
            print(f"  Average MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f} (valid: {len(valid_mse)}/{len(mse_scores)})")
            
            results['horizons'].append(horizon)
            results['mse'].append(avg_mse)
            results['mae'].append(avg_mae)
        else:
            print(f"  All samples failed for horizon {horizon}")
    
    return results

def create_plots(action_horizon_results):
    """Create plots for action horizon analysis."""
    
    if not action_horizon_results['horizons']:
        print("No valid results to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # MSE plot
    ax1.plot(action_horizon_results['horizons'], action_horizon_results['mse'], 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Action Horizon')
    ax1.set_ylabel('MSE')
    ax1.set_title('Reconstruction MSE vs Action Horizon')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # MAE plot
    ax2.plot(action_horizon_results['horizons'], action_horizon_results['mae'], 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Action Horizon')
    ax2.set_ylabel('MAE')
    ax2.set_title('Reconstruction MAE vs Action Horizon')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('action_horizon_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as 'action_horizon_analysis.png'")
    
    # Also create a combined plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ax.plot(action_horizon_results['horizons'], action_horizon_results['mse'], 'bo-', 
            linewidth=2, markersize=8, label='MSE')
    ax.plot(action_horizon_results['horizons'], action_horizon_results['mae'], 'ro-', 
            linewidth=2, markersize=8, label='MAE')
    
    ax.set_xlabel('Action Horizon')
    ax.set_ylabel('Error')
    ax.set_title('Reconstruction Error vs Action Horizon')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('combined_error_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Combined plot saved as 'combined_error_analysis.png'")
    
    plt.show()

if __name__ == "__main__":
    print("=" * 80)
    print("FASTTokenizer Comprehensive Analysis")
    print(f"Max token length: {MAX_LEN}")
    print("=" * 80)
    
    # # 0. Dataset tokenization with normalization
    # dataset_tokenization_results = test_dataset_tokenization_with_normalization()
    
    # # 1. Single timestep analysis
    # single_timestep_results = test_single_timestep_analysis()
    
    # # 2. 50 samples analysis
    # stats_without_norm, stats_with_norm = test_50_samples_analysis()
    
    # 3. Action horizon analysis
    action_horizon_results = test_action_horizon_analysis()
    
    # 4. Create plots
    create_plots(action_horizon_results)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # print("\n0. Dataset Tokenization with Normalization:")
    # print("  ✓ Tokenization successful with both normalization approaches")
    
    # print("\n1. Single Timestep Analysis:")
    # if 'without_norm' in single_timestep_results and 'with_norm' in single_timestep_results:
    #     print(f"  Without normalization - MSE: {single_timestep_results['without_norm']['mse']:.6f}, MAE: {single_timestep_results['without_norm']['mae']:.6f}")
    #     print(f"  With normalization - MSE: {single_timestep_results['with_norm']['mse']:.6f}, MAE: {single_timestep_results['with_norm']['mae']:.6f}")
    
    # print("\n2. 50 Samples Analysis:")
    # print(f"  Without normalization - MSE: {stats_without_norm['mse']['mean']:.6f} ± {stats_without_norm['mse']['std']:.6f}")
    # print(f"  With normalization - MSE: {stats_with_norm['mse']['mean']:.6f} ± {stats_with_norm['mse']['std']:.6f}")
    
    print("\n3. Action Horizon Analysis:")
    for i, horizon in enumerate(action_horizon_results['horizons']):
        print(f"  Horizon {horizon}: MSE={action_horizon_results['mse'][i]:.6f}, MAE={action_horizon_results['mae'][i]:.6f}")
    
    print("\n✓ Comprehensive analysis completed!")
    print("=" * 80)
