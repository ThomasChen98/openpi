# Reward Method Configuration Update

## Overview

Added support for specifying the reward model method in the YAML training configuration file. Users can now easily switch between "Ours" (fine-tuned Qwen model) and "GVL" (OpenAI GPT-5.2) reward models.

## Changes Made

### 1. Configuration File (`training_config.yaml`)

Added a new `reward.method` parameter:

```yaml
reward:
  method: "Ours"  # or "GVL"
  task_instruction: "..."
  checkpoint_path: "/path/to/checkpoint"  # Required only for "Ours"
  # ... other reward parameters
```

### 2. Scripts Updated

#### `scripts/integrated_training_h1.sh`
- Added `REWARD_METHOD` configuration variable (line 100)
- Reads `reward.method` from YAML config (defaults to "Ours")
- Validates method-specific requirements:
  - "Ours": Checks for valid checkpoint path
  - "GVL": Checks for OPENAI_API_KEY environment variable
- Passes `--reward-method` to downstream scripts
- Updated logging to display reward method being used

#### `scripts/convert_h1_data.sh`
- Added `--reward-method` command-line argument
- Added `REWARD_METHOD` variable (defaults to "Ours")
- Passes reward method to:
  - `compute_action_chunk_advantages.py`
  - `convert_h1_data_to_lerobot.py`

#### `examples/h1_control_client/compute_action_chunk_advantages.py`
- Added `--reward-method` argument with choices ["Ours", "GVL"]
- Updated `ActionChunkAdvantageComputer.__init__()`:
  - Accepts `reward_method` parameter
  - Conditionally loads Qwen model or OpenAI client based on method
- Updated `compute_episode_rewards()`:
  - Supports both Qwen (for "Ours") and OpenAI GVL (for "GVL") inference
- Cache files now include method name: `episode_000000_Ours_reward.pkl` or `episode_000000_GVL_reward.pkl`
- Added validation for method-specific requirements

#### `examples/h1_control_client/convert_h1_data_to_lerobot.py`
- Added `reward_method` parameter to `main()` function (defaults to "Ours")
- Updated docstring to document the new parameter
- Conditional import and execution:
  - "Ours": Uses `qwen_reward_labeling.label_episodes()`
  - "GVL": Uses `embodied_reward_labeling.label_episodes()`
- Method-specific validation and logging

### 3. Documentation

Created two new files:

#### `reward_method_config_example.yaml`
- Example configuration for both methods
- Detailed comments explaining differences
- Environment variable requirements
- Cache file naming conventions

#### `REWARD_METHOD_UPDATE.md` (this file)
- Complete documentation of changes
- Usage examples
- Migration guide

## Usage

### In Your Training Config YAML

```yaml
# Option 1: Use fine-tuned Qwen model
reward:
  method: "Ours"
  task_instruction: "Pick up the bottle and insert it into the tray."
  checkpoint_path: "/path/to/qwen/checkpoint"
  max_frames: 20
  advantage_threshold: 0.3

# Option 2: Use OpenAI GVL model
reward:
  method: "GVL"
  task_instruction: "Fold the towel into a small square."
  # No checkpoint_path needed
  max_frames: 30
  advantage_threshold: 0.3
```

### Environment Variables

**For "Ours" method:**
- No special environment variables needed
- Uses conda base environment automatically: `/home/yuxin/miniconda/bin/python`

**For "GVL" method:**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

### Running Training

```bash
# Use default config (includes reward.method setting)
./scripts/integrated_training_h1.sh

# Or specify custom config
./scripts/integrated_training_h1.sh --config my_config.yaml
```

## Method Comparison

| Feature | Ours (Qwen) | GVL (GPT-5.2) |
|---------|-------------|---------------|
| **Requires** | Checkpoint path | OpenAI API key |
| **Model** | Fine-tuned Qwen3VL | OpenAI GPT-5.2 |
| **Environment** | Conda base | Any |
| **Speed** | Fast (local GPU) | Slower (API calls) |
| **Parallelization** | Limited (GPU batch) | High (API workers) |
| **Cost** | Free (local) | Pay per API call |
| **Customization** | Fully customizable | Fixed model |

## Cache File System

Reward computations are cached with method-specific filenames:

```
data/chunk-000/
├── episode_000000.parquet
├── episode_000000_Ours_reward.pkl        # Cached for "Ours" method
├── episode_000000_GVL_reward.pkl         # Cached for "GVL" method
├── episode_000000_ego_image_embeddings.pkl  # Shared between methods
└── episode_000000_action_chunk_advantages.pkl
```

This allows switching between methods without recomputing all rewards.

## Migration Guide

### Existing Configs

If you have existing training configs without `reward.method`:

1. **No changes required**: Defaults to "Ours" (backward compatible)

2. **To use GVL**: Add to your config:
   ```yaml
   reward:
     method: "GVL"
     # Remove or comment out checkpoint_path
     # Set OPENAI_API_KEY environment variable
   ```

### Existing Cache Files

Old cache files without method suffix (`episode_000000_reward.pkl`) will be treated as "Ours" method caches. To regenerate with method-specific names:

```bash
# Option 1: Force recompute (pass --force-recompute flag internally)
rm -f data/chunk-000/*_reward.pkl

# Option 2: Rename existing caches
cd data/chunk-000
for f in episode_*_reward.pkl; do
    mv "$f" "${f/_reward.pkl/_Ours_reward.pkl}"
done
```

## Error Handling

The system validates method-specific requirements:

**"Ours" method without checkpoint:**
```
ERROR: reward.checkpoint_path not set in config file!
Qwen checkpoint path is required for reward.method='Ours'
```

**"GVL" method without API key:**
```
ERROR: OPENAI_API_KEY not set in environment!
OpenAI API key is required for reward.method='GVL'
```

**Invalid method:**
```
ERROR: Unknown reward.method: InvalidMethod
Supported methods: 'Ours', 'GVL'
```

## Testing

To test both methods:

```bash
# Test Ours method
export QWEN_REWARD_CHECKPOINT_PATH="/path/to/checkpoint"
./scripts/integrated_training_h1.sh --config config_ours.yaml

# Test GVL method
export OPENAI_API_KEY="your-key"
./scripts/integrated_training_h1.sh --config config_gvl.yaml
```

## Future Enhancements

Potential additions:
- Support for additional reward models (e.g., other VLMs)
- Ensemble methods combining multiple reward models
- Automatic method selection based on task characteristics
- Benchmark comparisons between methods

## Questions?

For issues or questions about the reward method configuration:
1. Check `reward_method_config_example.yaml` for examples
2. Verify environment variables are set correctly
3. Check cache file naming if switching methods
4. Review error messages for specific requirements
