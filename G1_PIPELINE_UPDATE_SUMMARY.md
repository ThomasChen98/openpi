# G1 Pipeline Update Summary

## Overview

Successfully updated the G1 training and data conversion pipeline to adopt all finalized features from the H1 pipeline, including support for multiple reward methods (Ours, GVL, RoboDopamine) and action chunk advantage labeling.

## Files Modified

### 1. Training Scripts

#### `/scripts/integrated_training_g1.sh`
**Purpose**: Main G1 training pipeline script

**New Features Added**:
- ✅ Multi-method reward support: `Ours` (Qwen3VL), `GVL` (GPT-4o), `RoboDopamine`
- ✅ Action chunk advantage mode with three-phase conversion
- ✅ Additional reward parameters:
  - `REWARD_METHOD`: Choose reward computation method
  - `REWARD_LOOK_AHEAD_WINDOW`: Temporal reward aggregation window
  - `REWARD_RANDOM_DROP_RATE`: Randomly keep original prompts
  - `REWARD_GOAL_IMAGE_PATH`: Goal image for RoboDopamine
- ✅ Method-specific validation and checks
- ✅ Proper environment switching (conda base for rewards, .venv for training)
- ✅ Separated convert phase in menu (matching H1)

**G1-Specific Settings Maintained**:
- Port 8001 for policy server
- Port 8081 for Viser visualizer  
- Fixed 29-dim action space (28 upper body + 1 waist_yaw)
- G1-specific data paths (`g1_data_auto`, `g1_data_lerobot`)

#### `/scripts/integrated_training_g1_SFT.sh`
**Purpose**: SFT ablation study version

**Key Features**:
- Forces `human_labeling` mode for all epochs
- Always filters to keep only good episodes (`--filter-good-only`)
- Disables action chunk advantage computation
- Simpler two-phase workflow (data collection → training)
- Uses only human-provided episode-level labels

### 2. Data Conversion Scripts

#### `/scripts/convert_g1_data.sh`
**New Parameters Added**:
```bash
--reward-method         # Ours | GVL | RoboDopamine
--reward-goal-image-path # For RoboDopamine
--reward-random-drop-rate # For action chunk advantage
```

**New Validation Logic**:
- Checks for required dependencies based on selected method:
  - `Ours`: Validates `QWEN_REWARD_CHECKPOINT_PATH`
  - `GVL`: Validates `OPENAI_API_KEY`
  - `RoboDopamine`: Validates `REWARD_GOAL_IMAGE_PATH`

**Action Chunk Advantage Support**:
- Pre-computation check for parquet files
- Automatic advantage computation if missing
- Three-phase conversion workflow
- Uses H1's `compute_action_chunk_advantages.py` (shared across robots)

**Environment Handling**:
- Conda base Python for reward computations
- Project .venv for policy training
- Proper `LD_LIBRARY_PATH` setup for CUDA libraries

#### `/examples/g1_control_client/convert_g1_data_to_lerobot.py`
**Major Updates**:

1. **New Labeling Modes**:
```python
LabelingMode = Literal["none", "human_labeling", "reward_labeling", "action_chunk_advantage"]
```

2. **New Parameters**:
```python
reward_method: str = "Ours"  # Ours | GVL | RoboDopamine
reward_random_drop_rate: float = 0.0  # For action chunk advantage
reward_goal_image_path: str = None  # For RoboDopamine
```

3. **Action Chunk Advantage Implementation**:
- Load pre-computed advantages from pickle files
- Per-frame advantage labeling (instead of episode-level)
- Random drop support to keep some original prompts
- Detailed statistics logging

4. **Method-Specific Validation**:
- RoboDopamine only supported for `action_chunk_advantage` mode
- Proper error messages for missing dependencies

## Three-Phase Conversion Workflow

For `action_chunk_advantage` mode, the conversion now follows a three-phase approach:

### Phase 1: Create Parquet Files
```bash
# Convert HDF5 → Parquet (without advantages)
./scripts/convert_g1_data.sh \
    --labeling-mode "none" \
    --task-name "grasp_bottle" \
    --epoch 1
```

### Phase 2: Compute Advantages
```bash
# Uses conda base environment
python examples/h1_control_client/compute_action_chunk_advantages.py \
    --data-dir "g1_data_lerobot/grasp_bottle/epoch_1/data/chunk-000" \
    --reward-method "Ours" \
    --checkpoint-path "/path/to/qwen_checkpoint" \
    --task-instruction "grasp the bottle"
```

### Phase 3: Re-convert with Advantages
```bash
# Convert Parquet → LeRobot (with advantages)
./scripts/convert_g1_data.sh \
    --labeling-mode "action_chunk_advantage" \
    --reward-method "Ours" \
    --task-name "grasp_bottle" \
    --epoch 1
```

## Reward Method Comparison

| Method | Model | Requirements | Use Case |
|--------|-------|--------------|----------|
| **Ours** | Qwen3-VL (fine-tuned) | Checkpoint path | Best performance, requires training |
| **GVL** | GPT-4o (OpenAI) | API key | Zero-shot, requires API costs |
| **RoboDopamine** | GRM-3B | Goal image | Goal-conditioned tasks |

## Configuration Example

### Action Chunk Advantage with Qwen (Ours)
```yaml
# training_config_g1.yaml

training:
  labeling_mode: "action_chunk_advantage"
  max_epochs: 1000
  gpu_id: 0

reward:
  method: "Ours"
  checkpoint_path: "/path/to/qwen/checkpoint-750"
  task_instruction: "Fold the towel into a small square"
  max_frames: 30
  look_ahead_window: 80
  advantage_threshold: 0.3
  random_drop_rate: 0.0

policy_server:
  host: "localhost"
  port: 8001  # G1 uses 8001, H1 uses 8000

visualization:
  viser_port: 8081  # G1 uses 8081, H1 uses 8080
```

### Action Chunk Advantage with RoboDopamine
```yaml
reward:
  method: "RoboDopamine"
  goal_image_path: "/path/to/goal_image.png"
  task_instruction: "Grasp the bottle"
  max_frames: 30
  look_ahead_window: 80
  advantage_threshold: 0.3
```

### Action Chunk Advantage with GVL (GPT-4o)
```yaml
reward:
  method: "GVL"
  task_instruction: "Insert the bottle into the cabinet"
  max_frames: 30
  look_ahead_window: 80
  advantage_threshold: 0.3
  
# Also set in environment:
# export OPENAI_API_KEY="sk-..."
```

### SFT Ablation (Good Episodes Only)
```yaml
# training_config_g1_SFT.yaml

training:
  labeling_mode: "human_labeling"  # SFT always uses human labels
  max_epochs: 1000

# No reward parameters needed for SFT mode
```

## Usage Examples

### Standard Pipeline with Action Chunk Advantages
```bash
./scripts/integrated_training_g1.sh --config training_config_g1.yaml
```

### SFT Ablation Study
```bash
./scripts/integrated_training_g1_SFT.sh --config training_config_g1_SFT.yaml
```

### Manual Data Conversion
```bash
# With action chunk advantages
./scripts/convert_g1_data.sh \
    --task-name "grasp_bottle" \
    --task-description "grasp the bottle" \
    --epoch 1 \
    --labeling-mode "action_chunk_advantage" \
    --reward-method "Ours" \
    --reward-task-instruction "grasp the bottle" \
    --reward-advantage-threshold 0.3 \
    --reward-random-drop-rate 0.0 \
    --data-dir "g1_data_auto/grasp_bottle/epoch_1/raw"

# SFT mode (good episodes only)
./scripts/convert_g1_data.sh \
    --task-name "grasp_bottle" \
    --task-description "grasp the bottle" \
    --epoch 0 \
    --labeling-mode "human_labeling" \
    --filter-good-only \
    --data-dir "g1_data_auto/grasp_bottle/epoch_0/raw"
```

## Key Differences from H1 Pipeline

While the G1 pipeline now has feature parity with H1, these G1-specific characteristics are maintained:

1. **Action Space**: Fixed 29-dim (28 upper body + 1 waist_yaw) vs H1's 14/26-dim
2. **Ports**: 8001 (server) and 8081 (viz) vs H1's 8000 and 8080
3. **Camera Setup**: Single head camera vs H1's head + wrist cameras
4. **Data Paths**: `g1_data_auto/`, `g1_data_lerobot/` vs H1's `h1_data_auto/`, `h1_data_lerobot/`
5. **Shared Components**: Uses H1's `compute_action_chunk_advantages.py` (robot-agnostic)

## Testing Checklist

- [x] Bash scripts have no syntax errors
- [x] Python conversion script has no linter errors
- [x] All new parameters are properly documented
- [x] Method-specific validation is working
- [x] Environment switching logic is correct
- [x] Action chunk advantage loading is implemented
- [x] Per-frame advantage labeling is working
- [x] Random drop rate is applied correctly
- [x] SFT ablation script is created
- [ ] End-to-end testing with actual G1 data (requires hardware)

## Migration Guide

For existing G1 users, update your config files:

```yaml
# Add these sections to your training_config_g1.yaml

reward:
  method: "Ours"  # or "GVL" or "RoboDopamine"
  checkpoint_path: "/path/to/checkpoint"  # for Ours
  task_instruction: "detailed task description"
  max_frames: 30
  look_ahead_window: 80
  advantage_threshold: 0.3
  random_drop_rate: 0.0
  # goal_image_path: "/path/to/goal.png"  # for RoboDopamine
```

## Future Work

1. Shared reward computation module for H1 and G1
2. G1-specific goal image support for RoboDopamine
3. Multi-GPU training support
4. Distributed data collection from multiple G1 robots

## Related Files

- `/scripts/integrated_training_h1.sh` - H1 reference implementation
- `/scripts/integrated_training_h1_SFT.sh` - H1 SFT ablation reference
- `/examples/h1_control_client/compute_action_chunk_advantages.py` - Shared advantage computation
- `/examples/h1_control_client/embodied_reward_labeling.py` - Reward model implementation
