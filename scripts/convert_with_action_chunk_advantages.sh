#!/bin/bash
# Two-phase conversion for action chunk advantages
#
# Phase 1: Convert HDF5 to parquet (creates parquet files)
# Phase 2: Compute advantages on parquet files and add to dataset
#
# Usage:
#   ./scripts/convert_with_action_chunk_advantages.sh \
#       --task-name fold_towel \
#       --epoch 0 \
#       --config examples/h1_control_client/training_config.yaml

set -e

# Default values
TASK_NAME=""
EPOCH_NUM=""
CONFIG_FILE=""
TASK_DESCRIPTION=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task-name)
            TASK_NAME="$2"
            shift 2
            ;;
        --epoch)
            EPOCH_NUM="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --task-description)
            TASK_DESCRIPTION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$TASK_NAME" ] || [ -z "$EPOCH_NUM" ]; then
    echo "ERROR: Missing required parameters"
    echo "Usage: $0 --task-name TASK --epoch N [--config CONFIG]"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load config if provided
if [ -n "$CONFIG_FILE" ]; then
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "ERROR: Config file not found: $CONFIG_FILE"
        exit 1
    fi
    
    # Read values from config
    TASK_DESCRIPTION=$(yq -r '.task.description // ""' "$CONFIG_FILE")
    CONFIG_NAME=$(yq -r '.policy.config_name // "pi05_h1_auto"' "$CONFIG_FILE")
    LABELING_MODE=$(yq -r '.training.labeling_mode // "none"' "$CONFIG_FILE")
    NUM_REPEATS=$(yq -r '.training.num_repeats // 1' "$CONFIG_FILE")
    ACTION_DIM=$(yq -r '.robot.include_hands // false' "$CONFIG_FILE")
    if [ "$ACTION_DIM" = "true" ]; then
        ACTION_DIM=26
    else
        ACTION_DIM=14
    fi
    
    # Reward parameters
    REWARD_TASK_INSTRUCTION=$(yq -r '.reward.task_instruction // ""' "$CONFIG_FILE")
    REWARD_CHECKPOINT_PATH=$(yq -r '.reward.checkpoint_path // ""' "$CONFIG_FILE")
    REWARD_MAX_FRAMES=$(yq -r '.reward.max_frames // 30' "$CONFIG_FILE")
    REWARD_LOOK_AHEAD_WINDOW=$(yq -r '.reward.look_ahead_window // 80' "$CONFIG_FILE")
    REWARD_ADVANTAGE_THRESHOLD=$(yq -r '.reward.advantage_threshold // 0.33' "$CONFIG_FILE")
    GPU_ID=$(yq -r '.training.gpu_id // 0' "$CONFIG_FILE")
fi

# Default values if not set
TASK_DESCRIPTION="${TASK_DESCRIPTION:-$TASK_NAME}"
CONFIG_NAME="${CONFIG_NAME:-pi05_h1_auto}"
NUM_REPEATS="${NUM_REPEATS:-1}"
ACTION_DIM="${ACTION_DIM:-26}"

echo "========================================================"
echo "  Two-Phase Conversion with Action Chunk Advantages"
echo "========================================================"
echo "Task: $TASK_NAME"
echo "Epoch: $EPOCH_NUM"
echo "Config: ${CONFIG_FILE:-none}"
echo ""

# Phase 1: Initial conversion (creates parquet files)
echo "========================================================"
echo "  PHASE 1: Initial Conversion"
echo "========================================================"
echo "Creating parquet files from HDF5..."
echo ""

./scripts/convert_h1_data.sh \
    --task-name "$TASK_NAME" \
    --task-description "$TASK_DESCRIPTION" \
    --epoch "$EPOCH_NUM" \
    --labeling-mode "none" \
    --num-repeats "$NUM_REPEATS" \
    --config-name "$CONFIG_NAME" \
    --action-dim "$ACTION_DIM"

echo ""
echo "✓ Phase 1 complete: Parquet files created"
echo ""

# Phase 2: Compute advantages
echo "========================================================"
echo "  PHASE 2: Compute Action Chunk Advantages"
echo "========================================================"

if [ -z "$REWARD_CHECKPOINT_PATH" ]; then
    echo "ERROR: reward.checkpoint_path not set in config!"
    echo "Cannot compute action chunk advantages without Qwen checkpoint."
    exit 1
fi

if [ ! -d "$REWARD_CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint path does not exist: $REWARD_CHECKPOINT_PATH"
    exit 1
fi

# Find parquet directory
PARQUET_DIR="$PROJECT_ROOT/examples/h1_control_client/h1_data_lerobot/$TASK_NAME/epoch_${EPOCH_NUM}/data/chunk-000"

if [ ! -d "$PARQUET_DIR" ]; then
    echo "ERROR: Parquet directory not found: $PARQUET_DIR"
    echo "Phase 1 conversion may have failed."
    exit 1
fi

echo "Computing advantages on parquet files..."
echo "  Directory: $PARQUET_DIR"
echo "  Checkpoint: $REWARD_CHECKPOINT_PATH"
echo "  Task: $REWARD_TASK_INSTRUCTION"
echo ""

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID
export QWEN_REWARD_CHECKPOINT_PATH="$REWARD_CHECKPOINT_PATH"

$PROJECT_ROOT/.venv/bin/python examples/h1_control_client/compute_action_chunk_advantages.py \
    --data-dir "$PARQUET_DIR" \
    --task-instruction "$REWARD_TASK_INSTRUCTION" \
    --checkpoint-path "$REWARD_CHECKPOINT_PATH" \
    --max-frames "$REWARD_MAX_FRAMES" \
    --look-ahead-window "$REWARD_LOOK_AHEAD_WINDOW" \
    --advantage-threshold "$REWARD_ADVANTAGE_THRESHOLD"

echo ""
echo "✓ Phase 2 complete: Advantages computed"
echo ""

# Phase 3: Re-convert with advantages
echo "========================================================"
echo "  PHASE 3: Re-convert with Advantages"
echo "========================================================"
echo "Adding advantages to dataset..."
echo ""

# Remove old dataset
rm -rf "$PROJECT_ROOT/examples/h1_control_client/h1_data_lerobot/$TASK_NAME/epoch_${EPOCH_NUM}"

# Re-convert with action_chunk_advantage mode
./scripts/convert_h1_data.sh \
    --task-name "$TASK_NAME" \
    --task-description "$TASK_DESCRIPTION" \
    --epoch "$EPOCH_NUM" \
    --labeling-mode "action_chunk_advantage" \
    --num-repeats "$NUM_REPEATS" \
    --config-name "$CONFIG_NAME" \
    --action-dim "$ACTION_DIM" \
    --reward-task-instruction "$REWARD_TASK_INSTRUCTION" \
    --reward-max-frames "$REWARD_MAX_FRAMES" \
    --reward-advantage-threshold "$REWARD_ADVANTAGE_THRESHOLD"

echo ""
echo "========================================================"
echo "  ✓ COMPLETE!"
echo "========================================================"
echo "Dataset created with action chunk advantages:"
echo "  $PROJECT_ROOT/examples/h1_control_client/h1_data_lerobot/$TASK_NAME/epoch_${EPOCH_NUM}"
echo ""
echo "You can now train with this dataset."
echo "========================================================"

