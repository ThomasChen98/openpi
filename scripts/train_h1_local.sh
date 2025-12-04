#!/bin/bash
# Train H1 policy with local LeRobot dataset
#
# Usage:
#   ./scripts/train_h1_local.sh                                    # Use defaults
#   ./scripts/train_h1_local.sh --task-name my_task --epoch 0      # With epoch
#   ./scripts/train_h1_local.sh --resume-from checkpoints/...      # Resume from checkpoint
#
# Environment variables can also be used:
#   TASK_NAME="my_task" EPOCH_NUM=0 ./scripts/train_h1_local.sh

set -e

# Parse command line arguments
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
        --config-name)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --resume-from)
            RESUME_FROM="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --max-epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --save-interval)
            SAVE_INTERVAL="$2"
            shift 2
            ;;
        --keep-period)
            KEEP_PERIOD="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default values
TASK_NAME="${TASK_NAME:-press_a_big_button}"
EPOCH_NUM="${EPOCH_NUM:-}"  # Empty means no epoch suffix
CONFIG_NAME="${CONFIG_NAME:-pi05_h1_auto}"
GPU_ID="${GPU_ID:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-}"  # Empty means use config default
SAVE_INTERVAL="${SAVE_INTERVAL:-}"  # Empty means use config default
KEEP_PERIOD="${KEEP_PERIOD:-}"  # Empty means use config default

# Base directories
BASE_LEROBOT_DIR="${BASE_LEROBOT_DIR:-examples/h1_control_client/h1_data_lerobot}"
BASE_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR:-checkpoints}"

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

# Construct paths based on whether epoch is specified
if [ -n "$EPOCH_NUM" ]; then
    # Epoch-based directory structure
    DATA_DIR="${TASK_NAME}/epoch_${EPOCH_NUM}"
    EXP_NAME="${CONFIG_NAME}_${TASK_NAME}_epoch${EPOCH_NUM}"
    CHECKPOINT_DIR="$BASE_CHECKPOINT_DIR/${CONFIG_NAME}/${TASK_NAME}/epoch_${EPOCH_NUM}"
else
    # Flat directory structure (backwards compatible)
    DATA_DIR="$TASK_NAME"
    EXP_NAME="${CONFIG_NAME}_${TASK_NAME}"
    CHECKPOINT_DIR="$BASE_CHECKPOINT_DIR/${CONFIG_NAME}/${TASK_NAME}"
fi

LEROBOT_DATA_DIR="$(pwd)/$BASE_LEROBOT_DIR/$DATA_DIR"

echo "========================================================"
echo "  > Training H1 with local dataset..."
echo "========================================================"
echo "Task name: $TASK_NAME"
echo "Config name: $CONFIG_NAME"
echo "Exp name: $EXP_NAME"
echo "Data directory: $DATA_DIR"
echo "LeRobot data directory: $LEROBOT_DATA_DIR"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "GPU: $GPU_ID"
if [ -n "$EPOCH_NUM" ]; then
    echo "Epoch: $EPOCH_NUM"
fi
if [ -n "$RESUME_FROM" ]; then
    echo "Resuming from: $RESUME_FROM"
fi
if [ -n "$MAX_EPOCHS" ]; then
    echo "Max epochs: $MAX_EPOCHS"
fi
if [ -n "$SAVE_INTERVAL" ]; then
    echo "Save interval: $SAVE_INTERVAL"
fi
if [ -n "$KEEP_PERIOD" ]; then
    echo "Keep period: $KEEP_PERIOD"
fi
echo "========================================================"

# Check if data directory exists
if [ ! -d "$LEROBOT_DATA_DIR" ]; then
    echo "ERROR: LeRobot data directory does not exist: $LEROBOT_DATA_DIR"
    echo "Have you run convert_data.sh first?"
    exit 1
fi

# Check if norm_stats.json exists
if [ ! -f "$LEROBOT_DATA_DIR/norm_stats.json" ]; then
    echo "ERROR: norm_stats.json not found in: $LEROBOT_DATA_DIR"
    echo "Have you run convert_data.sh first?"
    exit 1
fi

# Build training command
TRAIN_CMD="uv run scripts/train_auto.py \
    --config-name \"$CONFIG_NAME\" \
    --exp-name \"$EXP_NAME\" \
    --overwrite \
    --data-dir \"$LEROBOT_DATA_DIR\""

# Add resume flag if specified
if [ -n "$RESUME_FROM" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume-from \"$RESUME_FROM\""
fi

# Add training parameters if specified
if [ -n "$MAX_EPOCHS" ]; then
    TRAIN_CMD="$TRAIN_CMD --max-epochs $MAX_EPOCHS"
fi
if [ -n "$SAVE_INTERVAL" ]; then
    TRAIN_CMD="$TRAIN_CMD --save-interval $SAVE_INTERVAL"
fi
if [ -n "$KEEP_PERIOD" ]; then
    TRAIN_CMD="$TRAIN_CMD --keep-period $KEEP_PERIOD"
fi

echo ""
echo "Running training..."
eval $TRAIN_CMD

# Get the actual checkpoint path (latest step)
LATEST_CHECKPOINT=$(ls -d $CHECKPOINT_DIR/*/ 2>/dev/null | sort -V | tail -1)

echo ""
echo "========================================================"
echo "  > Training complete!"
echo "========================================================"
echo "Checkpoint saved to: ${LATEST_CHECKPOINT:-$CHECKPOINT_DIR}"
echo "========================================================"
