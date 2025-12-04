#!/bin/bash
# Convert H1 HDF5 data to LeRobot format with optional advantage labeling
#
# Usage:
#   ./scripts/convert_data.sh                                      # Use defaults
#   ./scripts/convert_data.sh --task-name my_task --epoch 0        # With epoch
#   ./scripts/convert_data.sh --labeling-mode human_labeling       # With advantage labeling
#
# Environment variables can also be used:
#   TASK_NAME="my_task" EPOCH_NUM=0 ./scripts/convert_data.sh

set -e

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task-name)
            TASK_NAME="$2"
            shift 2
            ;;
        --task-description)
            TASK_DESCRIPTION="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --epoch)
            EPOCH_NUM="$2"
            shift 2
            ;;
        --labeling-mode)
            LABELING_MODE="$2"
            shift 2
            ;;
        --num-repeats)
            NUM_REPEATS="$2"
            shift 2
            ;;
        --config-name)
            CONFIG_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default values (can be overridden by environment variables or command line)
TASK_NAME="${TASK_NAME:-press_a_big_button}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-press the big button}"
EPOCH_NUM="${EPOCH_NUM:-}"  # Empty means no epoch suffix
LABELING_MODE="${LABELING_MODE:-none}"  # Options: none, human_labeling, reward_labeling
NUM_REPEATS="${NUM_REPEATS:-1}"
CONFIG_NAME="${CONFIG_NAME:-pi05_h1_auto}"

# Base directories
BASE_DATA_DIR="${BASE_DATA_DIR:-examples/h1_control_client/h1_data_processed}"
BASE_LEROBOT_DIR="${BASE_LEROBOT_DIR:-examples/h1_control_client/h1_data_lerobot}"

# Construct paths based on whether epoch is specified
if [ -n "$EPOCH_NUM" ]; then
    # Epoch-based directory structure
    DATA_DIR="${DATA_DIR:-$BASE_DATA_DIR/$TASK_NAME/epoch_$EPOCH_NUM/raw}"
    SAVE_DIR="${TASK_NAME}/epoch_${EPOCH_NUM}"
    LEROBOT_DATA_DIR="$(pwd)/$BASE_LEROBOT_DIR/$SAVE_DIR"
else
    # Flat directory structure (backwards compatible)
    DATA_DIR="${DATA_DIR:-$BASE_DATA_DIR/$TASK_NAME}"
    SAVE_DIR="$TASK_NAME"
    LEROBOT_DATA_DIR="$(pwd)/$BASE_LEROBOT_DIR/$SAVE_DIR"
fi

echo "========================================================"
echo "  > Converting data to LeRobot format..."
echo "========================================================"
echo "Task name: $TASK_NAME"
echo "Task description: $TASK_DESCRIPTION"
echo "Data directory: $DATA_DIR"
echo "Save directory: $SAVE_DIR"
echo "LeRobot data directory: $LEROBOT_DATA_DIR"
echo "Number of repeats: $NUM_REPEATS"
echo "Labeling mode: $LABELING_MODE"
echo "Config name: $CONFIG_NAME"
if [ -n "$EPOCH_NUM" ]; then
    echo "Epoch: $EPOCH_NUM"
fi
echo "========================================================"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory does not exist: $DATA_DIR"
    exit 1
fi

# Check if there are HDF5 files in the directory
HDF5_COUNT=$(find "$DATA_DIR" -maxdepth 1 -name "*.hdf5" | wc -l)
if [ "$HDF5_COUNT" -eq 0 ]; then
    echo "ERROR: No HDF5 files found in: $DATA_DIR"
    exit 1
fi
echo "Found $HDF5_COUNT HDF5 file(s)"

# Build the convert command with optional labeling mode
CONVERT_CMD="uv run examples/h1_control_client/convert_h1_data_to_lerobot.py \
    --data_dir \"$DATA_DIR\" \
    --task_description \"$TASK_DESCRIPTION\" \
--num_repeats $NUM_REPEATS \
    --save_dir \"$SAVE_DIR\""

if [ "$LABELING_MODE" != "none" ]; then
    CONVERT_CMD="$CONVERT_CMD --labeling_mode $LABELING_MODE"
fi

echo ""
echo "Running conversion..."
eval $CONVERT_CMD

echo ""
echo "========================================================"
echo "  > Computing normalization statistics..."
echo "========================================================"

uv run scripts/compute_norm_stats.py \
--config-name "$CONFIG_NAME" \
--data-dir "$LEROBOT_DATA_DIR"

echo ""
echo "========================================================"
echo "  > Done!"
echo "========================================================"
echo "Output:"
echo "  LeRobot data: $LEROBOT_DATA_DIR"
echo "  Norm stats: $LEROBOT_DATA_DIR/norm_stats.json"
echo "========================================================"
