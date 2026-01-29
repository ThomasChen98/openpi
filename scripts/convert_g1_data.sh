#!/bin/bash
# Convert G1 HDF5 data to LeRobot format with optional advantage labeling
#
# Usage:
#   ./scripts/convert_g1_data.sh                                      # Use defaults
#   ./scripts/convert_g1_data.sh --task-name my_task --epoch 0        # With epoch
#   ./scripts/convert_g1_data.sh --labeling-mode human_labeling       # With advantage labeling
#   ./scripts/convert_g1_data.sh --task-name cabinet_bottle --task-description "Put the bottle into the cabinet and close the drawer" --num-repeats 1
# Environment variables can also be used:
#   TASK_NAME="my_task" EPOCH_NUM=0 ./scripts/convert_g1_data.sh
#
# Environment Notes:
#   - reward_labeling mode uses conda base Python (for Qwen/GPT/RoboDopamine)
#   - action_chunk_advantage mode uses conda base Python (for Qwen/GPT/RoboDopamine)
#   - Other modes use project .venv Python

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
        --reward-method)
            REWARD_METHOD="$2"
            shift 2
            ;;
        --reward-task-instruction)
            REWARD_TASK_INSTRUCTION="$2"
            shift 2
            ;;
        --reward-max-frames)
            REWARD_MAX_FRAMES="$2"
            shift 2
            ;;
        --reward-image-rotation)
            REWARD_IMAGE_ROTATION="$2"
            shift 2
            ;;
        --reward-advantage-threshold)
            REWARD_ADVANTAGE_THRESHOLD="$2"
            shift 2
            ;;
        --reward-random-drop-rate)
            REWARD_RANDOM_DROP_RATE="$2"
            shift 2
            ;;
        --reward-reject-rate)
            REWARD_REJECT_RATE="$2"
            shift 2
            ;;
        --reward-goal-image-path)
            REWARD_GOAL_IMAGE_PATH="$2"
            shift 2
            ;;
        --filter-good-only)
            FILTER_GOOD_ONLY="true"
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default values (can be overridden by environment variables or command line)
TASK_NAME="${TASK_NAME:-cabinet_bottle}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-pick up the bottle}"
EPOCH_NUM="${EPOCH_NUM:-}"  # Empty means no epoch suffix
LABELING_MODE="${LABELING_MODE:-none}"  # Options: none, human_labeling, reward_labeling, action_chunk_advantage
NUM_REPEATS="${NUM_REPEATS:-1}"
CONFIG_NAME="${CONFIG_NAME:-pi05_g1_auto}"

# Reward labeling parameters (used for reward_labeling and action_chunk_advantage modes)
REWARD_METHOD="${REWARD_METHOD:-Ours}"  # Options: Ours, GVL, RoboDopamine
REWARD_TASK_INSTRUCTION="${REWARD_TASK_INSTRUCTION:-}"
REWARD_MAX_FRAMES="${REWARD_MAX_FRAMES:-30}"
REWARD_IMAGE_ROTATION="${REWARD_IMAGE_ROTATION:-0}"
REWARD_ADVANTAGE_THRESHOLD="${REWARD_ADVANTAGE_THRESHOLD:-0.3}"
REWARD_LOOK_AHEAD_WINDOW="${REWARD_LOOK_AHEAD_WINDOW:-80}"
REWARD_RANDOM_DROP_RATE="${REWARD_RANDOM_DROP_RATE:-0.0}"
REWARD_REJECT_RATE="${REWARD_REJECT_RATE:-0.3}"
REWARD_GOAL_IMAGE_PATH="${REWARD_GOAL_IMAGE_PATH:-}"

# Filter good only (for epoch 0, filter out bad rollouts)
FILTER_GOOD_ONLY="${FILTER_GOOD_ONLY:-false}"

# Base directories
BASE_DATA_DIR="${BASE_DATA_DIR:-examples/g1_control_client/g1_data_processed}"
BASE_LEROBOT_DIR="${BASE_LEROBOT_DIR:-examples/g1_control_client/g1_data_lerobot}"

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
echo "  > Converting G1 data to LeRobot format..."
echo "========================================================"
echo "Task name: $TASK_NAME"
echo "Task description: $TASK_DESCRIPTION"
echo "Data directory: $DATA_DIR"
echo "Save directory: $SAVE_DIR"
echo "LeRobot data directory: $LEROBOT_DATA_DIR"
echo "Number of repeats: $NUM_REPEATS"
echo "Labeling mode: $LABELING_MODE"
echo "Config name: $CONFIG_NAME"
echo "Filter good only: $FILTER_GOOD_ONLY"
if [ -n "$EPOCH_NUM" ]; then
    echo "Epoch: $EPOCH_NUM"
fi
echo "Action dim: 29 (G1 fixed: 28 upper body + 1 waist_yaw)"
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

# If using action_chunk_advantage mode, check if we need pre-computation
if [ "$LABELING_MODE" = "action_chunk_advantage" ]; then
    echo ""
    echo "========================================================"
    echo "  > Checking for action chunk advantages..."
    echo "========================================================"
    
    # Check if checkpoint path is set (only required for "Ours" method)
    if [ "$REWARD_METHOD" = "Ours" ]; then
        if [ -z "$QWEN_REWARD_CHECKPOINT_PATH" ]; then
            echo "ERROR: QWEN_REWARD_CHECKPOINT_PATH not set for action_chunk_advantage mode with method='Ours'!"
            echo "Set it with: export QWEN_REWARD_CHECKPOINT_PATH='/path/to/checkpoint'"
            exit 1
        fi
        
        if [ ! -d "$QWEN_REWARD_CHECKPOINT_PATH" ]; then
            echo "ERROR: Checkpoint path does not exist: $QWEN_REWARD_CHECKPOINT_PATH"
            exit 1
        fi
    elif [ "$REWARD_METHOD" = "GVL" ]; then
        # Check for OpenAI API key
        if [ -z "$OPENAI_API_KEY" ]; then
            echo "ERROR: OPENAI_API_KEY not set in environment!"
            echo "Action chunk advantage mode with method='GVL' requires OpenAI API key."
            exit 1
        fi
    elif [ "$REWARD_METHOD" = "RoboDopamine" ]; then
        # Check for goal image path
        if [ -z "$REWARD_GOAL_IMAGE_PATH" ]; then
            echo "ERROR: REWARD_GOAL_IMAGE_PATH not set for action_chunk_advantage mode with method='RoboDopamine'!"
            echo "Pass it with: --reward-goal-image-path '/path/to/goal_image.png'"
            exit 1
        fi
        
        if [ ! -f "$REWARD_GOAL_IMAGE_PATH" ]; then
            echo "ERROR: Goal image path does not exist: $REWARD_GOAL_IMAGE_PATH"
            exit 1
        fi
    fi
    
    # Find parquet data directory (from previous conversion)
    # Structure: examples/g1_control_client/g1_data_lerobot/task_name/epoch_X/data/chunk-000/
    LEROBOT_DATA_DIR="$BASE_LEROBOT_DIR/$TASK_NAME"
    if [ -n "$EPOCH_NUM" ]; then
        LEROBOT_DATA_DIR="$LEROBOT_DATA_DIR/epoch_${EPOCH_NUM}"
    fi
    PARQUET_DIR="$LEROBOT_DATA_DIR/data/chunk-000"
    
    # Check if parquet files exist
    if [ -d "$PARQUET_DIR" ] && [ -n "$(ls -A "$PARQUET_DIR"/*.parquet 2>/dev/null)" ]; then
        echo "Found existing parquet files in: $PARQUET_DIR"
        
        # Check if advantages already computed
        ADVANTAGE_FILES=$(ls "$PARQUET_DIR"/*_action_chunk_advantages.pkl 2>/dev/null | wc -l)
        PARQUET_FILES=$(ls "$PARQUET_DIR"/episode_*.parquet 2>/dev/null | wc -l)
        
        if [ "$ADVANTAGE_FILES" -eq "$PARQUET_FILES" ] && [ "$ADVANTAGE_FILES" -gt 0 ]; then
            echo "✓ Action chunk advantages already computed ($ADVANTAGE_FILES files)"
        else
            echo "Computing action chunk advantages..."
            echo "  Reward method: $REWARD_METHOD"
            echo "  Parquet directory: $PARQUET_DIR"
            if [ "$REWARD_METHOD" = "Ours" ]; then
                echo "  Checkpoint: $QWEN_REWARD_CHECKPOINT_PATH"
            elif [ "$REWARD_METHOD" = "RoboDopamine" ]; then
                echo "  Goal image: $REWARD_GOAL_IMAGE_PATH"
            fi
            echo "  Task instruction: $REWARD_TASK_INSTRUCTION"
            
            # Use conda base environment for reward computation
            # All methods (Ours/GVL/RoboDopamine) need specialized vision models
            REWARD_PYTHON="/home/yuxin/miniconda/bin/python"
            echo "  Using conda base Python: $REWARD_PYTHON"
            
            # Build advantage computation command
            # Note: G1 uses the same compute_action_chunk_advantages.py as H1
            ADV_CMD="$REWARD_PYTHON examples/h1_control_client/compute_action_chunk_advantages.py \
                --data-dir \"$PARQUET_DIR\" \
                --task-instruction \"$REWARD_TASK_INSTRUCTION\" \
                --max-frames \"$REWARD_MAX_FRAMES\" \
                --look-ahead-window \"$REWARD_LOOK_AHEAD_WINDOW\" \
                --advantage-threshold \"$REWARD_ADVANTAGE_THRESHOLD\" \
                --reward-method \"$REWARD_METHOD\""
            
            # Add method-specific parameters
            if [ "$REWARD_METHOD" = "Ours" ]; then
                ADV_CMD="$ADV_CMD --checkpoint-path \"$QWEN_REWARD_CHECKPOINT_PATH\""
            elif [ "$REWARD_METHOD" = "RoboDopamine" ]; then
                ADV_CMD="$ADV_CMD --goal-image-path \"$REWARD_GOAL_IMAGE_PATH\""
            fi
            
            # Run action chunk advantage computation
            eval "$ADV_CMD"
            
            echo "✓ Action chunk advantages computed!"
        fi
    else
        echo "⚠ No parquet files found in: $PARQUET_DIR"
        echo ""
        echo "ACTION REQUIRED:"
        echo "For first-time conversion with action_chunk_advantage mode:"
        echo "  1. First convert data without advantages (use human_labeling or none)"
        echo "  2. Then run compute_action_chunk_advantages.py manually"
        echo "  3. Then re-convert with action_chunk_advantage mode"
        echo ""
        echo "Or simply run this script twice:"
        echo "  First run: Creates parquet files"
        echo "  Second run: Computes advantages and adds them to dataset"
        echo ""
        echo "Proceeding with conversion (advantages will be False for all frames)..."
    fi
    echo ""
fi

# Build the convert command with optional labeling mode
# Use conda base Python for all reward-based modes (Ours/GVL/RoboDopamine need special vision models)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Choose Python environment based on labeling mode
# reward_labeling needs conda base for vision models (Ours, GVL, RoboDopamine)
# action_chunk_advantage uses pre-computed advantages, so it just needs project .venv for lerobot
if [ "$LABELING_MODE" = "reward_labeling" ]; then
    # Use conda base environment for reward computation (all methods use specialized vision models)
    PYTHON_CMD="/home/yuxin/miniconda/bin/python"
    echo "Using conda base Python for reward-based labeling (method=$REWARD_METHOD): $PYTHON_CMD"
else
    # Use project .venv for other modes (including action_chunk_advantage)
    PYTHON_CMD="$PROJECT_ROOT/.venv/bin/python"
fi

CONVERT_CMD="$PYTHON_CMD examples/g1_control_client/convert_g1_data_to_lerobot.py \
    --data_dir \"$DATA_DIR\" \
    --task_description \"$TASK_DESCRIPTION\" \
    --num_repeats $NUM_REPEATS \
    --save_dir \"$SAVE_DIR\""

if [ "$LABELING_MODE" != "none" ]; then
    CONVERT_CMD="$CONVERT_CMD --labeling_mode $LABELING_MODE"
    
    # Add reward labeling parameters if in reward_labeling mode
    if [ "$LABELING_MODE" = "reward_labeling" ]; then
        CONVERT_CMD="$CONVERT_CMD --reward_method $REWARD_METHOD"
        if [ -n "$REWARD_TASK_INSTRUCTION" ]; then
            CONVERT_CMD="$CONVERT_CMD --reward_task_instruction \"$REWARD_TASK_INSTRUCTION\""
        fi
        CONVERT_CMD="$CONVERT_CMD --reward_max_frames $REWARD_MAX_FRAMES"
        CONVERT_CMD="$CONVERT_CMD --reward_image_rotation $REWARD_IMAGE_ROTATION"
        CONVERT_CMD="$CONVERT_CMD --reward_advantage_threshold $REWARD_ADVANTAGE_THRESHOLD"
        # Add goal image path for RoboDopamine
        if [ "$REWARD_METHOD" = "RoboDopamine" ] && [ -n "$REWARD_GOAL_IMAGE_PATH" ]; then
            CONVERT_CMD="$CONVERT_CMD --reward_goal_image_path \"$REWARD_GOAL_IMAGE_PATH\""
        fi
    fi
    
    # Add parameters for action_chunk_advantage mode
    if [ "$LABELING_MODE" = "action_chunk_advantage" ]; then
        CONVERT_CMD="$CONVERT_CMD --reward_method $REWARD_METHOD"
        if [ -n "$REWARD_TASK_INSTRUCTION" ]; then
            CONVERT_CMD="$CONVERT_CMD --reward_task_instruction \"$REWARD_TASK_INSTRUCTION\""
        fi
        # Pass random drop rate (default 0.0 means no dropping)
        CONVERT_CMD="$CONVERT_CMD --reward_random_drop_rate $REWARD_RANDOM_DROP_RATE"
        # Pass reject rate (default 0.3 means reject 30% of bad samples)
        CONVERT_CMD="$CONVERT_CMD --reward_reject_rate $REWARD_REJECT_RATE"
        # Add goal image path for RoboDopamine
        if [ "$REWARD_METHOD" = "RoboDopamine" ] && [ -n "$REWARD_GOAL_IMAGE_PATH" ]; then
            CONVERT_CMD="$CONVERT_CMD --reward_goal_image_path \"$REWARD_GOAL_IMAGE_PATH\""
        fi
        # Note: action chunk advantages are already pre-computed, just pass task instruction
    fi
fi

# Add filter good only flag if set
if [ "$FILTER_GOOD_ONLY" = "true" ]; then
    CONVERT_CMD="$CONVERT_CMD --filter_good_only"
fi

echo ""
echo "Running conversion..."
eval $CONVERT_CMD

echo ""
echo "========================================================"
echo "  > Computing normalization statistics..."
echo "========================================================"

python scripts/compute_norm_stats.py \
    --config-name "$CONFIG_NAME" \
    --data-dir "$LEROBOT_DATA_DIR"

# Fix: compute_norm_stats.py saves to assets dir with nested path
# Move it to the correct location in the data directory
WRONG_STATS_PATH="$(pwd)/assets/$CONFIG_NAME/$LEROBOT_DATA_DIR/norm_stats.json"
CORRECT_STATS_PATH="$LEROBOT_DATA_DIR/norm_stats.json"

if [ -f "$WRONG_STATS_PATH" ] && [ ! -f "$CORRECT_STATS_PATH" ]; then
    echo "Moving norm_stats.json to correct location..."
    mv "$WRONG_STATS_PATH" "$CORRECT_STATS_PATH"
    # Clean up empty nested directories
    rmdir -p "$(dirname "$WRONG_STATS_PATH")" 2>/dev/null || true
fi

echo ""
echo "========================================================"
echo "  > Done!"
echo "========================================================"
echo "Output:"
echo "  LeRobot data: $LEROBOT_DATA_DIR"
echo "  Norm stats: $LEROBOT_DATA_DIR/norm_stats.json"
echo "========================================================"
