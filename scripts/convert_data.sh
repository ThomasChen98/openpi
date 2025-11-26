DATA_DIR="examples/h1_control_client/h1_data_processed/box_action/good/"
TASK_DESCRIPTION="move the lid of the box and put it on the table"
NUM_REPEATS=3
SAVE_DIR="move_lid"
CONFIG_NAME="pi05_h1_auto"
# Construct the output data directory path (where LeRobot format data will be saved)
# Use absolute path for LeRobot compatibility
LEROBOT_DATA_DIR="$(pwd)/examples/h1_control_client/h1_data_lerobot/$SAVE_DIR"

echo "========================================================"
echo "  > Converting data to LeRobot format..."
echo "========================================================"
echo "Data directory: $DATA_DIR"
echo "Task description: $TASK_DESCRIPTION"
echo "Number of repeats: $NUM_REPEATS"
echo "Save directory: $SAVE_DIR"
echo "LeRobot data directory: $LEROBOT_DATA_DIR"
echo "Config name: $CONFIG_NAME"
echo "========================================================"

uv run examples/h1_control_client/convert_h1_data_to_lerobot.py \
--data_dir "$DATA_DIR" \
--task_description "$TASK_DESCRIPTION" \
--num_repeats $NUM_REPEATS \
--save_dir "$SAVE_DIR"

echo "========================================================"
echo "  > Computing normalization statistics..."
echo "========================================================"

uv run scripts/compute_norm_stats.py \
--config-name "$CONFIG_NAME" \
--data-dir "$LEROBOT_DATA_DIR"

echo "========================================================"
echo "  > Done!"
echo "========================================================"