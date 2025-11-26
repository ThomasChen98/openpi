export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
EXP_NAME="pi05_h1_move_lid"
CONFIG_NAME="pi05_h1_auto"
DATA_DIR="move_lid"
LEROBOT_DATA_DIR="$(pwd)/examples/h1_control_client/h1_data_lerobot/$DATA_DIR"

echo "========================================================"
echo "  > Training H1 with local dataset..."
echo "========================================================"
echo "  > Config name: $CONFIG_NAME"
echo "  > Exp name: $EXP_NAME"
echo "  > Data directory: $DATA_DIR"
echo "  > LeRobot data directory: $LEROBOT_DATA_DIR"
echo "========================================================"

# Run training
uv run scripts/train_auto.py \
    --config-name "$CONFIG_NAME" \
    --exp-name "$EXP_NAME" \
    --overwrite \
    --data-dir "$LEROBOT_DATA_DIR"
