export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

POLICY_CONFIG="${POLICY_CONFIG:-pi05_h1_auto}"
POLICY_DIR="${POLICY_DIR:-checkpoints/pi05_h1_auto/insert_bottle/1199}"
DATA_DIR="${DATA_DIR:-insert_bottle}"
# Construct absolute path for data directory
LEROBOT_DATA_DIR="$(pwd)/examples/h1_control_client/h1_data_lerobot/$DATA_DIR"

uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=${POLICY_CONFIG} \
  --policy.dir=${POLICY_DIR} \
  --policy.data-dir="${LEROBOT_DATA_DIR}"