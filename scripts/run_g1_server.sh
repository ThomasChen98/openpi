#!/bin/bash
# G1 Policy Server - serves G1 policy for inference
#
# Usage:
#   ./scripts/run_g1_server.sh
#
# Environment variables (optional):
#   POLICY_CONFIG  - Training config name (default: pi05_g1_auto or pi05_h1_auto for testing)
#   POLICY_DIR     - Checkpoint directory
#   DATA_DIR       - Dataset name in g1_data_lerobot/
#   PORT           - Server port (default: 8000)
#
# Examples:
#   # Use defaults (H1 checkpoint for testing)
#   ./scripts/run_g1_server.sh
#
#   # Use G1 checkpoint (when trained)
#   POLICY_CONFIG=pi05_g1_auto POLICY_DIR=checkpoints/pi05_g1_auto/my_task/999 DATA_DIR=/mnt/ssd1/yuxin/g1_data/my_task ./scripts/run_g1_server.sh
#
#   # Different port
#   PORT=8001 ./scripts/run_g1_server.sh

set -e

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
# Note: Using H1 checkpoint for testing until G1 checkpoints are trained
# Set POLICY_CONFIG=pi05_g1_auto to use G1-specific config
POLICY_CONFIG="${POLICY_CONFIG:-pi05_g1_auto}"
PORT="${PORT:-8001}"

# Set data and checkpoint directories based on config
if [[ "$POLICY_CONFIG" == "pi05_g1_auto" ]]; then
    # G1 specific paths
    POLICY_DIR="${POLICY_DIR:-checkpoints/pi05_g1_auto/insert_plate_jan16/4999}"
    DATA_DIR="${DATA_DIR:-${OPENPI_DIR}/examples/g1_control_client/g1_data_lerobot/insert_plate_jan16}"
else
    # H1 paths (for testing)
    POLICY_DIR="${POLICY_DIR:-checkpoints/pi05_h1_auto/fold_towel_reward_2/epoch_0/1499}"
    DATA_DIR="${DATA_DIR:-${OPENPI_DIR}/examples/h1_control_client/h1_data_lerobot/fold_towel}"
fi

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}G1 Policy Server${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Policy Config: ${CYAN}${POLICY_CONFIG}${NC}"
echo -e "  Checkpoint: ${POLICY_DIR}"
echo -e "  Data Dir: ${DATA_DIR}"
echo -e "  Port: ${PORT}"
echo ""

if [[ "$POLICY_CONFIG" == "pi05_g1_auto" ]]; then
    echo -e "${CYAN}G1 Configuration:${NC}"
    echo -e "  State Space (29 dims):"
    echo -e "    [0:14]  arm joints"
    echo -e "    [14:28] Dex3 hand joints"
    echo -e "    [28]    waist_yaw"
    echo ""
    echo -e "  Action Space (29 dims):"
    echo -e "    [0:14]  arm joint targets"
    echo -e "    [14:28] Dex3 hand joint targets"
    echo -e "    [28]    waist_yaw target"
else
    echo -e "${YELLOW}  Using H1 config for testing${NC}"
    echo -e "    H1 has 26-dim action space (arms + Inspire hands)"
    echo -e "    G1 has 29-dim action space (arms + Dex3 hands + waist_yaw)"
    echo ""
    echo -e "${YELLOW}To use G1 config, set:${NC}"
    echo -e "  POLICY_CONFIG=pi05_g1_auto ./scripts/run_g1_server.sh"
fi
echo ""
echo -e "${GREEN}Starting policy server...${NC}"
echo ""

cd "$OPENPI_DIR"

# Note: --port must come BEFORE the policy:checkpoint subcommand
# Optional: POLICY_ACTION_DIM=16|28|29 to match training (default: omit = config default 29)
EXTRA_ACTION_DIM=()
if [ -n "${POLICY_ACTION_DIM:-}" ]; then
    EXTRA_ACTION_DIM=(--policy.action-dim="${POLICY_ACTION_DIM}")
fi

uv run scripts/serve_policy.py \
    --port="${PORT}" \
    policy:checkpoint \
    --policy.config="${POLICY_CONFIG}" \
    --policy.dir="${POLICY_DIR}" \
    --policy.data-dir="${DATA_DIR}" \
    "${EXTRA_ACTION_DIM[@]}"
