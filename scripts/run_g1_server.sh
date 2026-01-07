#!/bin/bash
# G1 Policy Server - serves G1 policy for inference
#
# Usage:
#   ./scripts/run_g1_server.sh
#
# Environment variables (optional):
#   POLICY_CONFIG  - Training config name (default: pi05_g1_auto)
#   POLICY_DIR     - Checkpoint directory (default: checkpoints/pi05_g1_auto/example/1199)
#   DATA_DIR       - Dataset name in g1_data_lerobot/ (default: cabinetbottle)
#   PORT           - Server port (default: 8000)
#
# Examples:
#   # Use defaults
#   ./scripts/run_g1_server.sh
#
#   # Custom checkpoint
#   POLICY_DIR=checkpoints/pi05_g1_auto/my_task/999 DATA_DIR=my_task ./scripts/run_g1_server.sh
#
#   # Different port
#   PORT=8001 ./scripts/run_g1_server.sh

set -e

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

# Default values
POLICY_CONFIG="${POLICY_CONFIG:-pi05_g1_auto}"
POLICY_DIR="${POLICY_DIR:-checkpoints/pi05_g1_auto/example/1199}"
DATA_DIR="${DATA_DIR:-cabinetbottle}"
PORT="${PORT:-8000}"

# Construct absolute path for data directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_DIR="$(dirname "$SCRIPT_DIR")"
LEROBOT_DATA_DIR="${OPENPI_DIR}/examples/g1_control_client/g1_data_lerobot/${DATA_DIR}"

# Also check /mnt/ssd1 location as fallback
if [[ ! -d "$LEROBOT_DATA_DIR" ]] && [[ -d "/mnt/ssd1/yuxin/g1_data/${DATA_DIR}" ]]; then
    LEROBOT_DATA_DIR="/mnt/ssd1/yuxin/g1_data/${DATA_DIR}"
fi

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}G1 Policy Server${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Policy Config: ${POLICY_CONFIG}"
echo -e "  Checkpoint: ${POLICY_DIR}"
echo -e "  Data Dir: ${LEROBOT_DATA_DIR}"
echo -e "  Port: ${PORT}"
echo ""
echo -e "${YELLOW}G1 Action Space (32 dims):${NC}"
echo -e "  [0:28]  Upper body (14 arm + 14 Dex3 hand)"
echo -e "  [28]    vx (forward/backward)"
echo -e "  [29]    vy (strafe)"
echo -e "  [30]    vyaw (turn)"
echo -e "  [31]    padding"
echo ""
echo -e "${GREEN}Starting policy server...${NC}"
echo ""

cd "$OPENPI_DIR"

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config="${POLICY_CONFIG}" \
    --policy.dir="${POLICY_DIR}" \
    --policy.data-dir="${LEROBOT_DATA_DIR}" \
    --port="${PORT}"
