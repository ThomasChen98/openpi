#!/bin/bash
# Convenience script to run H1 policy inference visualization

set -e

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

# Default values
DATA_PATH="${DATA_PATH:-h1_data_processed/fold_cloth_with_both_hands_corrected/episode_01.hdf5}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
PROMPT="${PROMPT:-fold the cloth with both hands}"

POLICY_CONFIG="${POLICY_CONFIG:-pi05_h1_auto}"
POLICY_DIR="${POLICY_DIR:-checkpoints/pi05_h1_auto/fold_cloth_with_both_hands_corrected_delta14/1500}"

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}H1 Policy Inference Visualization Client${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Data Path: ${DATA_PATH}"
echo -e "  Policy Server: ${HOST}:${PORT}"
echo -e "  Task Prompt: ${PROMPT}"
echo ""
echo -e "${YELLOW}Note: Make sure the policy server is running first!${NC}"
echo -e "${YELLOW}Start server with:${NC}"
echo -e "  uv run scripts/serve_policy.py policy:checkpoint \\"
echo -e "    --policy.config=${POLICY_CONFIG} \\"
echo -e "    --policy.dir=${POLICY_DIR}"
echo ""
echo -e "${GREEN}Starting visualization client...${NC}"
echo ""

# Run the client
python examples/h1_control_client/h1_policy_viz_client.py \
    --data-path "$DATA_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --prompt "$PROMPT" \
    --robot-execution \
    "$@"

