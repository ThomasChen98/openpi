#!/bin/bash
# Convenience script to run H1 policy inference visualization

set -e

# Default values
HDF5_PATH="${HDF5_PATH:-h1_data_processed/box_action/good/episode_2.hdf5}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
PROMPT="${PROMPT:-h1 control task}" # check if this is right in case of overfitting

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
echo -e "  HDF5 Dataset: ${HDF5_PATH}"
echo -e "  Policy Server: ${HOST}:${PORT}"
echo -e "  Task Prompt: ${PROMPT}"
echo ""
echo -e "${YELLOW}Note: Make sure the policy server is running first!${NC}"
echo -e "${YELLOW}Start server with:${NC}"
echo -e "  uv run scripts/serve_policy.py policy:checkpoint \\"
echo -e "    --policy.config=pi05_h1_finetune \\"
echo -e "    --policy.dir=checkpoints/pi05_h1_finetune/pi05_h1_box_activate_h50_short/999"
echo ""
echo -e "${GREEN}Starting visualization client...${NC}"
echo ""

# Run the client
python h1_policy_viz_client.py \
    --hdf5-path "$HDF5_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --prompt "$PROMPT" \
    "$@"

