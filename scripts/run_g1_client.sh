#!/bin/bash
# G1 Policy Client - visualization and robot execution
#
# Usage:
#   ./scripts/run_g1_client.sh [OPTIONS]
#
# Environment variables (optional):
#   DATA_PATH      - HDF5 episode file for visualization (default: g1_data/cabinetbottle/episode_2.hdf5)
#   HOST           - Policy server hostname/IP (default: localhost)
#   PORT           - Policy server port (default: 8000)
#   PROMPT         - Task prompt (default: "pick up the bottle and put it in the cabinet")
#   MODE           - "viz" for visualization only, "robot" for robot execution (default: viz)
#   ROBOT_IP       - G1 robot IP for camera (default: 192.168.123.164)
#   LISTEN_PORT    - Port for listen mode (default: 5007)
#
# Examples:
#   # Visualization only (default)
#   ./scripts/run_g1_client.sh
#
#   # Connect to remote policy server
#   HOST=192.168.1.100 ./scripts/run_g1_client.sh
#
#   # Robot execution mode (connect to policy server and execute on robot)
#   MODE=robot HOST=192.168.1.100 ./scripts/run_g1_client.sh
#
#   # Listen mode (wait for viz client commands)
#   MODE=listen ./scripts/run_g1_client.sh

set -e

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

# Default values
DATA_PATH="${DATA_PATH:-/mnt/ssd1/yuxin/g1_data/cabinetbottle/episode_2.hdf5}"
HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
PROMPT="${PROMPT:-pick up the bottle and put it in the cabinet}"
MODE="${MODE:-viz}"
ROBOT_IP="${ROBOT_IP:-192.168.123.164}"
LISTEN_PORT="${LISTEN_PORT:-5007}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_DIR="$(dirname "$SCRIPT_DIR")"
G1_CLIENT_DIR="${OPENPI_DIR}/examples/g1_control_client"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}G1 Policy Client${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Mode: ${MODE}"
echo -e "  Policy Server: ${HOST}:${PORT}"
echo -e "  Task Prompt: ${PROMPT}"
if [[ "$MODE" == "viz" ]]; then
    echo -e "  Data Path: ${DATA_PATH}"
fi
if [[ "$MODE" == "robot" ]] || [[ "$MODE" == "listen" ]]; then
    echo -e "  Robot IP: ${ROBOT_IP}"
fi
echo ""

cd "$G1_CLIENT_DIR"

case "$MODE" in
    viz)
        echo -e "${YELLOW}Note: Make sure the policy server is running first!${NC}"
        echo -e "${YELLOW}Start server with: ./scripts/run_g1_server.sh${NC}"
        echo ""
        echo -e "${GREEN}Starting visualization client...${NC}"
        echo -e "${GREEN}Open http://localhost:8080 in browser to view${NC}"
        echo ""
        
        python g1_policy_viz_client.py \
            --data-path "$DATA_PATH" \
            --host "$HOST" \
            --port "$PORT" \
            --prompt "$PROMPT" \
            "$@"
        ;;
        
    robot)
        echo -e "${YELLOW}Prerequisites:${NC}"
        echo -e "  1. Policy server running: ./scripts/run_g1_server.sh"
        echo -e "  2. Image server on robot: python3 image_server.py --camera-id 0"
        echo ""
        echo -e "${GREEN}Starting robot execution client...${NC}"
        echo ""
        
        python g1_remote_client.py \
            --server-host "$HOST" \
            --server-port "$PORT" \
            --head-camera-server-ip "$ROBOT_IP" \
            --prompt "$PROMPT" \
            "$@"
        ;;
        
    listen)
        echo -e "${YELLOW}Listen mode: Waiting for commands from viz client${NC}"
        echo -e "${YELLOW}Prerequisites:${NC}"
        echo -e "  1. Image server on robot: python3 image_server.py --camera-id 0"
        echo ""
        echo -e "${GREEN}Starting in listen mode on port ${LISTEN_PORT}...${NC}"
        echo ""
        
        python g1_remote_client.py \
            --listen-mode \
            --listen-port "$LISTEN_PORT" \
            --head-camera-server-ip "$ROBOT_IP" \
            --prompt "$PROMPT" \
            "$@"
        ;;
        
    *)
        echo -e "${RED}Unknown mode: ${MODE}${NC}"
        echo -e "Valid modes: viz, robot, listen"
        exit 1
        ;;
esac
