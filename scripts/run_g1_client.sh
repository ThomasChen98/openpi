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
#   MODE           - Operating mode (see below) (default: viz)
#   ROBOT_IP       - G1 robot IP for camera (default: 192.168.123.164)
#   ROBOT_HOST     - Robot client host for viz+robot mode (default: localhost)
#   ROBOT_PORT     - Port for robot client / listen mode (default: 5007)
#   VISER_PORT     - Viser web UI port (default: 8080)
#
# Modes:
#   viz          - Visualization only (replay HDF5 in browser)
#   viz-robot    - Visualization with robot execution enabled (best for testing)
#   robot        - Direct robot control (connect to policy server, execute on robot)
#   listen       - Robot listens for commands from viz client (run on robot station)
#
# Examples:
#   # Visualization only (default)
#   ./scripts/run_g1_client.sh
#
#   # Visualization with robot execution (most useful for testing policies)
#   MODE=viz-robot ./scripts/run_g1_client.sh
#
#   # Connect to remote policy server for viz
#   HOST=192.168.1.100 MODE=viz-robot ./scripts/run_g1_client.sh
#
#   # Robot in listen mode (run this on the robot station computer)
#   MODE=listen ./scripts/run_g1_client.sh
#
#   # Direct robot execution (bypasses viz, runs policy in a loop)
#   MODE=robot HOST=192.168.1.100 ./scripts/run_g1_client.sh

set -e

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

# Default values
DATA_PATH="${DATA_PATH:-/mnt/ssd1/yuxin/g1_data/cabinetbottle/episode_2.hdf5}"
HOST="${HOST:-localhost}"
PORT="${PORT:-8001}"
PROMPT="${PROMPT:-pick up the bottle, put it in the cabinet, and then push to close the cabinet drawer}"
MODE="${MODE:-viz-robot}"
ROBOT_IP="${ROBOT_IP:-192.168.123.164}"
ROBOT_HOST="${ROBOT_HOST:-localhost}"
ROBOT_PORT="${ROBOT_PORT:-5008}"
VISER_PORT="${VISER_PORT:-8081}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_DIR="$(dirname "$SCRIPT_DIR")"
G1_CLIENT_DIR="${OPENPI_DIR}/examples/g1_control_client"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}G1 Policy Client${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Mode: ${CYAN}${MODE}${NC}"
echo -e "  Policy Server: ${HOST}:${PORT}"
echo -e "  Task Prompt: ${PROMPT}"

case "$MODE" in
    viz)
        echo -e "  Data Path: ${DATA_PATH}"
        echo -e "  Viser Port: ${VISER_PORT}"
        ;;
    viz-robot)
        echo -e "  Data Path: ${DATA_PATH}"
        echo -e "  Robot Client: ${ROBOT_HOST}:${ROBOT_PORT}"
        echo -e "  Viser Port: ${VISER_PORT}"
        ;;
    robot)
        echo -e "  Robot IP: ${ROBOT_IP}"
        ;;
    listen)
        echo -e "  Robot IP: ${ROBOT_IP}"
        echo -e "  Listen Port: ${ROBOT_PORT}"
        ;;
esac
echo ""

cd "$G1_CLIENT_DIR"

case "$MODE" in
    viz)
        echo -e "${YELLOW}Note: Visualization only mode (no robot execution)${NC}"
        echo -e "${YELLOW}For robot execution, use: MODE=viz-robot ./scripts/run_g1_client.sh${NC}"
        echo ""
        echo -e "${GREEN}Starting visualization client...${NC}"
        echo -e "${GREEN}Open http://localhost:${VISER_PORT} in browser to view${NC}"
        echo ""
        
        python g1_policy_viz_client.py \
            --data-path "$DATA_PATH" \
            --host "$HOST" \
            --port "$PORT" \
            --prompt "$PROMPT" \
            --viser-port "$VISER_PORT" \
            "$@"
        ;;
        
    viz-robot)
        echo -e "${YELLOW}============================================${NC}"
        echo -e "${YELLOW}VISUALIZATION + ROBOT EXECUTION MODE${NC}"
        echo -e "${YELLOW}============================================${NC}"
        echo ""
        echo -e "${YELLOW}Prerequisites (2 terminals):${NC}"
        echo -e "  ${CYAN}[Robot Station]${NC}"
        echo -e "    1. Start image server: python3 image_server.py --camera-id 0"
        echo -e "    2. Start robot client: MODE=listen ./scripts/run_g1_client.sh"
        echo ""
        echo -e "  ${CYAN}[This Machine]${NC}"
        echo -e "    3. Start policy server: ./scripts/run_g1_server.sh"
        echo -e "    4. (This script - already running)"
        echo ""
        echo -e "${GREEN}Starting visualization client with robot execution...${NC}"
        echo -e "${GREEN}Open http://localhost:${VISER_PORT} in browser${NC}"
        echo ""
        echo -e "${CYAN}GUI Features:${NC}"
        echo -e "  • 🔄 Reset Robot to Frame - move robot to HDF5 frame position"
        echo -e "  • 🚀 Execute on Robot - run predicted actions on real robot"
        echo -e "  • 🎮 Replay Teleop Data - replay raw HDF5 joint commands"
        echo -e "  • 📷 Use Live Cameras - toggle robot camera feed"
        echo -e "  • ⏯️ Single Step - infer + execute once"
        echo -e "  • 🔄 Start Continuous - continuous inference loop"
        echo -e "  • 🛑 EMERGENCY STOP - stop all motion"
        echo ""
        
        python g1_policy_viz_client.py \
            --data-path "$DATA_PATH" \
            --host "$HOST" \
            --port "$PORT" \
            --prompt "$PROMPT" \
            --viser-port "$VISER_PORT" \
            --robot-execution \
            --robot-host "$ROBOT_HOST" \
            --robot-port "$ROBOT_PORT" \
            "$@"
        ;;
        
    robot)
        echo -e "${YELLOW}Prerequisites:${NC}"
        echo -e "  1. Policy server running: ./scripts/run_g1_server.sh"
        echo -e "  2. Image server on robot: python3 image_server.py --camera-id 0"
        echo ""
        echo -e "${GREEN}Starting direct robot execution client...${NC}"
        echo -e "${YELLOW}This mode runs inference in a loop - be careful!${NC}"
        echo ""
        
        python g1_remote_client.py \
            --server-host "$HOST" \
            --server-port "$PORT" \
            --head-camera-server-ip "$ROBOT_IP" \
            --prompt "$PROMPT" \
            "$@"
        ;;
        
    listen)
        echo -e "${YELLOW}============================================${NC}"
        echo -e "${YELLOW}ROBOT LISTEN MODE${NC}"
        echo -e "${YELLOW}============================================${NC}"
        echo ""
        echo -e "${CYAN}Run this on the robot station computer!${NC}"
        echo ""
        echo -e "${YELLOW}Prerequisites:${NC}"
        echo -e "  1. Image server running: python3 image_server.py --camera-id 0"
        echo ""
        echo -e "${GREEN}Starting robot client in listen mode...${NC}"
        echo -e "${GREEN}Listening on port ${ROBOT_PORT} for commands from viz client${NC}"
        echo ""
        echo -e "${CYAN}Supported commands from viz client:${NC}"
        echo -e "  • ping - check connection"
        echo -e "  • reset - move to target joint positions"
        echo -e "  • execute - execute action chunk"
        echo -e "  • get_state - get current joint positions"
        echo -e "  • get_observation - get state + camera image"
        echo -e "  • emergency_stop - stop all motion"
        echo ""
        
        python g1_remote_client.py \
            --listen-mode \
            --listen-port "$ROBOT_PORT" \
            --head-camera-server-ip "$ROBOT_IP" \
            --prompt "$PROMPT" \
            "$@"
        ;;
        
    *)
        echo -e "${RED}Unknown mode: ${MODE}${NC}"
        echo ""
        echo -e "Valid modes:"
        echo -e "  ${CYAN}viz${NC}        - Visualization only (replay HDF5 in browser)"
        echo -e "  ${CYAN}viz-robot${NC}  - Visualization with robot execution (recommended for testing)"
        echo -e "  ${CYAN}robot${NC}      - Direct robot control (continuous inference loop)"
        echo -e "  ${CYAN}listen${NC}     - Robot listens for commands (run on robot station)"
        exit 1
        ;;
esac
