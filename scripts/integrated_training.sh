#!/bin/bash
# =============================================================================
# Integrated Training Pipeline for H1 Robot
# =============================================================================
#
# This script reads ALL configuration from training_config.yaml
# Edit that file to configure your training run.
#
# Usage:
#   ./scripts/integrated_training.sh                    # Use default config
#   ./scripts/integrated_training.sh --config my.yaml   # Use custom config
#
# =============================================================================

set -e

# =============================================================================
# Parse Arguments
# =============================================================================
CONFIG_FILE="examples/h1_control_client/training_config.yaml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--config path/to/config.yaml]"
            echo ""
            echo "All configuration is read from training_config.yaml"
            echo "Edit that file to configure your training run."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Setup
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Check for required tools
if ! command -v yq &> /dev/null; then
    echo "ERROR: yq is required but not installed."
    echo "Install with: pip install yq"
    echo "Or: sudo apt install yq"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "ERROR: jq is required but not installed."
    echo "Install with: sudo apt install jq"
    exit 1
fi

# Check config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# =============================================================================
# Read Configuration from YAML
# =============================================================================
echo "Reading configuration from: $CONFIG_FILE"

# Task
TASK_NAME=$(yq -r '.task.name' "$CONFIG_FILE")
TASK_DESCRIPTION=$(yq -r '.task.description' "$CONFIG_FILE")

# Policy
CONFIG_NAME=$(yq -r '.policy.config_name' "$CONFIG_FILE")
BASE_CHECKPOINT=$(yq -r '.policy.base_checkpoint // ""' "$CONFIG_FILE")

# Training
MAX_EPOCHS=$(yq -r '.training.max_epochs // 400' "$CONFIG_FILE")
SAVE_INTERVAL=$(yq -r '.training.save_interval // 200' "$CONFIG_FILE")
KEEP_PERIOD=$(yq -r '.training.keep_period // 100' "$CONFIG_FILE")
NUM_REPEATS=$(yq -r '.training.num_repeats // 1' "$CONFIG_FILE")
LABELING_MODE=$(yq -r '.training.labeling_mode // "human_labeling"' "$CONFIG_FILE")
GPU_ID=$(yq -r '.training.gpu_id // 0' "$CONFIG_FILE")

# Server
SERVER_HOST=$(yq -r '.policy_server.host // "localhost"' "$CONFIG_FILE")
SERVER_PORT=$(yq -r '.policy_server.port // 8000' "$CONFIG_FILE")

# Pipeline
START_PHASE=$(yq -r '.pipeline.start_phase // "data_collection"' "$CONFIG_FILE")
START_EPOCH=$(yq -r '.pipeline.start_epoch // 0' "$CONFIG_FILE")

# Paths
DATA_DIR="$PROJECT_ROOT/data/training_epochs/$TASK_NAME"
STATE_FILE="$PROJECT_ROOT/training_state_${TASK_NAME}.json"
LEROBOT_BASE_DIR="$PROJECT_ROOT/examples/h1_control_client/h1_data_lerobot"
CHECKPOINT_BASE_DIR="$PROJECT_ROOT/checkpoints/$CONFIG_NAME/$TASK_NAME"

# =============================================================================
# Color Codes
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Runtime state
SERVER_PID=""
VIZ_PID=""
EPOCH=0
LAST_CHECKPOINT=""
STATUS="idle"

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_phase() {
    echo ""
    echo -e "${CYAN}========================================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}========================================================${NC}"
}

# =============================================================================
# State Management
# =============================================================================

load_state() {
    if [ -f "$STATE_FILE" ]; then
        log_info "Loading state from $STATE_FILE..."
        EPOCH=$(jq -r '.epoch // 0' "$STATE_FILE")
        LAST_CHECKPOINT=$(jq -r '.last_checkpoint // empty' "$STATE_FILE")
        STATUS=$(jq -r '.status // "idle"' "$STATE_FILE")
        log_info "  Epoch: $EPOCH"
        log_info "  Last checkpoint: ${LAST_CHECKPOINT:-none}"
        log_info "  Status: $STATUS"
        return 0
    else
        log_info "No state file found. Starting fresh."
        return 1
    fi
}

save_state() {
    local status="${1:-$STATUS}"
    STATUS="$status"
    
    mkdir -p "$(dirname "$STATE_FILE")"
    
    cat > "$STATE_FILE" << EOF
{
  "task_name": "$TASK_NAME",
  "task_description": "$TASK_DESCRIPTION",
  "config_name": "$CONFIG_NAME",
  "epoch": $EPOCH,
  "last_checkpoint": "$LAST_CHECKPOINT",
  "base_checkpoint": "$BASE_CHECKPOINT",
  "status": "$status",
  "updated_at": "$(date -Iseconds)"
}
EOF
    log_info "State saved: epoch=$EPOCH, status=$status"
}

# =============================================================================
# Server Management
# =============================================================================

start_server() {
    local checkpoint_dir="$1"
    local data_dir="$2"
    
    log_phase "Starting Policy Server"
    log_info "Checkpoint: $checkpoint_dir"
    log_info "Data dir: $data_dir"
    log_info "Port: $SERVER_PORT"
    
    # Kill any existing server
    stop_server
    
    # Create logs directory
    mkdir -p "$PROJECT_ROOT/logs"
    
    # Build the data dir path for LeRobot format
    local lerobot_data_dir
    if [ -d "$data_dir" ]; then
        lerobot_data_dir="$data_dir"
    else
        # Fallback to base task directory
        lerobot_data_dir="$LEROBOT_BASE_DIR/$TASK_NAME"
    fi
    
    # Start server in background
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    log_info "Starting server..."
    nohup uv run scripts/serve_policy.py \
        --training-epoch "$EPOCH" \
        policy:checkpoint \
        --policy.config="$CONFIG_NAME" \
        --policy.dir="$checkpoint_dir" \
        --policy.data-dir="$lerobot_data_dir" \
        > "$PROJECT_ROOT/logs/server_epoch${EPOCH}.log" 2>&1 &
    SERVER_PID=$!
    
    log_info "Server started with PID $SERVER_PID"
    log_info "Log: $PROJECT_ROOT/logs/server_epoch${EPOCH}.log"
    
    # Wait for server to be ready
    log_info "Waiting for server to be ready..."
    local max_wait=120
    for i in $(seq 1 $max_wait); do
        if curl -s "http://localhost:$SERVER_PORT/health" > /dev/null 2>&1; then
            log_info "Server is ready! (took ${i}s)"
            echo ""
            echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
            echo -e "${GREEN}║  POLICY SERVER RUNNING                                         ║${NC}"
            echo -e "${GREEN}║                                                                ║${NC}"
            echo -e "${GREEN}║  WebSocket: ws://${SERVER_HOST}:${SERVER_PORT}                          ║${NC}"
            echo -e "${GREEN}║  Task: ${TASK_NAME}                                            ║${NC}"
            echo -e "${GREEN}║  Prompt: ${TASK_DESCRIPTION}, Advantage=True                   ║${NC}"
            echo -e "${GREEN}║                                                                ║${NC}"
            echo -e "${GREEN}║  Robot can now connect and collect data                        ║${NC}"
            echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
            echo ""
            
            # Start visualizer after server is ready
            start_visualizer
            
            return 0
        fi
        
        # Check if process died
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log_error "Server process died. Check logs: $PROJECT_ROOT/logs/server_epoch${EPOCH}.log"
            tail -20 "$PROJECT_ROOT/logs/server_epoch${EPOCH}.log"
            return 1
        fi
        
        printf "\r  Waiting... %d/%ds" $i $max_wait
        sleep 1
    done
    
    log_error "Server failed to start within ${max_wait}s"
    return 1
}

stop_server() {
    # Stop visualizer first
    stop_visualizer
    
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log_info "Stopping server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
    
    # Also kill any orphaned servers
    pkill -f "serve_policy.py" 2>/dev/null || true
    sleep 1
}

# =============================================================================
# Visualizer Management
# =============================================================================

start_visualizer() {
    log_info "Starting visualizer on port 8080..."
    
    # Find sample HDF5 data to use for visualization
    local hdf5_file=""
    
    # Try to find any HDF5 file
    if [ -f "$PROJECT_ROOT/training_data/h1/circular.hdf5" ]; then
        hdf5_file="$PROJECT_ROOT/training_data/h1/circular.hdf5"
    else
        # Search for any HDF5 file in training_data
        hdf5_file=$(find "$PROJECT_ROOT/training_data" -name "*.hdf5" -type f 2>/dev/null | head -1)
    fi
    
    if [ -z "$hdf5_file" ]; then
        log_warn "No HDF5 file found for visualization. Visualizer not started."
        log_warn "You can still test the policy manually by running:"
        log_warn "  uv run python examples/h1_control_client/h1_policy_viz_client.py --hdf5-path <path>"
        return 1
    fi
    
    log_info "Using HDF5 data: $hdf5_file"
    
    # Start visualizer in background
    nohup uv run python "$PROJECT_ROOT/examples/h1_control_client/h1_policy_viz_client.py" \
        --hdf5-path "$hdf5_file" \
        --host "localhost" \
        --port "$SERVER_PORT" \
        --prompt "$TASK_DESCRIPTION, Advantage=True" \
        > "$PROJECT_ROOT/logs/visualizer_epoch${EPOCH}.log" 2>&1 &
    VIZ_PID=$!
    
    # Wait for visualizer to start
    sleep 5
    
    if kill -0 "$VIZ_PID" 2>/dev/null; then
        log_info "Visualizer started with PID $VIZ_PID"
        echo ""
        echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${BLUE}║  VISUALIZER RUNNING                                            ║${NC}"
        echo -e "${BLUE}║                                                                ║${NC}"
        echo -e "${BLUE}║  Open in browser: http://localhost:8080                        ║${NC}"
        echo -e "${BLUE}║                                                                ║${NC}"
        echo -e "${BLUE}║  - Use 'Infer' button to test policy                           ║${NC}"
        echo -e "${BLUE}║  - Step through frames with slider                             ║${NC}"
        echo -e "${BLUE}║  - View camera feeds and predicted actions                     ║${NC}"
        echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        return 0
    else
        log_warn "Visualizer failed to start. Check: $PROJECT_ROOT/logs/visualizer_epoch${EPOCH}.log"
        VIZ_PID=""
        return 1
    fi
}

stop_visualizer() {
    if [ -n "$VIZ_PID" ] && kill -0 "$VIZ_PID" 2>/dev/null; then
        log_info "Stopping visualizer (PID $VIZ_PID)..."
        kill "$VIZ_PID" 2>/dev/null || true
        wait "$VIZ_PID" 2>/dev/null || true
        VIZ_PID=""
    fi
    
    # Kill any orphaned visualizers
    pkill -f "h1_policy_viz_client.py" 2>/dev/null || true
}

# =============================================================================
# Data Collection
# =============================================================================

wait_for_data() {
    local epoch_dir="$DATA_DIR/epoch_$EPOCH/raw"
    
    log_phase "Data Collection Phase (Epoch $EPOCH)"
    
    mkdir -p "$epoch_dir"
    
    echo ""
    echo -e "${YELLOW}Waiting for robot to collect data...${NC}"
    echo ""
    echo "Expected data location: $epoch_dir"
    echo ""
    echo "On the robot, run:"
    echo -e "  ${CYAN}python h1_training_client.py --config training_config.yaml${NC}"
    echo ""
    echo "The robot will:"
    echo "  1. Connect to policy server at $SERVER_HOST:$SERVER_PORT"
    echo "  2. Execute policy and record episodes"
    echo "  3. Label episodes as good (g) or bad (b)"
    echo "  4. Rsync data back when done"
    echo ""
    echo -e "Commands:"
    echo -e "  ${GREEN}r${NC} - Manually rsync from robot"
    echo -e "  ${GREEN}c${NC} - Check current file count"
    echo -e "  ${GREEN}d${NC} - Done collecting, proceed to training"
    echo -e "  ${GREEN}q${NC} - Quit pipeline"
    echo ""
    
    local start_count=$(find "$epoch_dir" -name "*.hdf5" 2>/dev/null | wc -l)
    log_info "Current file count: $start_count"
    
    while true; do
        read -t 5 -n 1 key 2>/dev/null || key=""
        
        case "$key" in
            r|R)
                echo ""
                rsync_from_robot
                ;;
            c|C)
                local count=$(find "$epoch_dir" -name "*.hdf5" 2>/dev/null | wc -l)
                echo ""
                log_info "Current file count: $count"
                ;;
            d|D)
                local final_count=$(find "$epoch_dir" -name "*.hdf5" 2>/dev/null | wc -l)
                if [ "$final_count" -eq 0 ]; then
                    log_warn "No data files found! Are you sure you want to proceed? (y/n)"
                    read -n 1 confirm
                    echo ""
                    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
                        continue
                    fi
                fi
                echo ""
                log_info "Proceeding to training with $final_count episodes"
                break
                ;;
            q|Q)
                echo ""
                log_info "Quitting..."
                exit 0
                ;;
        esac
        
        # Check for new files periodically
        local current_count=$(find "$epoch_dir" -name "*.hdf5" 2>/dev/null | wc -l)
        if [ "$current_count" -gt "$start_count" ]; then
            log_info "New data detected: $current_count files"
            start_count=$current_count
        fi
    done
}

rsync_from_robot() {
    local epoch_dir="$DATA_DIR/epoch_$EPOCH/raw"
    
    # Read rsync settings from config
    local rsync_target=$(yq -r '.data.rsync.target // ""' "$CONFIG_FILE")
    
    log_info "Syncing data from robot..."
    
    mkdir -p "$epoch_dir"
    
    # This is rsync FROM robot TO here, so we reverse the typical direction
    # Robot saves to: {robot_save_dir}/{task_name}/epoch_{N}/raw/
    # We pull from robot to local
    log_warn "Note: Rsync from robot not configured. Data should be pushed by robot."
    log_info "Expected local path: $epoch_dir"
    
    local count=$(find "$epoch_dir" -name "*.hdf5" 2>/dev/null | wc -l)
    log_info "Current files: $count"
}

# =============================================================================
# Training Pipeline
# =============================================================================

convert_epoch_data() {
    log_phase "Converting Data (Epoch $EPOCH)"
    
    local raw_dir="$DATA_DIR/epoch_$EPOCH/raw"
    local count=$(find "$raw_dir" -name "*.hdf5" 2>/dev/null | wc -l)
    
    if [ "$count" -eq 0 ]; then
        log_error "No HDF5 files found in $raw_dir"
        return 1
    fi
    
    log_info "Converting $count episodes..."
    
    ./scripts/convert_data.sh \
        --task-name "$TASK_NAME" \
        --task-description "$TASK_DESCRIPTION" \
        --epoch "$EPOCH" \
        --labeling-mode "$LABELING_MODE" \
        --num-repeats "$NUM_REPEATS" \
        --config-name "$CONFIG_NAME" \
        --data-dir "$raw_dir"
}

train_epoch() {
    local base_checkpoint="${1:-}"
    
    log_phase "Training (Epoch $EPOCH)"
    
    local train_args="--task-name $TASK_NAME --epoch $EPOCH --config-name $CONFIG_NAME --gpu $GPU_ID"
    train_args="$train_args --max-epochs $MAX_EPOCHS --save-interval $SAVE_INTERVAL --keep-period $KEEP_PERIOD"
    
    if [ -n "$base_checkpoint" ]; then
        train_args="$train_args --base-checkpoint $base_checkpoint"
    fi
    
    ./scripts/train_h1_local.sh $train_args
    
    # Find the latest checkpoint
    local checkpoint_epoch_dir="$CHECKPOINT_BASE_DIR/epoch_$EPOCH"
    if [ -d "$checkpoint_epoch_dir" ]; then
        LAST_CHECKPOINT=$(ls -d "$checkpoint_epoch_dir"/*/ 2>/dev/null | sort -V | tail -1)
    fi
    
    if [ -z "$LAST_CHECKPOINT" ] || [ ! -d "$LAST_CHECKPOINT" ]; then
        log_error "No checkpoint found after training!"
        return 1
    fi
    
    log_info "Training complete. Checkpoint: $LAST_CHECKPOINT"
}

# =============================================================================
# Main Functions
# =============================================================================

show_config() {
    echo ""
    echo -e "${CYAN}========================================================${NC}"
    echo -e "${CYAN}  Configuration (from $CONFIG_FILE)${NC}"
    echo -e "${CYAN}========================================================${NC}"
    echo ""
    echo -e "  Task Name:        ${GREEN}$TASK_NAME${NC}"
    echo -e "  Task Description: ${GREEN}$TASK_DESCRIPTION${NC}"
    echo -e "  Policy Config:    $CONFIG_NAME"
    echo -e "  Base Checkpoint:  ${BASE_CHECKPOINT:-none}"
    echo -e "  Max Epochs:       $MAX_EPOCHS"
    echo -e "  Save Interval:    $SAVE_INTERVAL"
    echo -e "  Keep Period:      $KEEP_PERIOD"
    echo -e "  Labeling Mode:    $LABELING_MODE"
    echo -e "  GPU:              $GPU_ID"
    echo -e "  Server:           $SERVER_HOST:$SERVER_PORT"
    echo ""
}

show_menu() {
    echo "Select starting phase:"
    echo ""
    echo -e "  ${GREEN}1${NC}) Data Collection - Start server, robot collects data"
    echo -e "  ${GREEN}2${NC}) Training - Convert existing data and train"
    echo -e "  ${GREEN}3${NC}) Resume from state file"
    echo -e "  ${GREEN}q${NC}) Quit"
    echo ""
}

determine_checkpoint() {
    local checkpoint=""
    
    # First, check if we have a checkpoint from a previous epoch
    if [ "$EPOCH" -gt 0 ]; then
        local prev_epoch=$((EPOCH - 1))
        local prev_checkpoint_dir="$CHECKPOINT_BASE_DIR/epoch_$prev_epoch"
        if [ -d "$prev_checkpoint_dir" ]; then
            checkpoint=$(ls -d "$prev_checkpoint_dir"/*/ 2>/dev/null | sort -V | tail -1)
        fi
    fi
    
    # Fall back to LAST_CHECKPOINT from state
    if [ -z "$checkpoint" ] && [ -n "$LAST_CHECKPOINT" ] && [ -d "$LAST_CHECKPOINT" ]; then
        checkpoint="$LAST_CHECKPOINT"
    fi
    
    # Fall back to BASE_CHECKPOINT
    if [ -z "$checkpoint" ] && [ -n "$BASE_CHECKPOINT" ] && [ "$BASE_CHECKPOINT" != "null" ] && [ -d "$BASE_CHECKPOINT" ]; then
        checkpoint="$BASE_CHECKPOINT"
    fi
    
    echo "$checkpoint"
}

determine_data_dir() {
    local data_dir=""
    
    # Try epoch-specific directory first
    if [ "$EPOCH" -gt 0 ]; then
        local prev_epoch=$((EPOCH - 1))
        local prev_data_dir="$LEROBOT_BASE_DIR/$TASK_NAME/epoch_$prev_epoch"
        if [ -d "$prev_data_dir" ] && [ -f "$prev_data_dir/norm_stats.json" ]; then
            data_dir="$prev_data_dir"
        fi
    fi
    
    # Fall back to flat task directory
    if [ -z "$data_dir" ]; then
        local flat_data_dir="$LEROBOT_BASE_DIR/$TASK_NAME"
        if [ -d "$flat_data_dir" ] && [ -f "$flat_data_dir/norm_stats.json" ]; then
            data_dir="$flat_data_dir"
        fi
    fi
    
    # Fall back to move_lid directory (for base checkpoint compatibility)
    if [ -z "$data_dir" ]; then
        local fallback_dir="$LEROBOT_BASE_DIR/move_lid"
        if [ -d "$fallback_dir" ] && [ -f "$fallback_dir/norm_stats.json" ]; then
            data_dir="$fallback_dir"
        fi
    fi
    
    echo "$data_dir"
}

run_data_collection_phase() {
    local checkpoint=$(determine_checkpoint)
    local data_dir=$(determine_data_dir)
    
    if [ -z "$checkpoint" ]; then
        log_error "No checkpoint available for serving!"
        log_error "Set policy.base_checkpoint in $CONFIG_FILE"
        exit 1
    fi
    
    if [ -z "$data_dir" ]; then
        log_warn "No LeRobot data directory found. Using checkpoint directory for norm_stats."
        data_dir="$(dirname "$checkpoint")"
    fi
    
    save_state "collecting_data"
    
    # Start server
    if ! start_server "$checkpoint" "$data_dir"; then
        log_error "Failed to start server"
        exit 1
    fi
    
    # Wait for data
    wait_for_data
    
    # Stop server
    stop_server
}

run_training_phase() {
    local checkpoint=$(determine_checkpoint)
    
    save_state "converting"
    convert_epoch_data
    
    save_state "training"
    train_epoch "$checkpoint"
    
    save_state "epoch_complete"
}

cleanup() {
    echo ""
    log_info "Cleaning up..."
    stop_server
    if [ -n "$STATUS" ] && [ "$STATUS" != "idle" ]; then
        save_state "interrupted"
    fi
    log_info "Cleanup complete."
}

trap cleanup EXIT

main() {
    # Show configuration
    show_config
    
    # Create directories
    mkdir -p "$DATA_DIR"
    mkdir -p "$PROJECT_ROOT/logs"
    
    # Determine phase
    local phase="$START_PHASE"
    EPOCH=$START_EPOCH
    
    # Check for existing state
    if load_state 2>/dev/null; then
        echo ""
        log_info "Found existing state file."
        echo ""
        show_menu
        read -p "Enter choice (or press Enter to resume): " choice
        
        case "$choice" in
            1)
                phase="data_collection"
                ;;
            2)
                phase="training"
                ;;
            3|"")
                # Resume from state
                case "$STATUS" in
                    collecting_data|idle)
                        phase="data_collection"
                        ;;
                    converting|training|epoch_complete)
                        phase="training"
                        ;;
                    *)
                        phase="data_collection"
                        ;;
                esac
                ;;
            q|Q)
                exit 0
                ;;
            *)
                log_error "Invalid choice"
                exit 1
                ;;
        esac
    else
        # No state file - show menu
        show_menu
        read -p "Enter choice: " choice
        
        case "$choice" in
            1)
                phase="data_collection"
                ;;
            2)
                phase="training"
                ;;
            q|Q)
                exit 0
                ;;
            *)
                log_error "Invalid choice"
                exit 1
                ;;
        esac
    fi
    
    log_info "Starting in phase: $phase, epoch: $EPOCH"
    
    # Main loop
    while [ "$EPOCH" -lt "$MAX_EPOCHS" ]; do
        log_phase "EPOCH $EPOCH"
        
        if [ "$phase" = "data_collection" ]; then
            run_data_collection_phase
            phase="training"
        fi
        
        if [ "$phase" = "training" ]; then
            run_training_phase
            EPOCH=$((EPOCH + 1))
            phase="data_collection"
        fi
    done
    
    log_phase "Training Complete!"
    log_info "Completed $EPOCH epochs"
    log_info "Final checkpoint: $LAST_CHECKPOINT"
    save_state "complete"
}

main "$@"
