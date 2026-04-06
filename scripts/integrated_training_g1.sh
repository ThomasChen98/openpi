#!/bin/bash
# =============================================================================
# Integrated Training Pipeline for G1 Robot
# =============================================================================
#
# This script reads ALL configuration from training_config_g1.yaml
# Edit that file to configure your training run.
#
# Key differences from H1 pipeline:
#   - Port 8001 for policy server (H1 uses 8000)
#   - Port 8081 for Viser visualizer (H1 uses 8080)
#   - Policy action width from YAML policy_server.action_dim: 29 (default), 28, or 16 (binary grippers)
#   - G1-specific data paths and conversion
#   - Single head camera (no wrist cameras)
#
# Usage:
#   ./scripts/integrated_training_g1.sh                       # Use default config
#   ./scripts/integrated_training_g1.sh --config my.yaml      # Use custom config
#
# Environment Notes:
#   - All reward computations use conda base environment (/home/yuxin/miniconda/bin/python)
#     This environment has specialized vision models (Qwen3VL, DINOv3, RoboDopamine)
#   - Policy training uses project .venv environment
#   - The script automatically switches between environments as needed
#
# =============================================================================

set -e

# =============================================================================
# Parse Arguments
# =============================================================================
CONFIG_FILE="examples/g1_control_client/training_config_g1.yaml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--config path/to/config.yaml]"
            echo ""
            echo "All configuration is read from training_config_g1.yaml"
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
WARMUP_CHECKPOINT=$(yq -r '.policy.warmup_checkpoint // ""' "$CONFIG_FILE")

# Training
MAX_EPOCHS=$(yq -r '.training.max_epochs // 1000' "$CONFIG_FILE")
SAVE_INTERVAL=$(yq -r '.training.save_interval // 1000' "$CONFIG_FILE")
KEEP_PERIOD=$(yq -r '.training.keep_period // 1000' "$CONFIG_FILE")
NUM_REPEATS=$(yq -r '.training.num_repeats // 2' "$CONFIG_FILE")
LABELING_MODE=$(yq -r '.training.labeling_mode // "human_labeling"' "$CONFIG_FILE")
GPU_ID=$(yq -r '.training.gpu_id // 0' "$CONFIG_FILE")

# Reward labeling (used for reward_labeling and action_chunk_advantage modes)
REWARD_METHOD=$(yq -r '.reward.method // "Ours"' "$CONFIG_FILE")
REWARD_TASK_INSTRUCTION=$(yq -r '.reward.task_instruction // ""' "$CONFIG_FILE")
REWARD_MAX_FRAMES=$(yq -r '.reward.max_frames // 30' "$CONFIG_FILE")
REWARD_IMAGE_ROTATION=$(yq -r '.reward.image_rotation // 0' "$CONFIG_FILE")
REWARD_ADVANTAGE_THRESHOLD=$(yq -r '.reward.advantage_threshold // 0.3' "$CONFIG_FILE")
REWARD_LOOK_AHEAD_WINDOW=$(yq -r '.reward.look_ahead_window // 80' "$CONFIG_FILE")
REWARD_DISTANCE_THRESHOLD=$(yq -r '.reward.distance_threshold // 0.45' "$CONFIG_FILE")
REWARD_CHECKPOINT_PATH=$(yq -r '.reward.checkpoint_path // ""' "$CONFIG_FILE")
REWARD_RANDOM_DROP_RATE=$(yq -r '.reward.random_drop_rate // 0.0' "$CONFIG_FILE")
REWARD_REJECT_RATE=$(yq -r '.reward.reject_rate // 0.3' "$CONFIG_FILE")
REWARD_GOAL_IMAGE_PATH=$(yq -r '.reward.goal_image_path // ""' "$CONFIG_FILE")

# Server (G1 uses different ports from H1)
SERVER_HOST=$(yq -r '.policy_server.host // "localhost"' "$CONFIG_FILE")
SERVER_PORT=$(yq -r '.policy_server.port // 8001' "$CONFIG_FILE")
ACTION_DIM=$(yq -r '.policy_server.action_dim // 29' "$CONFIG_FILE")
case "$ACTION_DIM" in
    16|28|29) ;;
    *)
        echo "ERROR: policy_server.action_dim must be 16, 28, or 29 (got: $ACTION_DIM) in $CONFIG_FILE"
        exit 1
        ;;
esac

# Visualization (G1 uses different ports from H1)
VISER_PORT=$(yq -r '.visualization.viser_port // 8081' "$CONFIG_FILE")
ROBOT_COMMAND_PORT=$(yq -r '.visualization.robot_command_port // 5008' "$CONFIG_FILE")

# Pipeline
START_PHASE=$(yq -r '.pipeline.start_phase // "data_collection"' "$CONFIG_FILE")
START_EPOCH=$(yq -r '.pipeline.start_epoch // 0' "$CONFIG_FILE")

# Paths (G1-specific)
DATA_DIR="$PROJECT_ROOT/examples/g1_control_client/g1_data_auto/$TASK_NAME"
STATE_FILE="$PROJECT_ROOT/training_state_g1_${TASK_NAME}.json"
LEROBOT_BASE_DIR="$PROJECT_ROOT/examples/g1_control_client/g1_data_lerobot"
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
  "robot": "G1",
  "epoch": $EPOCH,
  "last_checkpoint": "$LAST_CHECKPOINT",
  "warmup_checkpoint": "$WARMUP_CHECKPOINT",
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
    
    log_phase "Starting G1 Policy Server"
    log_info "Checkpoint: $checkpoint_dir"
    log_info "Data dir: $data_dir"
    log_info "Port: $SERVER_PORT (G1)"
    
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
    log_info "Policy action_dim: $ACTION_DIM (from YAML; passed to serve_policy and train_g1_local)"
    nohup uv run scripts/serve_policy.py \
        --port "$SERVER_PORT" \
        --training-epoch "$EPOCH" \
        policy:checkpoint \
        --policy.config="$CONFIG_NAME" \
        --policy.dir="$checkpoint_dir" \
        --policy.data-dir="$lerobot_data_dir" \
        --policy.action-dim="$ACTION_DIM" \
        > "$PROJECT_ROOT/logs/g1_server_epoch${EPOCH}.log" 2>&1 &
    SERVER_PID=$!
    
    log_info "Server started with PID $SERVER_PID"
    log_info "Log: $PROJECT_ROOT/logs/g1_server_epoch${EPOCH}.log"
    
    # Wait for server to be ready
    log_info "Waiting for server to be ready..."
    local max_wait=120
    for i in $(seq 1 $max_wait); do
        if curl -s "http://localhost:$SERVER_PORT/health" > /dev/null 2>&1; then
            log_info "Server is ready! (took ${i}s)"
            echo ""
            echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
            echo -e "${GREEN}║  G1 POLICY SERVER RUNNING                                      ║${NC}"
            echo -e "${GREEN}║                                                                ║${NC}"
            echo -e "${GREEN}║  WebSocket: ws://${SERVER_HOST}:${SERVER_PORT}                 ║${NC}"
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
            log_error "Server process died. Check logs: $PROJECT_ROOT/logs/g1_server_epoch${EPOCH}.log"
            tail -20 "$PROJECT_ROOT/logs/g1_server_epoch${EPOCH}.log"
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
    
    # Also kill any orphaned servers on G1 port
    pkill -f "serve_policy.py.*--port.*$SERVER_PORT" 2>/dev/null || true
    sleep 1
}

# =============================================================================
# Visualizer Management
# =============================================================================

start_visualizer() {
    log_info "Starting G1 visualizer on port $VISER_PORT..."
    
    # Find sample HDF5 data to use for visualization
    local hdf5_file=""
    
    # Try to find G1 HDF5 files
    local g1_data_dir="$PROJECT_ROOT/examples/g1_control_client/g1_data_raw"
    if [ -d "$g1_data_dir" ]; then
        hdf5_file=$(find "$g1_data_dir" -name "*.hdf5" -type f 2>/dev/null | head -1)
    fi
    
    if [ -z "$hdf5_file" ]; then
        log_warn "No HDF5 file found for visualization. Visualizer not started."
        log_warn "You can still test the policy manually by running:"
        log_warn "  uv run python examples/g1_control_client/g1_policy_viz_client.py --hdf5-path <path>"
        return 1
    fi
    
    log_info "Using HDF5 data: $hdf5_file"
    
    # Start G1 visualizer in background
    nohup uv run python "$PROJECT_ROOT/examples/g1_control_client/g1_policy_viz_client.py" \
        --hdf5-path "$hdf5_file" \
        --host "localhost" \
        --port "$SERVER_PORT" \
        --viser-port "$VISER_PORT" \
        --prompt "$TASK_DESCRIPTION, Advantage=True" \
        > "$PROJECT_ROOT/logs/g1_visualizer_epoch${EPOCH}.log" 2>&1 &
    VIZ_PID=$!
    
    # Wait for visualizer to start
    sleep 5
    
    if kill -0 "$VIZ_PID" 2>/dev/null; then
        log_info "Visualizer started with PID $VIZ_PID"
        echo ""
        echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${BLUE}║  G1 VISUALIZER RUNNING                                         ║${NC}"
        echo -e "${BLUE}║                                                                ║${NC}"
        echo -e "${BLUE}║  Open in browser: http://localhost:$VISER_PORT                 ║${NC}"
        echo -e "${BLUE}║                                                                ║${NC}"
        echo -e "${BLUE}║  - Use 'Infer' button to test policy                           ║${NC}"
        echo -e "${BLUE}║  - Step through frames with slider                             ║${NC}"
        echo -e "${BLUE}║  - View camera feeds and predicted actions                     ║${NC}"
        echo -e "${BLUE}║  - Override waist yaw with slider if enabled                   ║${NC}"
        echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        return 0
    else
        log_warn "Visualizer failed to start. Check: $PROJECT_ROOT/logs/g1_visualizer_epoch${EPOCH}.log"
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
    
    # Kill any orphaned G1 visualizers
    pkill -f "g1_policy_viz_client.py" 2>/dev/null || true
}

# =============================================================================
# Data Collection
# =============================================================================

wait_for_data() {
    local epoch_dir="$DATA_DIR/epoch_$EPOCH/raw"
    
    log_phase "G1 Data Collection Phase (Epoch $EPOCH)"
    
    mkdir -p "$epoch_dir"
    
    echo ""
    echo -e "${YELLOW}Waiting for G1 robot to collect data...${NC}"
    echo ""
    echo "Expected data location: $epoch_dir"
    echo ""
    echo "On the robot station, run:"
    echo -e "  ${CYAN}python g1_execution_client.py --config training_config_g1.yaml${NC}"
    echo ""
    echo "The robot will:"
    echo "  1. Connect to policy server at $SERVER_HOST:$SERVER_PORT"
    echo "  2. Execute policy and record episodes (policy action_dim=$ACTION_DIM from YAML)"
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
    
    log_info "Syncing G1 data from robot..."
    
    mkdir -p "$epoch_dir"
    
    # Data should be pushed by robot
    log_warn "Note: G1 data should be pushed by robot via rsync."
    log_info "Expected local path: $epoch_dir"
    
    local count=$(find "$epoch_dir" -name "*.hdf5" 2>/dev/null | wc -l)
    log_info "Current files: $count"
}

# =============================================================================
# Training Pipeline
# =============================================================================

convert_epoch_data() {
    log_phase "Converting G1 Data (Epoch $EPOCH)"
    
    local raw_dir="$DATA_DIR/epoch_$EPOCH/raw"
    local count=$(find "$raw_dir" -name "*.hdf5" 2>/dev/null | wc -l)
    
    if [ "$count" -eq 0 ]; then
        log_error "No HDF5 files found in $raw_dir"
        return 1
    fi
    
    log_info "Converting $count G1 episodes..."
    log_info "Labeling mode: $LABELING_MODE"
    log_info "Policy action_dim (YAML): $ACTION_DIM"
    
    # Determine effective labeling mode
    local effective_labeling_mode="$LABELING_MODE"
    
    # For epoch 0, use human_labeling (warmup epoch - all good episodes)
    if [ "$EPOCH" -eq 0 ]; then
        log_info "Epoch 0 (Warmup): Using human_labeling mode (all episodes are baseline data)"
        log_info "  Action chunk advantages will be applied from epoch 1 onwards"
        effective_labeling_mode="human_labeling"
    fi
    
    # Special handling for action_chunk_advantage mode (only for epoch 1+)
    # Need parquet files to compute advantages, so do two-phase conversion if needed
    if [ "$effective_labeling_mode" = "action_chunk_advantage" ]; then
        local lerobot_dir="$LEROBOT_BASE_DIR/$TASK_NAME/epoch_$EPOCH"
        local parquet_dir="$lerobot_dir/data/chunk-000"
        
        # Check if parquet files exist (from previous run or phase 1)
        if [ ! -d "$parquet_dir" ] || [ -z "$(ls -A "$parquet_dir"/*.parquet 2>/dev/null)" ]; then
            log_info "ACTION CHUNK ADVANTAGE: First-time conversion"
            log_info "  Phase 1: Creating parquet files (without advantages)..."
            
            # Do initial conversion without advantages
            local initial_convert_cmd="./scripts/convert_g1_data.sh \
                --task-name \"$TASK_NAME\" \
                --task-description \"$TASK_DESCRIPTION\" \
                --epoch \"$EPOCH\" \
                --labeling-mode \"none\" \
                --num-repeats \"$NUM_REPEATS\" \
                --config-name \"$CONFIG_NAME\" \
                --data-dir \"$raw_dir\""
            
            if [ "$EPOCH" -eq 0 ]; then
                initial_convert_cmd="$initial_convert_cmd --filter-good-only"
            fi
            
            eval "$initial_convert_cmd"
            
            log_info "  Phase 1 complete: Parquet files created"
        else
            log_info "ACTION CHUNK ADVANTAGE: Parquet files already exist"
        fi
    fi
    
    # Build convert command (use G1-specific script)
    local convert_cmd="./scripts/convert_g1_data.sh \
        --task-name \"$TASK_NAME\" \
        --task-description \"$TASK_DESCRIPTION\" \
        --epoch \"$EPOCH\" \
        --labeling-mode \"$effective_labeling_mode\" \
        --num-repeats \"$NUM_REPEATS\" \
        --config-name \"$CONFIG_NAME\" \
        --data-dir \"$raw_dir\""
    
    # For epoch 0, force filtering to only good episodes
    if [ "$EPOCH" -eq 0 ]; then
        convert_cmd="$convert_cmd --filter-good-only"
    fi
    
    # Compute action chunk advantages if needed (after parquet files exist)
    if [ "$effective_labeling_mode" = "action_chunk_advantage" ]; then
        local parquet_dir="$LEROBOT_BASE_DIR/$TASK_NAME/epoch_$EPOCH/data/chunk-000"
        
        if [ -d "$parquet_dir" ]; then
            log_info "  Phase 2: Computing action chunk advantages..."
            
            # Check if checkpoint path is set (only required for "Ours" method)
            if [ "$REWARD_METHOD" = "Ours" ]; then
                if [ -z "$REWARD_CHECKPOINT_PATH" ]; then
                    log_error "reward.checkpoint_path not set in config file!"
                    log_error "Qwen checkpoint path is required for reward.method='Ours'"
                    return 1
                fi
                
                if [ ! -d "$REWARD_CHECKPOINT_PATH" ]; then
                    log_error "Reward checkpoint path does not exist: $REWARD_CHECKPOINT_PATH"
                    return 1
                fi
            elif [ "$REWARD_METHOD" = "GVL" ]; then
                # Check for OpenAI API key
                if [ -z "$OPENAI_API_KEY" ]; then
                    log_error "OPENAI_API_KEY not set in environment!"
                    log_error "OpenAI API key is required for reward.method='GVL'"
                    return 1
                fi
            elif [ "$REWARD_METHOD" = "RoboDopamine" ]; then
                # Check for goal image path
                if [ -z "$REWARD_GOAL_IMAGE_PATH" ]; then
                    log_error "reward.goal_image_path not set in config file!"
                    log_error "Action chunk advantage mode with method='RoboDopamine' requires goal image path."
                    log_error "Add to config: reward.goal_image_path: '/path/to/goal_image.png'"
                    return 1
                fi
                
                if [ ! -f "$REWARD_GOAL_IMAGE_PATH" ]; then
                    log_error "Goal image path does not exist: $REWARD_GOAL_IMAGE_PATH"
                    return 1
                fi
            else
                log_error "Unknown reward.method: $REWARD_METHOD"
                log_error "Supported methods: 'Ours', 'GVL', 'RoboDopamine'"
                return 1
            fi
            
            # Use the miniconda base environment Python that has vision model dependencies
            # (all reward methods need specialized models: Qwen, DINOv3, RoboDopamine)
            REWARD_PYTHON="/home/yuxin/miniconda/bin/python"
            log_info "Switching to conda base environment for reward computation (method=$REWARD_METHOD)"
            log_info "  Python: $REWARD_PYTHON"
            
            # Export environment variables
            export QWEN_REWARD_CHECKPOINT_PATH="$REWARD_CHECKPOINT_PATH"
            export CUDA_VISIBLE_DEVICES=$GPU_ID
            
            # Set LD_LIBRARY_PATH to use PyTorch's CUDA 13 libraries
            # This fixes cuBLAS version mismatch issues
            NVIDIA_LIB_PATH="/home/yuxin/miniconda/lib/python3.13/site-packages/nvidia/cublas/lib:/home/yuxin/miniconda/lib/python3.13/site-packages/nvidia/cu13/lib"
            export LD_LIBRARY_PATH="$NVIDIA_LIB_PATH:${LD_LIBRARY_PATH:-}"
            
            # Run advantage computation (will use cache if already computed)
            log_info "  Running compute_action_chunk_advantages.py with conda base..."
            
            # Build command with method-specific parameters
            # Note: G1 uses the same compute_action_chunk_advantages.py as H1
            ADV_CMD="$REWARD_PYTHON examples/h1_control_client/compute_action_chunk_advantages.py \
                --data-dir \"$parquet_dir\" \
                --task-instruction \"$REWARD_TASK_INSTRUCTION\" \
                --max-frames \"$REWARD_MAX_FRAMES\" \
                --look-ahead-window \"$REWARD_LOOK_AHEAD_WINDOW\" \
                --advantage-threshold \"$REWARD_ADVANTAGE_THRESHOLD\" \
                --distance-threshold \"$REWARD_DISTANCE_THRESHOLD\" \
                --reward-method \"$REWARD_METHOD\""
            
            # Add method-specific parameters
            if [ "$REWARD_METHOD" = "Ours" ]; then
                ADV_CMD="$ADV_CMD --checkpoint-path \"$REWARD_CHECKPOINT_PATH\""
            elif [ "$REWARD_METHOD" = "RoboDopamine" ]; then
                ADV_CMD="$ADV_CMD --goal-image-path \"$REWARD_GOAL_IMAGE_PATH\""
            fi
            
            eval "$ADV_CMD"
            
            log_info "  Switching back to project .venv for training"
            
            log_info "  Phase 2 complete: Advantages computed"
            
            # Copy advantages to raw/ directory with HDF5-compatible names
            # Parquet uses episode_000000, HDF5 uses episode_0
            log_info "  Copying advantages to raw/ directory..."
            for adv_file in "$parquet_dir"/episode_*_action_chunk_advantages.pkl; do
                if [ -f "$adv_file" ]; then
                    # Convert episode_000000_action_chunk_advantages.pkl -> episode_0_action_chunk_advantages.pkl
                    basename_file=$(basename "$adv_file")
                    # Extract the number (e.g., 000000 from episode_000000_action_chunk_advantages.pkl)
                    if [[ $basename_file =~ episode_([0-9]+)_action_chunk_advantages\.pkl ]]; then
                        episode_num="${BASH_REMATCH[1]}"
                        # Remove leading zeros
                        episode_num_stripped=$((10#$episode_num))
                        new_name="episode_${episode_num_stripped}_action_chunk_advantages.pkl"
                        cp "$adv_file" "$raw_dir/$new_name"
                        log_info "    Copied $basename_file -> $new_name"
                    fi
                fi
            done
            
            # Save norm_stats.json to temp before deleting directory
            local norm_stats_file="$lerobot_dir/norm_stats.json"
            local temp_norm_stats="/tmp/openpi_g1_norm_stats_epoch${EPOCH}.json"
            if [ -f "$norm_stats_file" ]; then
                log_info "  Saving norm_stats.json to temp location..."
                cp "$norm_stats_file" "$temp_norm_stats"
            fi
            
            # Clean up old parquet dataset to re-convert with advantages
            log_info "  Phase 3: Re-converting with advantages..."
            rm -rf "$LEROBOT_BASE_DIR/$TASK_NAME/epoch_$EPOCH"
        else
            log_error "Parquet directory not found: $parquet_dir"
            log_error "Phase 1 conversion may have failed"
            return 1
        fi
    fi
    
    # Add reward labeling parameters if in reward_labeling or action_chunk_advantage mode
    if [ "$effective_labeling_mode" = "reward_labeling" ] || [ "$effective_labeling_mode" = "action_chunk_advantage" ]; then
            # Check if checkpoint path is set (only required for "Ours" method)
            if [ "$REWARD_METHOD" = "Ours" ]; then
                if [ -z "$REWARD_CHECKPOINT_PATH" ]; then
                    log_error "reward.checkpoint_path not set in config file!"
                    log_error "Reward labeling with method='Ours' requires Qwen checkpoint path."
                    log_error "Add to config: reward.checkpoint_path: '/path/to/checkpoint'"
                    return 1
                fi
                
                if [ ! -d "$REWARD_CHECKPOINT_PATH" ]; then
                    log_error "Reward checkpoint path does not exist: $REWARD_CHECKPOINT_PATH"
                    return 1
                fi
            elif [ "$REWARD_METHOD" = "GVL" ]; then
                # Check for OpenAI API key
                if [ -z "$OPENAI_API_KEY" ]; then
                    log_error "OPENAI_API_KEY not set in environment!"
                    log_error "Reward labeling with method='GVL' requires OpenAI API key."
                    return 1
                fi
            elif [ "$REWARD_METHOD" = "RoboDopamine" ]; then
                # Check for goal image path
                if [ -z "$REWARD_GOAL_IMAGE_PATH" ]; then
                    log_error "reward.goal_image_path not set in config file!"
                    log_error "Reward labeling with method='RoboDopamine' requires goal image path."
                    log_error "Add to config: reward.goal_image_path: '/path/to/goal_image.png'"
                    return 1
                fi
                
                if [ ! -f "$REWARD_GOAL_IMAGE_PATH" ]; then
                    log_error "Goal image path does not exist: $REWARD_GOAL_IMAGE_PATH"
                    return 1
                fi
            else
                log_error "Unknown reward.method: $REWARD_METHOD"
                log_error "Supported methods: 'Ours', 'GVL', 'RoboDopamine'"
                return 1
            fi
        
        # Note: We use miniconda Python for Qwen operations
        # No need to install Qwen dependencies to .venv
        
        # Export checkpoint path for the convert script
        export QWEN_REWARD_CHECKPOINT_PATH="$REWARD_CHECKPOINT_PATH"
        export REWARD_LOOK_AHEAD_WINDOW="$REWARD_LOOK_AHEAD_WINDOW"
        
        # Set CUDA_VISIBLE_DEVICES for reward labeling (use same GPU as training)
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        
        if [ "$effective_labeling_mode" = "action_chunk_advantage" ]; then
            log_info "Using action chunk advantage labeling with:"
            log_info "  Reward Method: $REWARD_METHOD"
            log_info "  Mode: Fine-grained per-frame advantages"
            if [ "$REWARD_METHOD" = "Ours" ]; then
                log_info "  Checkpoint: $REWARD_CHECKPOINT_PATH"
            elif [ "$REWARD_METHOD" = "RoboDopamine" ]; then
                log_info "  Goal Image: $REWARD_GOAL_IMAGE_PATH"
            fi
            log_info "  Max frames: $REWARD_MAX_FRAMES"
            log_info "  Look-ahead window: $REWARD_LOOK_AHEAD_WINDOW frames"
            log_info "  Advantage threshold: ${REWARD_ADVANTAGE_THRESHOLD} (top ${REWARD_ADVANTAGE_THRESHOLD} percentile)"
            log_info "  Random drop rate: ${REWARD_RANDOM_DROP_RATE} (keep original prompt without advantage)"
            log_info "  GPU: $GPU_ID"
        else
            log_info "Using reward labeling with:"
            log_info "  Reward Method: $REWARD_METHOD"
            log_info "  Mode: Episode-level advantages"
            if [ "$REWARD_METHOD" = "Ours" ]; then
                log_info "  Checkpoint: $REWARD_CHECKPOINT_PATH"
            elif [ "$REWARD_METHOD" = "RoboDopamine" ]; then
                log_info "  Goal Image: $REWARD_GOAL_IMAGE_PATH"
            fi
            log_info "  Max frames: $REWARD_MAX_FRAMES"
            log_info "  Image rotation: $REWARD_IMAGE_ROTATION"
            log_info "  Advantage threshold: ${REWARD_ADVANTAGE_THRESHOLD} (percentile)"
            log_info "  GPU: $GPU_ID"
        fi
        
        convert_cmd="$convert_cmd \
            --reward-method \"$REWARD_METHOD\" \
            --reward-task-instruction \"$REWARD_TASK_INSTRUCTION\" \
            --reward-max-frames \"$REWARD_MAX_FRAMES\" \
            --reward-image-rotation \"$REWARD_IMAGE_ROTATION\" \
            --reward-advantage-threshold \"$REWARD_ADVANTAGE_THRESHOLD\" \
            --reward-random-drop-rate \"$REWARD_RANDOM_DROP_RATE\" \
            --reward-reject-rate \"$REWARD_REJECT_RATE\""
        
        # Add goal image path for RoboDopamine
        if [ "$effective_labeling_mode" = "reward_labeling" ] || [ "$effective_labeling_mode" = "action_chunk_advantage" ]; then
            if [ "$REWARD_METHOD" = "RoboDopamine" ] && [ -n "$REWARD_GOAL_IMAGE_PATH" ]; then
                convert_cmd="$convert_cmd --reward-goal-image-path \"$REWARD_GOAL_IMAGE_PATH\""
            fi
        fi
    fi
    
    # Set LD_LIBRARY_PATH to use PyTorch's CUDA 13 libraries
    # This fixes cuBLAS version mismatch issues for reward-based labeling
    NVIDIA_LIB_PATH="/home/yuxin/miniconda/lib/python3.13/site-packages/nvidia/cublas/lib:/home/yuxin/miniconda/lib/python3.13/site-packages/nvidia/cu13/lib"
    export LD_LIBRARY_PATH="$NVIDIA_LIB_PATH:${LD_LIBRARY_PATH:-}"
    
    # Execute conversion
    eval "$convert_cmd"
    
    # Restore norm_stats.json from temp if it was saved (for action_chunk_advantage mode)
    if [ "$effective_labeling_mode" = "action_chunk_advantage" ]; then
        local lerobot_data_dir="$LEROBOT_BASE_DIR/$TASK_NAME/epoch_$EPOCH"
        local temp_norm_stats="/tmp/openpi_g1_norm_stats_epoch${EPOCH}.json"
        local norm_stats_file="$lerobot_data_dir/norm_stats.json"
        
        if [ -f "$temp_norm_stats" ]; then
            log_info "  Restoring norm_stats.json from temp..."
            cp "$temp_norm_stats" "$norm_stats_file"
            rm -f "$temp_norm_stats"
            log_info "    Restored to: $norm_stats_file"
        fi
    fi
    
    # Final message for action_chunk_advantage mode
    if [ "$effective_labeling_mode" = "action_chunk_advantage" ]; then
        # Clean up temporary advantage files from raw/ directory
        rm -f "$raw_dir"/*_action_chunk_advantages.pkl 2>/dev/null || true
        
        log_info "  Phase 3 complete: Dataset created with action chunk advantages"
        log_info ""
        log_info "  ✓ Three-phase conversion complete!"
        log_info "    1. Parquet files created from HDF5"
        log_info "    2. Action chunk advantages computed"
        log_info "    3. Dataset re-created with advantages"
    fi
}

train_epoch() {
    local base_checkpoint="${1:-}"
    
    log_phase "Training G1 Policy (Epoch $EPOCH)"
    
    local train_args="--task-name $TASK_NAME --epoch $EPOCH --config-name $CONFIG_NAME --gpu $GPU_ID"
    train_args="$train_args --max-epochs $MAX_EPOCHS --save-interval $SAVE_INTERVAL --keep-period $KEEP_PERIOD"
    train_args="$train_args --action-dim $ACTION_DIM"
    
    if [ -n "$base_checkpoint" ]; then
        train_args="$train_args --base-checkpoint $base_checkpoint"
    fi
    
    # Use G1-specific training script
    ./scripts/train_g1_local.sh $train_args
    
    # Find the latest checkpoint
    local checkpoint_epoch_dir="$CHECKPOINT_BASE_DIR/epoch_$EPOCH"
    log_info "Looking for checkpoints in: $checkpoint_epoch_dir"
    
    if [ -d "$checkpoint_epoch_dir" ]; then
        log_info "Directory contents:"
        ls -la "$checkpoint_epoch_dir" 2>/dev/null || true
        
        local found_checkpoint=$(ls -d "$checkpoint_epoch_dir"/[0-9]*/ 2>/dev/null | sort -V | tail -1)
        
        if [ -n "$found_checkpoint" ] && [ -d "$found_checkpoint" ]; then
            LAST_CHECKPOINT="${found_checkpoint%/}"
            log_info "Found checkpoint: $LAST_CHECKPOINT"
        else
            log_error "No numeric checkpoint directories found!"
            return 1
        fi
    else
        log_error "Checkpoint directory does not exist: $checkpoint_epoch_dir"
        return 1
    fi
    
    if [ ! -d "$LAST_CHECKPOINT/params" ]; then
        log_error "Checkpoint missing params directory: $LAST_CHECKPOINT/params"
        return 1
    fi
    
    log_info "Training complete. Valid checkpoint: $LAST_CHECKPOINT"
}

# =============================================================================
# Main Functions
# =============================================================================

show_config() {
    echo ""
    echo -e "${CYAN}========================================================${NC}"
    echo -e "${CYAN}  G1 Configuration (from $CONFIG_FILE)${NC}"
    echo -e "${CYAN}========================================================${NC}"
    echo ""
    echo -e "  Robot:             ${GREEN}G1${NC}"
    echo -e "  Task Name:         ${GREEN}$TASK_NAME${NC}"
    echo -e "  Task Description:  ${GREEN}$TASK_DESCRIPTION${NC}"
    echo -e "  Policy Config:     $CONFIG_NAME"
    echo -e "  Warmup Checkpoint: ${WARMUP_CHECKPOINT:-none}"
    echo -e "  Policy action_dim:  $ACTION_DIM  (YAML policy_server.action_dim: 29 full, 28 no waist, 16 binary grippers)"
    echo -e "  Max Epochs:        $MAX_EPOCHS"
    echo -e "  Save Interval:     $SAVE_INTERVAL"
    echo -e "  Keep Period:       $KEEP_PERIOD"
    echo -e "  Labeling Mode:     $LABELING_MODE"
    echo -e "  GPU:               $GPU_ID"
    echo -e "  Policy Server:     $SERVER_HOST:$SERVER_PORT"
    echo -e "  Viser UI:          http://localhost:$VISER_PORT"
    echo ""
}

show_menu() {
    echo "Select starting phase:"
    echo ""
    echo -e "  ${GREEN}1${NC}) Data Collection - Start server, G1 robot collects data"
    echo -e "  ${GREEN}2${NC}) Convert Data - Convert existing HDF5 data to LeRobot format"
    echo -e "  ${GREEN}3${NC}) Training - Train policy on converted data"
    echo -e "  ${GREEN}4${NC}) Resume from state file"
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
            local found=$(ls -d "$prev_checkpoint_dir"/[0-9]*/ 2>/dev/null | sort -V | tail -1)
            if [ -n "$found" ]; then
                checkpoint="${found%/}"
            fi
        fi
    fi
    
    # Fall back to LAST_CHECKPOINT from state
    if [ -z "$checkpoint" ] && [ -n "$LAST_CHECKPOINT" ]; then
        local cleaned="${LAST_CHECKPOINT%/}"
        if [ -d "$cleaned" ]; then
            checkpoint="$cleaned"
        fi
    fi
    
    # Fall back to WARMUP_CHECKPOINT
    if [ -z "$checkpoint" ] && [ -n "$WARMUP_CHECKPOINT" ] && [ "$WARMUP_CHECKPOINT" != "null" ]; then
        local cleaned="${WARMUP_CHECKPOINT%/}"
        if [ -d "$cleaned" ]; then
            checkpoint="$cleaned"
        fi
    fi
    
    # Final validation
    if [ -n "$checkpoint" ] && [ ! -d "$checkpoint/params" ]; then
        log_warn "Checkpoint $checkpoint missing params directory, may be invalid"
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
    
    echo "$data_dir"
}

run_data_collection_phase() {
    local checkpoint=$(determine_checkpoint)
    local data_dir=$(determine_data_dir)
    
    if [ -z "$checkpoint" ]; then
        log_error "No checkpoint available for serving!"
        if [ "$EPOCH" -eq 0 ]; then
            log_error "Set policy.warmup_checkpoint in $CONFIG_FILE for epoch 0 data collection"
        else
            log_error "No checkpoint found from previous epoch"
        fi
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

run_convert_phase() {
    save_state "converting"
    convert_epoch_data
    save_state "convert_complete"
    log_info "Data conversion complete for epoch $EPOCH"
}

run_training_phase() {
    local checkpoint=""
    
    # For epoch 0, train from scratch using base weights
    # For epoch 1+, fine-tune from previous epoch's checkpoint
    if [ "$EPOCH" -eq 0 ]; then
        log_info "Epoch 0: Training from scratch using base weights in config ($CONFIG_NAME)"
        checkpoint=""
    else
        checkpoint=$(determine_checkpoint)
        if [ -z "$checkpoint" ]; then
            log_error "No checkpoint available for epoch $EPOCH training!"
            return 1
        fi
        log_info "Epoch $EPOCH: Fine-tuning from checkpoint: $checkpoint"
    fi
    
    save_state "training"
    train_epoch "$checkpoint"
    
    log_info "After training, LAST_CHECKPOINT = $LAST_CHECKPOINT"
    
    if [ -z "$LAST_CHECKPOINT" ] || [ ! -d "$LAST_CHECKPOINT" ]; then
        log_error "LAST_CHECKPOINT not set correctly after training!"
        return 1
    fi
    
    save_state "epoch_complete"
    log_info "Saved state with checkpoint: $LAST_CHECKPOINT"
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
        log_info "Found existing G1 state file."
        echo ""
        show_menu
        read -p "Enter choice (or press Enter to resume): " choice
        
        case "$choice" in
            1)
                phase="data_collection"
                ;;
            2)
                phase="convert"
                ;;
            3)
                phase="training"
                ;;
            4|"")
                # Resume from state
                case "$STATUS" in
                    collecting_data|idle)
                        phase="data_collection"
                        ;;
                    converting|convert_complete)
                        phase="convert"
                        ;;
                    training|epoch_complete)
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
                phase="convert"
                ;;
            3)
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
        log_phase "G1 EPOCH $EPOCH"
        
        if [ "$phase" = "data_collection" ]; then
            run_data_collection_phase
            phase="convert"
        fi
        
        if [ "$phase" = "convert" ]; then
            run_convert_phase
            phase="training"
        fi
        
        if [ "$phase" = "training" ]; then
            run_training_phase
            EPOCH=$((EPOCH + 1))
            phase="data_collection"
        fi
    done
    
    log_phase "G1 Training Complete!"
    log_info "Completed $EPOCH epochs"
    log_info "Final checkpoint: $LAST_CHECKPOINT"
    save_state "complete"
}

main "$@"
