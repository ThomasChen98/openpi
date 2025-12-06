# H1-2 Auto-Training Pipeline

This document describes the automated human-in-the-loop training pipeline for the H1-2 robot. The pipeline iteratively:
1. Serves a policy checkpoint
2. Collects robot execution data with human advantage labels
3. Trains on the labeled data
4. Repeats with the improved policy

---

## Overview

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GPU Server (P6000)                          │
│                                                                     │
│  ┌─────────────────────┐    ┌─────────────────────┐                │
│  │integrated_training.sh│───►│   serve_policy.py   │                │
│  │                     │    │   (port 8000)       │                │
│  │  - State management │    └─────────────────────┘                │
│  │  - Data conversion  │              ▲                            │
│  │  - Training         │              │ WebSocket                  │
│  │  - Epoch tracking   │              │                            │
│  └─────────────────────┘              │                            │
└───────────────────────────────────────┼────────────────────────────┘
                                        │
                              SSH Tunnel (port 8000)
                                        │
┌───────────────────────────────────────┼────────────────────────────┐
│                    Station Computer (Laptop)                        │
│                                        │                            │
│  ┌─────────────────────┐              ▼                            │
│  │h1_execution_client.py│◄────────────────────────►  H1-2 Robot    │
│  │                     │                                           │
│  │  State Machine:     │                                           │
│  │  WAITING → READY → EXECUTING → LABELING → SAVING → DECIDING    │
│  │                                                                 │
│  └─────────────────────┘                                           │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Epoch N:
  1. Server loads checkpoint (epoch N-1 or base)
  2. Robot executes policy, records episodes
  3. Human labels each episode: GOOD (Advantage=True) or BAD (Advantage=False)
  4. Data synced to server
  5. Convert HDF5 → LeRobot format with advantage labels
  6. Train new checkpoint
  7. Increment epoch → repeat
```

---

## Setup

### 1. Configure the Pipeline

Edit `examples/h1_control_client/training_config.yaml`:

```yaml
# Task definition
task:
  name: "lift_lid"
  description: "Lift the lid off the bowl"

# Policy configuration
policy:
  config_name: "pi0_fast_h1"
  base_checkpoint: "checkpoints/pi0_fast_h1/pretrained/999"

# Training parameters
training:
  max_epochs: 400
  save_interval: 200
  keep_period: 100
  labeling_mode: "human_labeling"
  gpu_id: 0

# Server settings
policy_server:
  host: "localhost"
  port: 8000

# Data paths
data:
  save_dir: "./h1_data_auto"
  rsync:
    enabled: true
    target: "user@P6000:~/openpi/data/training_epochs"
```

### 2. Set Up SSH Tunnel

On the station computer, create a tunnel to the GPU server:

```bash
ssh -L 8000:localhost:8000 -L 8080:localhost:8080 P6000
```

This forwards:
- Port 8000: Policy server WebSocket
- Port 8080: Visualizer web interface

### 3. Start Image Server on Robot

```bash
ssh unitree@192.168.123.163
python3 image_server/image_server.py
```

---

## Running the Pipeline

### On GPU Server (P6000)

```bash
cd ~/openpi

# Start the integrated training pipeline
./scripts/integrated_training.sh
```

The script will:
1. Show current configuration
2. Prompt you to select a phase:
   - `1` - Data Collection (start server, wait for robot)
   - `2` - Training (convert data, train model)
   - `3` - Resume from saved state

### On Station Computer (Robot Side)

```bash
cd ~/openpi/examples/h1_control_client

python h1_execution_client.py --config training_config.yaml
```

Or to skip waiting for server and start immediately:
```bash
python h1_execution_client.py --config training_config.yaml --start-immediately
```

---

## Client State Machine

### States

| State | Description |
|-------|-------------|
| **WAITING** | Polling server for new policy weights |
| **READY** | Policy loaded, waiting to start episode |
| **EXECUTING** | Running policy, recording data |
| **LABELING** | Episode complete, awaiting advantage label |
| **DAMPING** | (Optional) Manual robot adjustment |
| **SAVING** | Saving episode with label |
| **DECIDING** | Continue or finish epoch |
| **SYNCING** | Upload data to server |
| **FINISHED** | Session complete |

### Controls

| Key | State | Action |
|-----|-------|--------|
| `y` | READY | Start new episode |
| `n` | READY | Finish epoch, sync data |
| `s` | EXECUTING | Stop execution, enter labeling |
| `g` | LABELING | Label as GOOD (Advantage=True) |
| `b` | LABELING | Label as BAD (Advantage=False) |
| `r` | LABELING | Reset robot to starting pose |
| `d` | LABELING | Enter damping mode |
| `e` | DAMPING | Exit damping, save episode |
| `Ctrl+C` | Any | Emergency exit |

---

## Workflow Example

### Epoch 0 (First Training Cycle)

1. **Server starts** with base checkpoint
2. **Robot connects**, enters READY state
3. **Press `y`** to start episode 1
4. **Robot executes policy**, press `s` to stop
5. **Label episode**: `g` (good) or `b` (bad)
6. **Choose**: `r` (reset) or `d` (damping)
7. **Press `y`** for more episodes, or `n` to finish epoch
8. **Data syncs** to server
9. **Server converts** data and trains
10. **Epoch 1 begins** with new checkpoint

### Console Output (Robot Side)

```
============================================================
[READY] Policy ready (Epoch 0)
  Episodes collected this epoch: 0
  Press 'y' to start episode 1
  Press 'n' to finish this epoch and wait for new training
============================================================
Your choice: y

============================================================
[EXECUTING] Running policy (epoch 0, episode 1)
  Press 's' to stop execution and enter labeling mode
============================================================
  Resetting robot to starting pose...
  Querying policy for chunk 1...
  Received action chunk: shape=(50, 14)
  Executing 50 actions at 50Hz...
  Step 0/50: joints = [-1.38, 0.56, ...]
  ...
  Stop signal received at action 127/50

============================================================
[LABELING] Episode 1 execution complete
  Recorded 127 frames (2.5 seconds)
  Was this execution successful?
    'g' - GOOD (Advantage=True)
    'b' - BAD (Advantage=False)
============================================================
Label this episode (g/b): g
Episode labeled as GOOD (Advantage=True)

------------------------------------------------------------
  Robot is holding position. Choose next action:
    'r' - Reset to starting pose
    'd' - Enter damping mode (manual adjustment)
------------------------------------------------------------
Your choice (r/d): r
  Resetting robot to starting pose...
```

---

## Directory Structure

```
openpi/
├── examples/h1_control_client/
│   ├── h1_execution_client.py      # Robot-side client
│   ├── training_config.yaml        # Configuration file
│   └── h1_data_auto/               # Recorded data
│       └── lift_lid/
│           └── epoch_0/
│               └── raw/
│                   ├── episode_0.hdf5
│                   └── episode_1.hdf5
│
├── scripts/
│   ├── integrated_training.sh      # Main orchestrator
│   ├── convert_data.sh             # HDF5 → LeRobot
│   └── train_h1_local.sh           # Training script
│
├── checkpoints/
│   └── pi0_fast_h1/
│       └── lift_lid/
│           ├── epoch_0/
│           │   └── 399/            # Trained checkpoint
│           └── epoch_1/
│               └── 399/
│
└── training_state_lift_lid.json    # Pipeline state
```

---

## State File

The pipeline maintains state in `training_state_{task_name}.json`:

```json
{
  "task_name": "lift_lid",
  "task_description": "Lift the lid off the bowl",
  "config_name": "pi0_fast_h1",
  "epoch": 1,
  "last_checkpoint": "/path/to/checkpoints/epoch_0/399",
  "base_checkpoint": "/path/to/pretrained/checkpoint",
  "status": "collecting_data",
  "updated_at": "2025-12-06T10:30:00-08:00"
}
```

This allows the pipeline to resume after interruption.

---

## Advantage Labeling

The key innovation is **advantage function labeling**:

- **GOOD (Advantage=True)**: Task completed successfully
- **BAD (Advantage=False)**: Task failed or needs improvement

During training, prompts are formatted as:
```
"Lift the lid off the bowl, Advantage=True"
"Lift the lid off the bowl, Advantage=False"
```

At inference time, we always use `Advantage=True` to sample from the "good" action distribution.

---

## Troubleshooting

### Robot can't connect to server
```bash
# Check SSH tunnel is active
lsof -i :8000

# Verify server is running
curl http://localhost:8000/health
```

### Low frame count
- Ensure you're executing full action chunks (50 actions each)
- Check that recording is active during execution
- The client should log "Recorded X frames" after each episode

### Policy produces erratic movements
- Verify the checkpoint is valid
- Check that norm_stats.json exists in data directory
- Ensure camera feeds are working (not dummy images)

### Data not syncing
- Check rsync configuration in `training_config.yaml`
- Verify SSH keys are set up correctly
- Manual sync: `rsync -avz ./h1_data_auto/ user@P6000:~/openpi/data/`

---

## References

- [Training Configuration](../examples/h1_control_client/training_config.yaml)
- [Execution Client](../examples/h1_control_client/h1_execution_client.py)
- [Integrated Training Script](../scripts/integrated_training.sh)

