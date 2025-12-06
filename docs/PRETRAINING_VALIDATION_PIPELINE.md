# Pre-Training & Validation Pipeline

This document covers:
1. **Pre-training** - Training policies on existing datasets (e.g., humanoid_everyday)
2. **Validation** - Testing policies before deployment using the visualizer

---

## Part 1: Pre-Training

### Data Sources

#### Option A: Humanoid Everyday Dataset

Download from the LeRobot Hub:
```bash
# Using huggingface-cli
huggingface-cli download lerobot/humanoid_everyday --local-dir ./humanoid_everyday
```

#### Option B: Convert Custom HDF5 Data

If you have HDF5 files from teleoperation:

```bash
cd ~/openpi

# Convert HDF5 to LeRobot format
python examples/h1_control_client/convert_h1_data_to_lerobot.py \
    --input-dir ./path/to/hdf5/files \
    --output-dir ./h1_data_lerobot/my_task \
    --task-description "your task description" \
    --labeling-mode human_labeling
```

#### Option C: Convert Humanoid Everyday to Local Format

```bash
python examples/h1_control_client/convert_humanoid_everyday_to_hdf5.py \
    --input-dir ./humanoid_everyday \
    --output-dir ./training_data/h1/converted \
    --subset move_lid
```

### Training

#### Basic Training

```bash
cd ~/openpi

# Train on local data
./scripts/train_h1_local.sh \
    --task-name my_task \
    --config-name pi0_fast_h1 \
    --gpu 0
```

#### Training with Pre-trained Checkpoint

```bash
./scripts/train_h1_local.sh \
    --task-name my_task \
    --config-name pi0_fast_h1 \
    --base-checkpoint checkpoints/pi0_fast_h1/pretrained/999 \
    --gpu 0
```

#### Full Training Options

```bash
./scripts/train_h1_local.sh \
    --task-name lift_lid \
    --epoch 0 \
    --config-name pi0_fast_h1 \
    --gpu 0 \
    --max-epochs 400 \
    --save-interval 200 \
    --keep-period 100 \
    --base-checkpoint /path/to/checkpoint
```

### Training Configuration

Key parameters in training configs (`configs/pi0_fast_h1.py`):

```python
@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 2.5e-5
    num_train_steps: int = 30_000
    save_interval: int = 1000
    
    # Data
    dataset_name: str = "h1_data_lerobot/my_task"
    
    # Model
    action_horizon: int = 50
    action_dim: int = 14  # H1-2 arm joints
```

### Monitoring Training

Training logs are saved to:
```
logs/train_{config_name}_{task_name}.log
```

Use TensorBoard:
```bash
tensorboard --logdir=checkpoints/pi0_fast_h1/my_task/
```

---

## Part 2: Validation

### Policy Visualizer

The visualizer (`h1_policy_viz_client.py`) provides a web interface to:
- View camera feeds from recorded data
- Step through trajectory frames
- Run policy inference on frames
- Compare predicted vs recorded actions
- Execute predictions on real robot (optional)

### Basic Validation (No Robot)

```bash
cd ~/openpi

# Start policy server
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_h1 \
    --policy.dir=checkpoints/pi0_fast_h1/my_task/999

# In another terminal, start visualizer
uv run python examples/h1_control_client/h1_policy_viz_client.py \
    --hdf5-path training_data/h1/test_episode.hdf5 \
    --host localhost \
    --port 8000 \
    --prompt "your task description, Advantage=True"
```

Open `http://localhost:8080` in your browser.

### Visualizer Features

#### Frame Navigation
- Use slider to scrub through recorded trajectory
- View joint positions at each frame
- See camera images (head, left wrist, right wrist)

#### Policy Inference
- Click "Infer" to query policy at current frame
- View predicted action trajectory
- Compare to recorded ground truth

#### 3D Visualization
- URDF model shows robot pose
- Predicted trajectory shown as ghost poses
- Color coding: blue = recorded, green = predicted

### Validation with Real Robot

For testing on the actual robot:

1. **On station computer** - Start robot client in listen mode:
   ```bash
   cd ~/openpi/examples/h1_control_client
   
   python h1_remote_client.py \
       --listen-mode \
       --listen-port 5007
   ```

2. **SSH tunnel** (if visualizer on remote server):
   ```bash
   ssh -R 5007:localhost:5007 -L 8080:localhost:8080 P6000
   ```

3. **On GPU server** - Start visualizer with robot execution:
   ```bash
   uv run python examples/h1_control_client/h1_policy_viz_client.py \
       --hdf5-path training_data/h1/test_episode.hdf5 \
       --host localhost \
       --port 8000 \
       --prompt "your task description" \
       --robot-execution \
       --robot-host localhost \
       --robot-port 5007
   ```

#### Robot Execution Controls

| Button | Action |
|--------|--------|
| Reset to Frame | Move robot to joint positions at current frame |
| Reset Zombie | Move robot to arms-forward pose |
| Execute Chunk | Run predicted actions on robot |
| Start Recording | Begin recording episode |
| Stop Recording | Save recorded episode |

---

## Part 3: Validation Checklist

Before deploying a policy, verify:

### 1. Visual Inspection
- [ ] Policy produces smooth, continuous actions
- [ ] Predicted trajectory is similar to demonstrations
- [ ] No erratic jumps or oscillations

### 2. Action Statistics
- [ ] Action values within expected range (typically -3 to 3 rad)
- [ ] Gradual transitions between timesteps
- [ ] Consistent behavior across different frames

### 3. Camera Feed Check
- [ ] All cameras producing valid images
- [ ] No dummy gray images
- [ ] Correct camera assignment (head, left wrist, right wrist)

### 4. Prompt Verification
- [ ] Task description matches training data
- [ ] Advantage=True appended for inference
- [ ] Prompt format: `"task description, Advantage=True"`

### 5. Robot Execution Test
- [ ] Reset to frame works correctly
- [ ] Single action chunk executes smoothly
- [ ] Robot returns to safe pose after test

---

## Directory Structure

```
openpi/
├── training_data/
│   └── h1/
│       ├── circular.hdf5          # Sample data
│       └── converted/             # Converted datasets
│
├── h1_data_lerobot/               # LeRobot format data
│   └── my_task/
│       ├── data/
│       │   └── train-*.parquet
│       ├── meta/
│       │   └── info.json
│       └── norm_stats.json
│
├── checkpoints/
│   └── pi0_fast_h1/
│       └── my_task/
│           └── 999/
│               ├── params/        # Model weights
│               └── config.json
│
├── examples/h1_control_client/
│   ├── h1_policy_viz_client.py    # Visualizer
│   ├── convert_h1_data_to_lerobot.py
│   └── convert_humanoid_everyday_to_hdf5.py
│
└── scripts/
    ├── serve_policy.py
    └── train_h1_local.sh
```

---

## Common Issues

### "No norm_stats.json found"
```bash
# Regenerate normalization statistics
python examples/h1_control_client/compute_norm_stats.py \
    --data-dir h1_data_lerobot/my_task
```

### Policy produces constant output
- Check that input images are not dummy (gray) images
- Verify state vector has correct dimension (14 for H1-2)
- Ensure checkpoint matches config name

### Visualizer won't connect to server
```bash
# Check server is running
curl http://localhost:8000/health

# Check for port conflicts
lsof -i :8000
lsof -i :8080
```

### Robot execution fails
- Verify SSH tunnel is active
- Check robot is in ready mode (not damping)
- Ensure h1_remote_client.py is in listen mode

---

## Quick Reference

### Start Policy Server
```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_h1 \
    --policy.dir=checkpoints/pi0_fast_h1/my_task/999
```

### Start Visualizer (Local)
```bash
uv run python examples/h1_control_client/h1_policy_viz_client.py \
    --hdf5-path training_data/h1/test.hdf5 \
    --host localhost \
    --port 8000 \
    --prompt "task description, Advantage=True"
```

### Convert Data
```bash
python examples/h1_control_client/convert_h1_data_to_lerobot.py \
    --input-dir ./raw_data \
    --output-dir ./h1_data_lerobot/my_task \
    --task-description "my task"
```

### Train Model
```bash
./scripts/train_h1_local.sh --task-name my_task --config-name pi0_fast_h1
```

---

## References

- [OpenPi Training Documentation](../README.md)
- [LeRobot Data Format](https://huggingface.co/docs/lerobot)
- [H1-2 Robot Documentation](https://support.unitree.com/home/en/H1_developer)

