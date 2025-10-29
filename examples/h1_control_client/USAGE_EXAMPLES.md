# H1 Control Client - Usage Examples

Quick reference for common workflows with the H1 control client.

## 1. Replay Recorded Demonstrations

View your collected data without any policy inference:

```bash
cd examples/h1_control_client/utils
python data_replay.py --hdf5-path ../processed_data/circular.hdf5
```

**Use Cases:**
- Verify data collection worked correctly
- Inspect robot motions frame-by-frame
- Check camera angles and image quality
- Review joint trajectories

## 2. Test Policy with Mock Server

Quick testing with random actions:

```bash
# Terminal 1: Start mock policy server
cd /path/to/openpi
uv run scripts/mock_policy_server.py --port 8000

# Terminal 2: Run client (robot or simulation)
cd examples/h1_control_client
python h1_remote_client.py --host 0.0.0.0 --port 8000
```

## 3. Validate Trained Policy (Visualization Only)

Visualize what your trained policy predicts without deploying to robot:

```bash
# Terminal 1: Start trained policy server
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_h1_finetune \
  --policy.dir=checkpoints/pi05_h1_finetune/pi05_h1_H50/1999

# Terminal 2: Run visualization client
cd examples/h1_control_client
python h1_policy_viz_client.py \
  --hdf5-path processed_data/circular.hdf5 \
  --prompt "move arm circularly"
```

**Interactive Workflow:**
1. Select a frame from your dataset
2. Click "🤖 Infer Action Chunk" to get policy predictions  
3. Click "▶️ Play Action Chunk" to visualize the motion
4. Toggle between ground truth and predictions to compare
5. Iterate on different frames to test robustness

**Use Cases:**
- Verify policy learned the task correctly
- Debug prediction failures before robot deployment
- Compare predictions with ground truth motions
- Test generalization across dataset

## 4. Deploy Trained Policy to Robot

Run your trained policy on the actual H1-2 robot:

```bash
# Terminal 1: Stream camera from robot
# (on robot)
cd ~/xr_teleoperate
python3 image_server.py

# Terminal 2: Start trained policy server
# (on GPU server)
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_h1_finetune \
  --policy.dir=checkpoints/pi05_h1_finetune/pi05_h1_H50/1999

# Terminal 3: Run robot client
# (on control laptop)
cd examples/h1_control_client
python h1_remote_client.py \
  --host <gpu_server_ip> \
  --port 8000 \
  --camera-source remote \
  --camera-host <robot_ip>
```

## 5. Development Workflow

Typical iteration cycle when developing a new policy:

### Step 1: Collect Data
```bash
# Use teleoperation or other method to collect demonstrations
# Save to: processed_data/my_task.hdf5
```

### Step 2: Convert to LeRobot Format
```bash
uv run training_data/h1/convert_h1_data_to_lerobot.py \
  --data_dir examples/h1_control_client/processed_data/my_task.hdf5 \
  --repo_id username/h1_my_task \
  --num_repeats 256
```

### Step 3: Train Policy
```bash
uv run scripts/train.py \
  --config pi05_h1_finetune \
  --dataset username/h1_my_task
```

### Step 4: Validate with Visualization
```bash
# Terminal 1: Policy server
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_h1_finetune \
  --policy.dir=checkpoints/pi05_h1_finetune/latest

# Terminal 2: Visualization
cd examples/h1_control_client
python h1_policy_viz_client.py \
  --hdf5-path processed_data/my_task.hdf5 \
  --prompt "description of my task"
```

### Step 5: Deploy to Robot
```bash
# If validation looks good, deploy!
python h1_remote_client.py --host <server> --port 8000
```

## Common Command Line Patterns

### Custom Dataset and Prompt
```bash
python h1_policy_viz_client.py \
  --hdf5-path processed_data/pick_place.hdf5 \
  --prompt "pick up the cup and place it on the shelf"
```

### Different Policy Server
```bash
python h1_policy_viz_client.py \
  --host 192.168.1.100 \
  --port 8080
```

### Fast Playback
```bash
python h1_policy_viz_client.py \
  --fps 60 \
  --start-frame 100
```

### Lightweight Mode (No Meshes)
```bash
python h1_policy_viz_client.py \
  --no-load-meshes
```

### Using Environment Variables
```bash
# Set defaults
export HDF5_PATH="processed_data/my_task.hdf5"
export HOST="192.168.1.100"
export PORT="8000"
export PROMPT="my custom task"

# Run with defaults
./run_policy_viz.sh

# Or override specific values
HDF5_PATH="processed_data/other_task.hdf5" ./run_policy_viz.sh
```

## Debugging Tips

### Check Server Connection
```bash
# Test if policy server is reachable
curl http://<host>:<port>/health
```

### Verify Dataset Format
```bash
python -c "
import h5py
f = h5py.File('processed_data/circular.hdf5', 'r')
print('Keys:', list(f.keys()))
print('Actions shape:', f['action'].shape)
print('Qpos shape:', f['observations/qpos'].shape)
print('Images:', list(f['observations/images'].keys()))
"
```

### Monitor Policy Latency
The visualization client displays inference timing in the status panel. For real-time control, aim for:
- < 50ms for 20Hz control
- < 33ms for 30Hz control
- < 20ms for 50Hz control

### Compare Actions
```python
# Load predicted vs ground truth and compare
import numpy as np
import h5py

# Ground truth
with h5py.File('processed_data/task.hdf5', 'r') as f:
    gt_actions = f['action'][frame_idx:frame_idx+10]

# Predicted (from policy)
pred_actions = ...  # From inference

# Compare
diff = np.abs(gt_actions - pred_actions)
print(f"Mean difference: {diff.mean()}")
print(f"Max difference: {diff.max()}")
```

## Tips for Best Results

1. **Always validate before deploying**: Use the visualization client to catch issues early
2. **Test on diverse frames**: Don't just test on the first frame - try various scenarios
3. **Check action magnitudes**: Large deviations from ground truth may indicate issues
4. **Monitor inference time**: Ensure your policy can run at the target frequency
5. **Iterate quickly**: Use visualization to debug faster than deploying to robot

## File Locations

- **Datasets**: `examples/h1_control_client/processed_data/*.hdf5`
- **URDF/Meshes**: `examples/h1_control_client/assets/h1_2/`
- **Checkpoints**: `checkpoints/<config_name>/<run_name>/`
- **Client Scripts**: `examples/h1_control_client/*.py`
- **Utils**: `examples/h1_control_client/utils/*.py`

## Next Steps

- Read `README_POLICY_VIZ.md` for detailed visualization client documentation
- Check `README.md` for full setup and deployment instructions
- See `../../examples/droid/README_train.md` for training tips

