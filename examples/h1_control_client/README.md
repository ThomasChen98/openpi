# H1-2 Control & Validation Client

Complete toolkit for Unitree H1-2 humanoid robot with OpenPi policy integration. Two main tools:
1. **`h1_remote_client.py`** - Deploy policies on real robot
2. **`h1_policy_viz_client.py`** - Validate policies with visualization

---

## Two Ways to Use This

### 🤖 Robot Control (Deploy Trained Policy)

Control the H1-2 robot with a trained policy:

```bash
# Terminal 1: Start policy server (GPU machine)
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_h1_finetune \
  --policy.dir=checkpoints/pi05_h1_finetune/pi05_h1_H50/1999

# Terminal 2: Run robot client (control laptop)
cd examples/h1_control_client
python h1_remote_client.py \
  --server-host <gpu-ip> \
  --server-port 8000 \
  --head-camera-server-ip <robot-ip>
```

**What it does**: Connects to robot → streams cameras → queries policy @ 10Hz → executes actions @ 250Hz

### 📊 Policy Validation (Visualize Predictions)

Test policy predictions without touching the robot:

```bash
# Terminal 1: Start policy server
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_h1_finetune \
  --policy.dir=checkpoints/pi05_h1_finetune/pi05_h1_H50/1999

# Terminal 2: Run visualization client
cd examples/h1_control_client
python h1_policy_viz_client.py --hdf5-path processed_data/circular.hdf5
# Or: ./run_policy_viz.sh
```

**What it does**: Loads dataset → sends observations → receives action chunks → visualizes in 3D

**Interactive workflow:**
1. Select a frame from your dataset
2. Click "🤖 Infer Action Chunk" to get predictions
3. Click "▶️ Play Action Chunk" to visualize motion
4. Compare ground truth vs predictions

---

## Quick Setup

### 1. Clone Repository
```bash
git clone https://github.com/PhysicalIntelligence/openpi.git
cd openpi/examples/h1_control_client
```

### 2. Copy Required Files

**From your `unitree_h12_bimanual` repository:**
```bash
# Copy URDF models and meshes
cp -r /path/to/unitree_h12_bimanual/assets ./

# Copy SDK libraries (with compiled .so files)
cp -r /path/to/unitree_h12_bimanual/libraries ./
```

These are not in the OpenPi repo to avoid duplication.

### 3. Install Dependencies

**For robot control** (needs Pinocchio from conda):
```bash
# Create environment with Pinocchio
conda create -n h1_client python=3.10 pinocchio=3.1.0 numpy=1.26.4 -c conda-forge
conda activate h1_client

# Install other dependencies
./setup.sh
```

**For visualization only** (can use existing openpi venv):
```bash
# Already in openpi venv
cd examples/h1_control_client
uv pip install yourdfpy h5py viser tyro cv2 einops
```

---

## Tool Details

### 🤖 h1_remote_client.py

**Purpose**: Deploy trained policies on the actual H1-2 robot.

**Command line options:**
```bash
python h1_remote_client.py [OPTIONS]

Key options:
  --server-host HOST          Policy server IP
  --server-port PORT          Policy server port (default: 8000)
  --head-camera-server-ip IP  Robot head camera IP
  --left-hand-ip IP           Left hand IP (default: 192.168.123.211)
  --right-hand-ip IP          Right hand IP (default: 192.168.123.210)
  --network-interface IF      Network interface (default: eno1)
  --visualization             Enable IK visualization
  --prompt STR                Task instruction
  --duration SECONDS          Run duration
```

**Architecture:**
```
Robot Cameras → Client → Policy Server → Client → IK Solver → Robot Arms
    (USB/ZMQ)    (10Hz)    (WebSocket)   (250Hz)  (Pinocchio)   (DDS)
```

**Key features:**
- Real-time control @ 250Hz execution
- Policy queries @ 10Hz
- 3 cameras: head (via ZMQ), left/right wrist (USB)
- IK solver for smooth trajectories
- Dexterous hand control (Inspire FTP)
- Graceful degradation (continues with missing cameras)

**Testing without robot:**
```bash
# Use mock server
uv run scripts/mock_policy_server.py --port 8000
```

### 📊 h1_policy_viz_client.py

**Purpose**: Validate trained policies by visualizing predicted action chunks before deploying to robot.

**Command line options:**
```bash
python h1_policy_viz_client.py [OPTIONS]

Key options:
  --hdf5-path STR         Dataset path (default: processed_data/circular.hdf5)
  --urdf-path STR         Robot URDF (default: assets/h1_2/h1_2.urdf)
  --host STR              Policy server host (default: 0.0.0.0)
  --port INT              Policy server port (default: 8000)
  --prompt STR            Task instruction
  --fps FLOAT             Playback speed (default: 30.0)
  --start-frame INT       Initial frame
  --no-load-meshes        Faster loading
```

**Interactive UI:**
- **Frame selection**: Choose observation from dataset
- **Inference**: Get action chunk predictions from policy
- **Playback**: Animate predicted motions
- **Comparison**: Toggle ground truth vs predictions
- **Timing**: Monitor inference latency
- **Camera views**: Display dataset images

**Why use this?**
- ✅ Catch policy errors before robot deployment
- ✅ Test on diverse scenarios from your dataset
- ✅ Verify action chunks make sense
- ✅ Measure inference speed for real-time feasibility
- ✅ Compare predictions with ground truth
- ✅ Much faster iteration than robot testing

**Dataset format:**
```
/action                           # (N, 14 or 26) - Joint actions
/observations/
  /qpos                          # (N, 14 or 26) - Joint positions
  /images/
    /ego_cam                     # (N, H, W, 3) or JPEG
    /cam_left_wrist              # (N, H, W, 3) or JPEG
    /cam_right_wrist             # (N, H, W, 3) or JPEG
```

### 🔄 data_replay.py

**Purpose**: Replay recorded demonstrations without any policy.

```bash
cd utils
python data_replay.py --hdf5-path ../processed_data/circular.hdf5
```

**Use for:**
- Verify data collection worked
- Inspect trajectories frame-by-frame
- Check camera quality

---

## Action Space

51-dimensional action space matching H1-2 URDF:

| Dimensions | Description | Control |
|------------|-------------|---------|
| 0-12 | Legs + Torso | Locked |
| **13-26** | **Arms (14 DOF)** | **Active** |
| 27-50 | Hands (24 joints) | 12 actuated |

**Arm joints we control (14 DOF):**
```
Left arm:  13-19 (shoulder_pitch/roll/yaw, elbow_pitch/roll, wrist_pitch/yaw)
Right arm: 20-26 (shoulder_pitch/roll/yaw, elbow_pitch/roll, wrist_pitch/yaw)
```

---

## Development Workflow

### 1. Collect Data
Use teleoperation or other methods to collect demonstrations.

### 2. Convert to LeRobot Format
```bash
uv run training_data/h1/convert_h1_data_to_lerobot.py \
  --data_dir examples/h1_control_client/processed_data/my_task.hdf5 \
  --repo_id username/h1_my_task \
  --num_repeats 256
```

### 3. Train Policy
```bash
uv run scripts/train.py \
  --config pi05_h1_finetune \
  --dataset username/h1_my_task
```

### 4. Validate with Visualization ⭐
```bash
# Terminal 1: Policy server
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_h1_finetune \
  --policy.dir=checkpoints/pi05_h1_finetune/latest

# Terminal 2: Visualization
python h1_policy_viz_client.py \
  --hdf5-path processed_data/my_task.hdf5 \
  --prompt "my task description"
```

**Check:**
- ✓ Actions look reasonable?
- ✓ Inference fast enough?
- ✓ Predictions match ground truth?

### 5. Deploy to Robot
```bash
python h1_remote_client.py \
  --server-host <gpu-ip> \
  --server-port 8000 \
  --prompt "my task description"
```

---

## Camera Setup

### Head Camera (on robot)
```bash
# On robot (Terminal 1)
cd ~/xr_teleoperate
python3 image_server.py
```
- RealSense camera on robot head
- Streams via ZMQ to control laptop
- Configure serial number in image_server.py

### Wrist Cameras (on control laptop)
- Left wrist: RealSense D405, serial `218622271789`
- Right wrist: RealSense D405, serial `241222076627`
- Connected directly via USB to control laptop
- Auto-detected by client

**Graceful degradation**: Client continues with dummy images if cameras fail.

---

## Troubleshooting

### Setup Issues

**"pinocchio not found"**
```bash
# Pinocchio ONLY available via conda (not pip):
conda create -n h1_client python=3.10 pinocchio=3.1.0 -c conda-forge
```

**"unitree_sdk2py not found"**
```bash
cd libraries/unitree_sdk2_python && pip install -e .
cd libraries/inspire_hand_sdk && pip install -e .
```

**"yourdfpy not found" (for visualization)**
```bash
uv pip install yourdfpy h5py viser tyro
```

### Connection Issues

**"Connection refused"**
- Check policy server is running: `curl http://<host>:<port>/health`
- Check firewall: `sudo ufw allow 8000/tcp`
- Verify network: `ping <server-ip>`

**"Hand bridges failing"**
```bash
# Check hand IPs
ping 192.168.123.211  # Left
ping 192.168.123.210  # Right

# Check network interface
ip addr show
```

### Runtime Issues

**"Robot not moving"**
1. Robot in correct mode?
2. Arms unlocked?
3. Policy returning valid actions?
4. Enable `--visualization` to debug

**"Inference too slow"**
- Target: <50ms for 20Hz, <33ms for 30Hz
- Check GPU utilization on policy server
- Try simpler model or optimize

**"Actions look wrong" (in viz client)**
1. Toggle ground truth vs predicted
2. Try different frames
3. Check if training data was good
4. Retrain with more data

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│              GPU Server (Policy Inference)               │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  serve_policy.py                                   │ │
│  │  • Receives observations (images + state)          │ │
│  │  • Returns action chunks (H, 14)                   │ │
│  └────────────────────────────────────────────────────┘ │
└───────────────────────┬──────────────────────────────────┘
                        │ WebSocket
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌─────────────────────┐      ┌─────────────────────────┐
│ h1_remote_client.py │      │ h1_policy_viz_client.py │
│                     │      │                         │
│ Robot Control       │      │ Policy Validation       │
│ • Real cameras      │      │ • Dataset observations  │
│ • 10Hz query        │      │ • 3D visualization      │
│ • 250Hz execution   │      │ • Action chunk playback │
│ • IK solver         │      │ • Ground truth compare  │
│                     │      │                         │
│        ↓            │      │        ↓                │
│   Real Robot        │      │   Viser Display         │
└─────────────────────┘      └─────────────────────────┘
```

---

## Files Structure

```
h1_control_client/
├── h1_remote_client.py          # Robot control
├── h1_policy_viz_client.py      # Policy validation
├── run_policy_viz.sh            # Convenience launcher
│
├── utils/
│   ├── data_replay.py           # Replay demonstrations
│   └── weighted_moving_filter.py
│
├── robot_control/
│   ├── robot_arm_ik.py          # IK solver
│   └── robot_arm.py             # Robot controller
│
├── assets/
│   └── h1_2/
│       ├── h1_2.urdf
│       └── meshes/
│
├── libraries/
│   ├── unitree_sdk2_python/
│   └── inspire_hand_sdk/
│
├── processed_data/              # Your HDF5 datasets
│   └── circular.hdf5
│
├── setup.sh                     # Installation script
└── requirements.txt
```

---

## Quick Reference

### Common Commands

**Start policy server:**
```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_h1_finetune \
  --policy.dir=checkpoints/path/to/checkpoint
```

**Validate policy (safe):**
```bash
python h1_policy_viz_client.py --hdf5-path processed_data/task.hdf5
```

**Deploy to robot:**
```bash
python h1_remote_client.py --server-host <ip> --server-port 8000
```

**Replay data (no policy):**
```bash
cd utils && python data_replay.py --hdf5-path ../processed_data/task.hdf5
```

### Tips

1. **Always validate first**: Use viz client before robot deployment
2. **Test diverse frames**: Don't just test on first frame
3. **Monitor timing**: Ensure inference meets real-time requirements
4. **Compare carefully**: Large deviations = potential issues
5. **Iterate fast**: Viz client is much faster than robot testing

---

## Related Documentation

- **OpenPi Setup**: `../../SETUP_SUMMARY.md`
- **Training Guide**: `../../examples/droid/README_train.md`
- **Policy Server**: `../../scripts/serve_policy.py`
- **Mock Server**: `../../scripts/mock_policy_server.py`
- **Usage Examples**: `USAGE_EXAMPLES.md`

---

**Built for**: Unitree H1-2 + Inspire FTP Hands  
**Integrated with**: OpenPi Policy Framework  
**Ready to deploy!** 🚀
