# G1 Control Client

Visualization and control client for the Unitree G1 robot with Dex3 hands.

## Features

- **Data Replay**: Visualize recorded teleoperation episodes from HDF5 files
- **Policy Visualization**: Run policy inference and visualize predicted action chunks
- **Robot Execution**: Execute policies on the real G1 robot (requires robot connection)
- **Locomotion Display**: View mode_machine state, IMU data, and controller inputs

## Directory Structure

```
g1_control_client/
├── assets/
│   └── g1/ -> symlink to xr_teleoperate/assets/g1
├── robot_control/
│   └── __init__.py  (imports from xr_teleoperate)
├── utils/
│   ├── __init__.py
│   └── data_replay.py
├── g1_data_replay.py
├── g1_policy_viz_client.py
├── g1_remote_client.py (for robot execution)
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Data Replay

Replay recorded G1 teleoperation data with visualization:

```bash
python g1_data_replay.py --hdf5_path /mnt/ssd1/yuxin/g1_data/cabinetbottle/episode_2.hdf5
```

Open http://localhost:8080 in your browser to view the visualization.

### 2. Policy Visualization

Run policy inference on recorded data:

```bash
# First, start the policy server (in another terminal)
uv run scripts/serve_policy.py policy:checkpoint --policy.config=g1_config --policy.dir=checkpoints/g1

# Then run the visualization client
python g1_policy_viz_client.py --data-path /mnt/ssd1/yuxin/g1_data/cabinetbottle/episode_2.hdf5
```

## Data Format

### G1 HDF5 Format (28 DOF)

The G1 uses 28 DOF for upper body control:
- **[0:7]** Left arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)
- **[7:14]** Right arm
- **[14:21]** Left hand (thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1)
- **[21:28]** Right hand

### Locomotion Data

If recorded with `--motion` flag, the HDF5 files also contain:

**loco_state (17 DOF)**:
- [0] mode_machine (robot state: 0=damping, 1=stand, 5=walk, etc.)
- [1:4] IMU RPY (roll, pitch, yaw in radians)
- [4:8] IMU quaternion (w, x, y, z)
- [8:11] Accelerometer (x, y, z in m/s^2)
- [11:14] Gyroscope (x, y, z in rad/s)
- [14:17] Leg joints (left_knee, right_knee, avg_knee as height proxy)

**loco_action (20 DOF)**:
- [0:4] Joysticks (Lx, Ly, Rx, Ry in range [-1, 1])
- [4:20] Buttons (L1, L2, R1, R2, A, B, X, Y, Up, Down, Left, Right, Select, Start, F1, F3)

## GUI Controls

### Playback Control
- **Frame Slider**: Jump to specific frame
- **Play/Pause**: Start/stop automatic playback
- **Speed**: Adjust playback speed (0.1x - 5.0x)

### Visualization
- **Show Ground Truth**: Display recorded qpos
- **Show Predicted Actions**: Display policy predictions
- **Action Index**: Scrub through predicted action chunk

### Locomotion State (if available)
- **Mode**: Current robot FSM state
- **RPY**: IMU orientation in degrees
- **Accel**: Accelerometer readings
- **Gyro**: Gyroscope readings
- **Knee Angles**: Leg joint positions

### Controller Input (if available)
- **Joysticks**: Left/Right stick positions
- **Active Buttons**: Currently pressed buttons

## Comparison with H1 Client

| Feature | H1-2 | G1 |
|---------|------|-----|
| Action DOF | 14 or 26 | 28 |
| Hand type | Inspire (6 joints/hand) | Dex3 (7 joints/hand) |
| URDF | h1_2.urdf | g1_body29_hand14.urdf |
| Cameras | ego + wrist cameras | ego camera only |
| Loco data | Not recorded | loco_state [17], loco_action [20] |

## Troubleshooting

### URDF not loading
Ensure the symlink to xr_teleoperate assets is correct:
```bash
ls -la assets/g1/
# Should point to ../../xr_teleoperate/assets/g1
```

### Policy server connection failed
Check that the policy server is running and accessible:
```bash
curl http://localhost:8000/metadata
```

### Missing dependencies
Install required packages:
```bash
pip install -r requirements.txt
```
