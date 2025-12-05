# H1 Training Pipeline

A human-in-the-loop training system for the H1-2 robot that alternates between policy execution and human corrections, with continuous data recording.

## Overview

The training pipeline implements a state machine that:
1. Waits for a policy server to signal new weights are ready
2. Runs the policy while recording data
3. Allows human corrections in damping mode
4. Syncs data to a remote server for retraining

```
┌─────────┐     ┌───────┐     ┌───────────┐     ┌──────────┐
│ WAITING │────▶│ READY │────▶│ EXECUTING │────▶│ LABELING │
└─────────┘     └───────┘     └───────────┘     └──────────┘
     ▲                              │                  │
     │                              │ press 's'        │
     │                              ▼                  ▼
┌─────────┐     ┌──────────┐     ┌─────────┐     ┌─────────┐
│ SYNCING │◀────│ DECIDING │◀────│ SAVING  │◀────│ DAMPING │
└─────────┘     └──────────┘     └─────────┘     └─────────┘
                     │                               │
                     │ press 'y'                     │ press 'e'
                     └───────────▶ READY ◀───────────┘
```

## What It Polls For

The client polls the policy server's `/training_status` HTTP endpoint:

```bash
GET http://<policy_server>:<port>/training_status
```

**Response:**
```json
{
  "ready": true,    // Whether new weights are ready for inference
  "epoch": 5        // Current training epoch number
}
```

The client transitions from WAITING → READY when:
- `ready` is `true`
- `epoch` is greater than the last seen epoch

This allows the training loop to detect when the policy server has finished a training iteration and loaded new weights.

## Data Storage Format

Data is saved to HDF5 files in the configured `save_dir`:

```
./examples/h1_control_client/h1_data_auto
└── training_session_epoch1/
    ├── episode_0.hdf5
    ├── episode_1.hdf5
    └── ...
```

### HDF5 Structure

```
episode_N.hdf5
├── observations/
│   ├── qpos          [T, 14]   float32  - Joint positions
│   └── images/
│       ├── ego_cam         [T,]  vlen uint8  - Head camera (JPEG)
│       ├── cam_left_wrist  [T,]  vlen uint8  - Left wrist (JPEG)
│       └── cam_right_wrist [T,]  vlen uint8  - Right wrist (JPEG)
├── action            [T, 14]   float32  - Policy outputs (zeros during damping)
├── phase             [T,]      string   - "policy" or "human"
└── attrs:
    ├── episode_length
    ├── fps
    ├── timestamp
    ├── policy_frames
    └── human_frames
```

### Recording Details

- **Rate**: 50Hz (matches robot control rate)
- **Images**: JPEG compressed at quality=85
- **All cameras recorded**: Head, left wrist, right wrist
- **Continuous**: Records during both policy execution AND damping mode

## Tester's Guide

### Prerequisites

1. **Robot Setup**:
   - H1-2 robot powered on and standing
   - Inspire hands connected to laptop via Modbus
   - Head camera streaming via `image_server.py` on robot
   - Wrist cameras (RealSense D405) connected to laptop

2. **Policy Server**:
   - Running on GPU server with `/training_status` endpoint
   - Example: `uv run scripts/serve_policy.py --port 8000`

3. **Network**:
   - Laptop can reach robot (192.168.123.x network)
   - Laptop can reach GPU server
   - SSH keys set up for rsync to P6000

### Configuration

Edit `training_config.yaml`:

```yaml
policy_server:
  host: "msc-server"     # Your GPU server hostname/IP
  port: 8000

robot:
  head_camera_server_ip: "192.168.123.163"  # Robot IP

rsync:
  target: "yuxin@p6000:/data/h1_training/"  # Your data destination
```

### Running a Training Session

```bash
cd examples/h1_control_client
python h1_execution_client.py --config training_config.yaml
```

### Controls

| Key | Action |
|-----|--------|
| `y` | Yes / Confirm |
| `n` | No / Decline |
| `s` | Stop execution, enter damping mode |
| `e` | End epoch, save data |
| `Ctrl+C` | Emergency exit (saves current data) |

### Typical Session Flow

1. **Startup**: Robot initializes, moves to home position
   ```
   [WAITING] Polling policy server for new weights...
   ```

2. **Policy Ready**: New weights detected
   ```
   [READY] New weights available!
     Press 'y' to start epoch 1, 'n' to skip: _
   ```

3. **Execution**: Robot runs policy inference
   ```
   [EXECUTING] Running policy (epoch 1)
     Press 's' to stop execution and enter damping mode
   ```
   - Watch the robot's behavior
   - Press `s` when you want to stop (either because it succeeded or failed)

4. **Damping Mode**: Make corrections
   ```
   [DAMPING] Robot in damping mode - adjust pose freely
     Press 'e' to end epoch and save
   ```
   - Physically move the robot's arms to desired positions
   - Guide through the correct motion
   - All movements are recorded as human demonstrations

5. **Save**: Episode saved to HDF5
   ```
   [SAVING] Saving episode data...
     Saved: ./examples/h1_control_client/h1_data_auto/training_session_epoch1/episode_0.hdf5
     Total timesteps: 2847
     Policy frames: 1200, Human frames: 1647
   ```

6. **Continue or End**:
   ```
   [DECIDING] What's next?
     Press 'y' to start another epoch
     Press 'n' to finish training session
   ```

7. **Sync**: Data uploaded to remote server
   ```
   [SYNCING] Uploading data to remote server...
   ```

### Tips for Testers

1. **Guide smoothly**: During damping mode, move the robot through the task smoothly and at a reasonable pace.

2. **Don't rush**: Take time during human corrections to demonstrate the task properly.

3. **Watch for limits**: Be aware of joint limits when moving the robot manually.

4. **Multiple epochs**: Do several epochs per session for more data. You can vary:
   - Starting positions
   - Object placements
   - Speed of corrections

5. **Emergency stop**: `Ctrl+C` will save current data and exit safely.

### Troubleshooting

**"Cameras ready: All using dummy images"**
- Check that `image_server.py` is running on robot
- Verify wrist cameras are connected (check `ls /dev/video*`)
- Check RealSense serial numbers match config

**"Failed to poll training status"**
- Verify policy server is running
- Check `host` and `port` in config
- Test with: `curl http://<host>:<port>/training_status`

**Robot not moving during execution**
- Check policy server is returning valid actions
- Verify WebSocket connection established

**Rsync fails**
- Verify SSH key authentication works: `ssh yuxin@p6000`
- Check target path exists on remote server

