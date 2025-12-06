# H1-2 Control Client

Control and training clients for the Unitree H1-2 humanoid robot with OpenPi policies.

## Documentation

Full documentation is in the `docs/` folder:

| Guide | Description |
|-------|-------------|
| [Teleoperation Pipeline](../../docs/TELEOPERATION_PIPELINE.md) | VR data collection & policy inference with `h1_remote_client.py` |
| [Auto-Training Pipeline](../../docs/AUTO_TRAINING_PIPELINE.md) | Human-in-the-loop training with `h1_execution_client.py` |
| [Pre-Training & Validation](../../docs/PRETRAINING_VALIDATION_PIPELINE.md) | Offline training & visualization with `h1_policy_viz_client.py` |

## Quick Reference

### Key Scripts

| Script | Purpose |
|--------|---------|
| `h1_remote_client.py` | Policy inference client (connects to server, executes on robot) |
| `h1_execution_client.py` | Auto-training client (state machine for iterative learning) |
| `h1_policy_viz_client.py` | Web-based visualizer for policy validation |
| `training_config.yaml` | Configuration for auto-training pipeline |

### Directory Structure

```
h1_control_client/
├── h1_remote_client.py       # Policy execution
├── h1_execution_client.py    # Auto-training
├── h1_policy_viz_client.py   # Visualization
├── training_config.yaml      # Configuration
├── robot_control/            # IK solver & robot controller
├── utils/                    # Episode writer, data tools
├── assets/                   # URDF files & meshes
└── libraries/                # Unitree SDK
```

### Common Commands

**Policy Inference:**
```bash
python h1_remote_client.py --server-host localhost --server-port 8000
```

**Auto-Training (robot side):**
```bash
python h1_execution_client.py --config training_config.yaml
```

**Visualization:**
```bash
python h1_policy_viz_client.py --hdf5-path data.hdf5 --host localhost --port 8000
```

## Network Setup

| Device | IP Address |
|--------|------------|
| Robot (Unitree) | 192.168.123.163 |
| Left Hand | 192.168.123.211 |
| Right Hand | 192.168.123.210 |

**SSH Tunnel (to GPU server):**
```bash
ssh -L 8000:localhost:8000 -L 8080:localhost:8080 P6000
```

## Configuration

Edit `training_config.yaml` for:
- Task name and description
- Policy checkpoint paths
- Training parameters
- Server host/port
- Data paths and rsync settings

See [Auto-Training Pipeline](../../docs/AUTO_TRAINING_PIPELINE.md) for full configuration options.
