# H1-2 Remote Policy Client

Self-contained control system for Unitree H1-2 humanoid robot with Inspire dexterous hands, integrated with OpenPi policy server.

## What This Is

A complete, ready-to-deploy solution for controlling H1-2 arms remotely using OpenPi models. Just clone, setup, and run!

**Key Features:**
- Self-contained - all dependencies included
- Inverse kinematics solver (Pinocchio + CasADi)
- Dexterous hand control (Inspire FTP hands)
- Remote policy server integration (OpenPi)
- Easy setup on new laptops

## What's Included

```
h1_control_client/
├── h1_remote_client.py       # Main control script
├── setup.sh                    # One-command installation
├── requirements.txt            # Python dependencies
├── robot_control/              # IK solver & robot controller
│   ├── robot_arm_ik.py        # Pinocchio IK solver
│   └── robot_arm.py           # H1-2 low-level controller
├── utils/                      # Utilities
│   └── weighted_moving_filter.py
├── assets/                     # URDF and meshes
│   └── h1_2/
│       ├── h1_2.urdf
│       └── meshes/
└── libraries/                  # SDKs (self-contained)
    ├── unitree_sdk2_python/
    └── inspire_hand_sdk/
```

## Quick Start (New Laptop)

### 1. Clone OpenPi Repository

```bash
git clone https://github.com/PhysicalIntelligence/openpi.git
cd openpi/examples/h1_control_client
```

### 2. Run Setup Script

```bash
./setup.sh
```

This installs:
- Python dependencies (pinocchio, casadi, etc.)
- OpenPi client
- Unitree SDK
- Inspire Hand SDK

**Note:** If using conda, pinocchio installation is recommended:
```bash
conda create -n h1_client python=3.10 pinocchio=3.1.0 numpy=1.26.4 -c conda-forge
conda activate h1_client
./setup.sh
```

### 3. Configure Network

Edit IP addresses if needed:
```bash
# In h1_remote_client.py or use command-line args
--left-hand-ip 192.168.123.211
--right-hand-ip 192.168.123.210
--network-interface eno1
```

### 4. Start Policy Server (GPU Machine)

On your GPU server:
```bash
cd /path/to/openpi
uv run scripts/mock_policy_server.py --port 8000
```

For real models:
```bash
uv run scripts/serve_policy.py --env DROID --port 8000
```

### 5. Run H1-2 Client (Robot Laptop)

```bash
python h1_remote_client.py \
    --server-host <gpu-server-ip> \
    --server-port 8000
```

That's it! The robot will:
1. Connect to policy server
2. Move to home position
3. Start querying policy at 10Hz
4. Execute actions on robot at 250Hz

## Command Line Options

```bash
python h1_remote_client.py [OPTIONS]

Options:
  --server-host HOST          Policy server IP (default: localhost)
  --server-port PORT          Policy server port (default: 8000)
  --network-interface IF      Network interface (default: eno1)
  --left-hand-ip IP           Left hand IP (default: 192.168.123.211)
  --right-hand-ip IP          Right hand IP (default: 192.168.123.210)
  --visualization             Enable IK visualization (meshcat)
  --duration SECONDS          Run duration (default: infinite)
  -h, --help                  Show help message
```

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                GPU Server (Policy Inference)                 │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  mock_policy_server.py  or  serve_policy.py           │ │
│  │  • Receives observations                               │ │
│  │  • Returns action chunks (50, 51)                      │ │
│  └────────────────────────────────────────────────────────┘ │
└───────────────────────────────┬──────────────────────────────┘
                                │ Websocket (10Hz policy queries)
                                │
┌───────────────────────────────▼──────────────────────────────┐
│           Robot Control Laptop (H1-2 Local Controller)       │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  h1_remote_client.py                                   │ │
│  │                                                         │ │
│  │  ┌───────────────────┐  ┌────────────────────────────┐│ │
│  │  │ OpenPi Client     │  │ IK Solver (Pinocchio)      ││ │
│  │  │ • Get observation │  │ • EE pose → Joint angles   ││ │
│  │  │ • Query policy    │  │ • Smooth trajectory        ││ │
│  │  └───────────────────┘  └────────────────────────────┘│ │
│  │                                                         │ │
│  │  ┌─────────────────────────────────────────────────────┤ │
│  │  │ Robot Controller (H1_2_ArmController)              ││ │
│  │  │ • Execute joint commands @ 250Hz                   ││ │
│  │  │ • Control dexterous hands                          ││ │
│  │  └────────────────────────────────────────────────────┘│ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Unitree SDK  │  │ Inspire SDK  │  │  DDS Bridge  │      │
│  └───────┬──────┘  └──────┬───────┘  └──────┬───────┘      │
└──────────┼─────────────────┼─────────────────┼──────────────┘
           │                 │                 │
           ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │   H1-2 Arms  │  │ Left Hand    │  │ Right Hand   │
    │   (14 DOF)   │  │ (6 DOF)      │  │ (6 DOF)      │
    └──────────────┘  └──────────────┘  └──────────────┘
```

## Action Space

The system uses **51-dimensional action space** matching H1-2 URDF:

| Dimensions | Description | Range | Control |
|------------|-------------|-------|---------|
| 0-12 | Legs + Torso | [varies] | Locked (not controlled) |
| **13-26** | **Arms (14 DOF)** | **[-π, π]** | **Active control** |
| 27-50 | Hands (24 URDF joints) | [0, 1000] | 12 actuated, 12 mimic |

### Arm Joints (14 DOF - What We Control)
```
Left arm (7 DOF):  indices 13-19
  13: left_shoulder_pitch
  14: left_shoulder_roll
  15: left_shoulder_yaw
  16: left_elbow_pitch
  17: left_elbow_roll
  18: left_wrist_pitch
  19: left_wrist_yaw

Right arm (7 DOF): indices 20-26
  20: right_shoulder_pitch
  21: right_shoulder_roll
  22: right_shoulder_yaw
  23: right_elbow_pitch
  24: right_elbow_roll
  25: right_wrist_pitch
  26: right_wrist_yaw
```

## Camera Integration (TODO)

Replace dummy images in `h1_remote_client.py`:

```python
def get_observation(self) -> dict:
    # TODO: Add your camera interface here
    # external_image = your_camera.get_external_view()
    # wrist_image = your_camera.get_wrist_view()
    
    # Process images
    from openpi_client import image_tools
    external_processed = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(external_image, 224, 224)
    )
    
    observation = {
        "observation/image": external_processed,
        "observation/wrist_image": wrist_image,
        "observation/state": current_arm_q,
        "prompt": "your task instruction",
    }
    return observation
```

## Testing Without Robot

Test the setup without actual hardware:

```bash
# Terminal 1: Start mock server
cd /path/to/openpi
uv run scripts/mock_policy_server.py --port 8000

# Terminal 2: Test connection only
python3 << 'EOF'
from openpi_client import websocket_client_policy
import numpy as np

client = websocket_client_policy.WebsocketClientPolicy(
    host="localhost",
    port=8000
)

print("Connected:", client.get_server_metadata())

obs = {
    "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    "observation/state": np.zeros(14),
    "prompt": "test"
}

result = client.infer(obs)
print(f"Got actions: {result['actions'].shape}")
EOF
```

## 🔍 Troubleshooting

### Setup Issues

**"pinocchio not found"**
```bash
# Use conda for pinocchio:
conda install pinocchio=3.1.0 -c conda-forge
```

**"unitree_sdk2py not found"**
```bash
# Reinstall SDKs:
cd libraries/unitree_sdk2_python && pip install -e .
cd libraries/inspire_hand_sdk && pip install -e .
```

### Connection Issues

**"Connection refused"**
```bash
# Check server is running:
ping <server-ip>
telnet <server-ip> 8000

# Check firewall:
sudo ufw allow 8000/tcp
```

**"Hand bridges failing"**
```bash
# Check hand IPs are correct:
ping 192.168.123.211  # Left hand
ping 192.168.123.210  # Right hand

# Check network interface:
ip addr show  # Find correct interface name
```

### Runtime Issues

**"Robot not moving"**
1. Check robot is in correct mode
2. Verify arms are unlocked
3. Check policy server is returning valid actions
4. Enable visualization to see IK solver output: `--visualization`

**"Actions out of range"**
- Mock server should return actions in valid ranges
- Check server is configured for H1-2: `action_dim=51, action_horizon=50`

## File Structure Details

### Core Files

- **`h1_remote_client.py`** - Main control script
  - Connects to OpenPi policy server
  - Queries policy at 10Hz
  - Executes actions via robot controller

- **`robot_control/robot_arm_ik.py`** - IK solver
  - Pinocchio + CasADi optimization
  - Converts EE poses → joint angles
  - Smooth trajectory generation

- **`robot_control/robot_arm.py`** - Robot controller
  - Low-level DDS communication
  - Joint command execution @ 250Hz
  - Hand control via Modbus bridges

### Dependencies

- **Pinocchio** - Rigid body dynamics
- **CasADi** - Nonlinear optimization  
- **Meshcat** - 3D visualization (optional)
- **Unitree SDK** - Robot communication
- **Inspire SDK** - Hand control

## Next Steps

1. Clone and setup (you are here!)
2. Integrate real cameras
3. Test with mock server
4. Collect demonstration data
5. Train custom H1-2 policy
6. Deploy trained model

## Related Documentation

- **OpenPi Setup**: `../../SETUP_SUMMARY.md`
- **Mock Server**: `../../scripts/mock_policy_server.py`
- **Real Server**: `../../scripts/serve_policy.py`
- **Training**: `../../examples/droid/README_train.md`

## Support

For issues:
1. Check this README's troubleshooting section
2. Review OpenPi documentation: `../../README.md`
3. Open issue on OpenPi GitHub

---

**Built for**: Unitree H1-2 + Inspire FTP Hands  
**Integrated with**: OpenPi Policy Framework  
**Ready to deploy!** 

