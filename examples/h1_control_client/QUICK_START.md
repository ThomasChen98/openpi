# H1-2 Control Client - Quick Start Guide

## 🎯 Goal

Control your Unitree H1-2 robot arms using OpenPi policies with a single self-contained directory.

## ⚡ Setup (3 Steps)

### 1. Clone Repository
```bash
git clone https://github.com/PhysicalIntelligence/openpi.git
cd openpi/examples/h1_control_client
```

### 2. Run Setup
```bash
./setup.sh
```

### 3. Test Installation
```bash
python test_setup.py
```

If all tests pass: ✅ Ready to go!

## 🚀 Usage

### Start Policy Server (GPU Machine)

```bash
cd /path/to/openpi
uv run scripts/mock_policy_server.py --port 8000
```

### Run H1-2 Client (Robot Laptop)

```bash
python h1_remote_client.py --server-host <gpu-ip> --server-port 8000
```

## 📋 What It Does

1. **Connects** to OpenPi policy server
2. **Queries** policy at 10Hz (gets 50-step action chunks)
3. **Solves** IK for smooth trajectories  
4. **Executes** joint commands at 250Hz on robot
5. **Controls** dexterous hands (6 DOF each)

## 🔧 Common Commands

```bash
# Local testing (mock server + client on same machine)
# Terminal 1:
uv run ../../scripts/mock_policy_server.py

# Terminal 2:
python h1_remote_client.py

# Remote setup (server on GPU, client on robot)
python h1_remote_client.py --server-host 192.168.1.100

# Enable IK visualization
python h1_remote_client.py --visualization

# Run for specific duration
python h1_remote_client.py --duration 60  # 60 seconds
```

## 📁 Directory Contents

```
h1_control_client/
├── h1_remote_client.py    ← Main script (run this!)
├── setup.sh               ← Installation
├── test_setup.py          ← Verify setup
├── README.md              ← Full documentation
├── robot_control/         ← IK solver
├── utils/                 ← Utilities
├── assets/                ← URDF files
└── libraries/             ← SDKs (self-contained)
```

## ⚙️ Configuration

Default values (change in command line):
- Server: `localhost:8000`
- Left hand IP: `192.168.123.211`
- Right hand IP: `192.168.123.210`
- Network interface: `eno1`

## 🆘 Troubleshooting

**Setup failed?**
```bash
# Pinocchio issues - use conda:
conda install pinocchio=3.1.0 -c conda-forge

# Then retry setup:
./setup.sh
```

**Connection refused?**
```bash
# Check server is running and reachable:
ping <server-ip>
telnet <server-ip> 8000
```

**Robot not moving?**
```bash
# Verify robot is powered and in correct mode
# Check arms are unlocked
# Try visualization to debug IK:
python h1_remote_client.py --visualization
```

## 📚 More Info

- Full documentation: `README.md`
- OpenPi docs: `../../README.md`
- Training guide: `../../examples/droid/README_train.md`

---

**That's it!** From fresh laptop to robot control in 3 commands. 🚀

