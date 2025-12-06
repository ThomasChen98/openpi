# H1-2 Teleoperation Pipeline

This document covers two teleoperation modes for the H1-2 humanoid robot:
1. **VR Teleoperation** - Direct control via Pico VR headset
2. **Policy Inference** - Remote policy execution via `h1_remote_client.py`

---

## Prerequisites

### Hardware
- H1-2 Humanoid Robot with Inspire hands
- Station computer (laptop) connected to robot network
- Pico VR headset (for VR teleoperation)
- RealSense D405 wrist cameras (connected to station computer)

### Network Setup
- Robot IP: `192.168.123.163` (Unitree onboard computer)
- Left hand IP: `192.168.123.211`
- Right hand IP: `192.168.123.210`
- Station computer on same subnet

---

## Part 1: VR Teleoperation (Data Collection)

### Setup

1. **Turn on the robot and enter damping mode**
   - Press `L2+R2`, then `L1+A` to enter damping mode
   - Press `L2+A` to move to ready position

2. **Start the head camera server on the robot**
   ```bash
   # SSH into Unitree onboard computer
   ssh unitree@192.168.123.163
   # Password: Unitree0408

   # Start image server
   python3 image_server/image_server.py
   ```
   You should see the head camera start broadcasting.

3. **Start the teleoperation client on the station computer**
   ```bash
   cd ~/xr_teleoperate/teleop
   conda activate tv

   python3 teleop_hand_and_arm.py \
       --xr-mode=hand \
       --arm=H1_2 \
       --ee=inspire1 \
       --inspire-bridge \
       --network-interface=eno1 \
       --record \
       --use-multi-camera \
       --task-name=YOUR_TASK_NAME
   ```

   The robot will perform a startup sequence (raise arms, open/close hands twice).

4. **Connect the Pico VR headset**
   - Open the Pico browser and navigate to:
     ```
     https://10.41.253.121:8012?ws=wss://10.41.253.121:8012
     ```
   - Ensure the headset is on the correct WiFi network (not Berkeley Visitor)
   - Confirm you see the head camera feed in the TeleVuer window

### Operation

1. **Start teleoperation**
   - Click "Pass Through" on the TeleVuer tab
   - Press `r` to start teleoperation
   - The robot should now mirror your movements

2. **Record an episode**
   - Press `s` to start recording
   - Perform the manipulation task
   - Press `s` again to stop recording and save

3. **Tips for good recordings**
   - Face the correct direction (robot uses global positioning)
   - Use slow, deliberate movements with clear arm poses
   - The viewer has latency - you may prefer looking directly at the robot

4. **End the session**
   - Double-tap right side of VR to enter see-through mode
   - Press `q` in terminal to quit and reset robot
   - Close browser tab and remove headset
   - Put robot in damping mode (`L2+B`)
   - Terminate image_server on robot

### Data Output

Recorded episodes are saved to:
```
~/xr_teleoperate/teleop/data/{task_name}/episode_*.hdf5
```

---

## Part 2: Policy Inference (h1_remote_client.py)

Use this mode to execute trained policies on the robot.

### Architecture

```
┌─────────────────┐     WebSocket      ┌─────────────────┐
│   GPU Server    │◄──────────────────►│ Station Computer│
│  (P6000/GPU)    │     (port 8000)    │   (Laptop)      │
│                 │                    │                 │
│ serve_policy.py │                    │h1_remote_client │
│                 │                    │       ▼         │
└─────────────────┘                    │   H1-2 Robot    │
                                       └─────────────────┘
```

### Setup

1. **On the GPU server (P6000)** - Start the policy server:
   ```bash
   cd ~/openpi
   
   # Serve a trained checkpoint
   uv run scripts/serve_policy.py policy:checkpoint \
       --policy.config=pi0_fast_h1 \
       --policy.dir=checkpoints/pi0_fast_h1/my_task/999
   ```

2. **SSH tunnel from station computer**:
   ```bash
   # Forward policy server (8000) and visualizer (8080)
   ssh -R 5007:localhost:5007 -L 8080:localhost:8080 P6000
   ```

3. **On station computer** - Start image server on robot:
   ```bash
   ssh unitree@192.168.123.163
   python3 image_server/image_server.py
   ```

4. **On station computer** - Start the remote client:
   ```bash
   cd ~/openpi/examples/h1_control_client
   
   python h1_remote_client.py \
       --server-host localhost \
       --server-port 8000 \
       --prompt "your task description"
   ```

### Listen Mode (for Visualization Control)

Run the client in listen mode to accept commands from the visualizer:

```bash
python h1_remote_client.py \
    --listen-mode \
    --listen-port 5007 \
    --prompt "your task description"
```

Then on the GPU server, run the visualizer:
```bash
uv run python examples/h1_control_client/h1_policy_viz_client.py \
    --hdf5-path training_data/h1/sample.hdf5 \
    --robot-execution \
    --robot-host localhost \
    --robot-port 5007
```

Open `http://localhost:8080` in browser to:
- View camera feeds
- Step through recorded frames
- Execute policy predictions on the robot
- Reset robot to specific poses

### Command Reference

| Key | Action |
|-----|--------|
| `Ctrl+C` | Stop client and move robot to home |

---

## Troubleshooting

### Port in use
```bash
lsof -i :8012
kill -9 <PID>
```

### VR won't connect
- Verify Pico is on correct WiFi network
- Check firewall settings on station computer
- Restart the teleop script

### Camera not working
- Check USB connections for wrist cameras
- Verify image_server is running on robot
- Check serial numbers match expected values:
  - Left wrist: `218622271789`
  - Right wrist: `241222076627`

### Robot not responding
- Ensure robot is in ready mode (not damping)
- Check network interface setting (`--network-interface`)
- Verify hand IPs are correct

---

## References

- [xr_teleoperate repository](https://github.com/ThomasChen98/xr_teleoperate)
- [OpenPi documentation](../README.md)

