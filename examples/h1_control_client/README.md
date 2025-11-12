# H1 Control Client Setup

## Quick Start

### 1. Start Robot Client (on client computer)

In one terminal, establish SSH tunnel to P6000 server:

```bash
ssh -R 5007:localhost:5007 -L 8080:localhost:8080 yuxin@msc-server
```

In another terminal on the client computer, navigate to `openpi/examples/h1_control_client`:

```bash
conda activate h1_client
python3 h1_remote_client.py --listen-mode
```

### 2. Start Policy Visualization (on P6000 server)

In a new terminal, navigate to `openpi`:

```bash
source .venv/bin/activate
./run_policy_viz.sh --robot-execution
```

### 3. Start Policy Server (on P6000 server)

In another terminal:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_h1_finetune \
  --policy.dir=checkpoints/pi05_h1_finetune/pi05_h1_box_activate_h50_with_lan/999
```

## Notes

- The client computer and P6000 server must be connected via SSH tunnel
- Start the robot client first, then the policy viz, then the policy server
- Access the visualization at http://localhost:8081
