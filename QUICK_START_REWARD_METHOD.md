# Quick Start: Using Reward Method Configuration

## TL;DR

Add this to your `training_config.yaml`:

```yaml
reward:
  method: "Ours"  # or "GVL"
  # ... rest of your reward config
```

That's it! The system will automatically use the specified reward model.

## Method Options

### "Ours" - Your Fine-tuned Qwen Model

```yaml
reward:
  method: "Ours"
  checkpoint_path: "/home/yuxin/Projects/openpi/third_party/emboided_reward/IB-checkpoint-750"
  task_instruction: "Pick up the bottle and insert it into the tray."
  max_frames: 20
  advantage_threshold: 0.3
```

- Uses your fine-tuned Qwen3VL model
- Fast (runs on local GPU)
- Free

### "GVL" - OpenAI GPT-5.2

```yaml
reward:
  method: "GVL"
  task_instruction: "Fold the towel into a small square."
  max_frames: 30
  advantage_threshold: 0.3
```

Before running:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

- Uses OpenAI's GPT-5.2 model
- Can parallelize across many workers
- Requires API key and costs money

## That's All!

The training script handles everything else automatically:
- Loads the correct model
- Validates requirements (checkpoint or API key)
- Uses appropriate Python environment
- Caches results per method

## Example Full Config

```yaml
# training_config.yaml
task:
  name: "bottle_insertion"
  description: "Pick up the bottle and insert it into the tray"

policy:
  config_name: "pi05_h1_auto"
  warmup_checkpoint: "/path/to/warmup/checkpoint"

training:
  max_epochs: 10
  labeling_mode: "action_chunk_advantage"  # or "reward_labeling"
  gpu_id: 0

reward:
  method: "Ours"  # ← NEW! Choose "Ours" or "GVL"
  checkpoint_path: "/path/to/qwen/checkpoint"  # Only for "Ours"
  task_instruction: "Pick up the bottle and insert it into the tray."
  max_frames: 20
  advantage_threshold: 0.3
  look_ahead_window: 80

robot:
  include_hands: false

policy_server:
  host: "localhost"
  port: 8000
```

Run training:
```bash
./scripts/integrated_training_h1.sh --config training_config.yaml
```

Done!
