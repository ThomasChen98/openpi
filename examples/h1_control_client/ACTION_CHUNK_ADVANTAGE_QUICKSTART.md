# Action Chunk Advantage - Quick Start Guide

## What is it?

Instead of labeling entire episodes as "good" or "bad", action chunk advantage labels **each frame individually** based on whether the actions from that state lead to good future outcomes.

**Example:**
```
Episode with 200 frames:
  Frame 0-50:   Advantage=False  (robot is confused, wrong approach)
  Frame 51-120: Advantage=True   (robot found good strategy)
  Frame 121-200: Advantage=False (robot made a mistake at the end)
```

## Quick Setup

### 1. Update Your Training Config

Edit `training_config_fold_towel_jan10.yaml`:

```yaml
training:
  labeling_mode: "action_chunk_advantage"  # Changed from "human_labeling"

reward:
  checkpoint_path: "/path/to/qwen/checkpoint-750"
  task_instruction: "Fold the towel into a small square."
  max_frames: 30
  advantage_threshold: 0.33      # Top 33% get Advantage=True
  look_ahead_window: 80          # Look 80 frames ahead
```

### 2. Run Training Pipeline

```bash
./scripts/integrated_training.sh --config training_config_fold_towel_jan10.yaml
```

That's it! The pipeline will automatically:
1. Collect data
2. Compute action chunk advantages
3. Convert to LeRobot format with per-frame labels
4. Train the model

## What Happens Behind the Scenes

### During Data Conversion

```bash
# 1. Pre-compute rewards (Qwen3-VL)
Computing rewards for episode_000000.parquet...
Computing rewards for episode_000001.parquet...
...

# 2. Pre-compute visual embeddings (DINOv3)
Computing embeddings for episode_000000.parquet...
Computing embeddings for episode_000001.parquet...
...

# 3. Compute action chunk advantages
Computing advantages for episode_000000.parquet...
  episode_000000.parquet: 234/600 frames with advantage (39.0%)
Computing advantages for episode_000001.parquet...
  episode_000001.parquet: 156/600 frames with advantage (26.0%)
...

SUMMARY STATISTICS
================================================================================
Total episodes: 20
Total frames: 12000
Frames with Advantage=True: 3600 (30.0%)
Frames with Advantage=False: 8400 (70.0%)

# 4. Convert to LeRobot format
Processing episode 1/20...
  [DEBUG] Episode advantages: 234/600 frames True (39.0%)
  Saved episode 1 with 600 frames
Processing episode 2/20...
  [DEBUG] Episode advantages: 156/600 frames True (26.0%)
  Saved episode 2 with 600 frames
...
```

### During Training

```bash
# Training starts
INFO: ACTION CHUNK ADVANTAGE MODE DETECTED
================================================================================
Dataset has 'advantage' feature - applying ActionChunkAdvantagePrompt transform
Each chunk will be labeled with Advantage=True/False based on its start frame
Monitor the debug output to verify advantage distribution during training
================================================================================

# Every 100 chunks processed:
[ActionChunkAdvantage] Processed 100 chunks: True=32 (32.0%), False=68 (68.0%)
[ActionChunkAdvantage] Processed 200 chunks: True=64 (32.0%), False=136 (68.0%)
[ActionChunkAdvantage] Processed 300 chunks: True=98 (32.7%), False=202 (67.3%)
...
```

### Cache Files Created

For each episode, three cache files are created:

```
epoch_0/data/chunk-000/
  ├── episode_000000.parquet
  ├── episode_000000_reward.pkl                      # Dense rewards (600 floats)
  ├── episode_000000_ego_image_embeddings.pkl        # Visual embeddings (600 x 384)
  └── episode_000000_action_chunk_advantages.pkl     # Per-frame advantages (600 bools)
```

These are cached, so re-running is fast!

## Manual Testing

### Test on a Single Episode

```bash
# 1. Compute advantages
python examples/h1_control_client/compute_action_chunk_advantages.py \
    --data-dir examples/h1_control_client/h1_data_lerobot/fold_towel/epoch_0/data/chunk-000 \
    --task-instruction "Fold the towel into a small square." \
    --checkpoint-path third_party/emboided_reward/checkpoint-750

# 2. Check the results
python -c "
import pickle
import numpy as np

# Load advantages
adv = pickle.load(open('examples/h1_control_client/h1_data_lerobot/fold_towel/epoch_0/data/chunk-000/episode_000000_action_chunk_advantages.pkl', 'rb'))

print(f'Total frames: {len(adv)}')
print(f'Advantage=True: {adv.sum()} ({100*adv.sum()/len(adv):.1f}%)')
print(f'Advantage=False: {(~adv).sum()} ({100*(~adv).sum()/len(adv):.1f}%)')

# Show advantage pattern
print('\nAdvantage pattern (first 100 frames):')
print(''.join(['T' if a else 'F' for a in adv[:100]]))
"
```

### Visualize Advantages

```python
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load data
rewards = pickle.load(open('episode_000000_reward.pkl', 'rb'))
advantages = pickle.load(open('episode_000000_action_chunk_advantages.pkl', 'rb'))

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Plot rewards
ax1.plot(rewards, label='Predicted Reward')
ax1.set_ylabel('Reward')
ax1.legend()
ax1.grid(True)

# Plot advantages
ax2.fill_between(range(len(advantages)), 0, advantages, alpha=0.5, label='Advantage=True')
ax2.set_ylabel('Advantage')
ax2.set_xlabel('Frame')
ax2.set_ylim(-0.1, 1.1)
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('advantage_visualization.png')
print('Saved to advantage_visualization.png')
```

## Comparing Modes

### Episode-Level (reward_labeling)
```yaml
labeling_mode: "reward_labeling"
```
- All frames in episode get same label
- Fast (no similarity search needed)
- Good for clearly good/bad episodes

### Action Chunk (action_chunk_advantage)
```yaml
labeling_mode: "action_chunk_advantage"
```
- Each frame gets individual label
- Slower (needs similarity search)
- Better for mixed-quality episodes
- More fine-grained learning signal

## Tuning Parameters

### advantage_threshold
- **0.2:** Top 20% (more selective, fewer True labels)
- **0.33:** Top 33% (balanced, recommended)
- **0.5:** Top 50% (more permissive, more True labels)

### look_ahead_window
- **40:** Short-term outcomes (quick tasks)
- **80:** Medium-term outcomes (recommended)
- **120:** Long-term outcomes (complex tasks)

### max_frames
- **20:** Faster, less accurate rewards
- **30:** Balanced (recommended)
- **40:** Slower, more accurate rewards

## Verifying Everything Works

### Run the Verification Script

After converting data, verify advantages are properly stored:

```bash
python examples/h1_control_client/verify_action_chunk_advantages.py \
    --dataset-path examples/h1_control_client/h1_data_lerobot/fold_towel/epoch_0
```

**Expected output:**
```
ACTION CHUNK ADVANTAGE VERIFICATION
================================================================================
STEP 1: Verify Advantages Stored in Dataset
================================================================================
✅ 'advantage' feature found in dataset metadata
✅ 'advantage' column found in parquet file: episode_000000.parquet

STEP 2: Verify Cache Consistency
================================================================================
✅ episode_000000.parquet: Cache matches parquet
✅ episode_000001.parquet: Cache matches parquet
...

STEP 3: Verify Advantage Distribution
================================================================================
Overall Statistics:
  Total frames: 12000
  Advantage=True:  3600 (30.0%)
  Advantage=False: 8400 (70.0%)
✅ Distribution looks reasonable (15-50% True)

STEP 4: Verify Transform Application
================================================================================
Loaded dataset with 11950 chunks
Sampling 10 chunks:
  Chunk   123: ✅ Start frame advantage=True, Task='Fold towel'
  Chunk   456: ❌ Start frame advantage=False, Task='Fold towel'
  ...

VERIFICATION SUMMARY
================================================================================
  ✅ PASSED: Stored Advantages
  ✅ PASSED: Cache Consistency
  ✅ PASSED: Distribution
  ✅ PASSED: Transform
  
✅ All checks passed! Action chunk advantages are properly configured.
```

### Check Debug Output During Training

Watch for these messages:

1. **At training start:**
```
INFO: ACTION CHUNK ADVANTAGE MODE DETECTED
Dataset has 'advantage' feature - applying ActionChunkAdvantagePrompt transform
```

2. **Every 100 chunks:**
```
[ActionChunkAdvantage] Processed 100 chunks: True=32 (32.0%), False=68 (68.0%)
```

If you see these, advantages are being applied correctly!

## Troubleshooting

### "No parquet files found"

You need to have collected data first. The parquet files are in:
```
examples/h1_control_client/h1_data_lerobot/fold_towel/epoch_0/data/chunk-000/
```

### "CUDA out of memory"

Reduce GPU memory usage:
```bash
# Option 1: Use CPU (slower)
python compute_action_chunk_advantages.py --device cpu ...

# Option 2: Use different GPU
export CUDA_VISIBLE_DEVICES=1
```

### "Checkpoint not found"

Make sure you have the Qwen reward model checkpoint:
```bash
ls -la third_party/emboided_reward/checkpoint-750/
# Should contain: adapter_config.json, adapter_model.safetensors, etc.
```

## Expected Performance

### Computation Time (20 episodes, 600 frames each)
- Reward prediction: ~15 minutes
- Embedding extraction: ~3 minutes
- Advantage computation: ~8 minutes
- **Total: ~26 minutes**

### Cache Hit (re-running)
- **Total: ~1 minute** (just loads from cache)

### Training Impact
- Same training time as before
- Better learning signal (more nuanced labels)
- Potentially faster convergence

## Next Steps

1. **Collect more data:** More episodes = better advantage estimates
2. **Tune parameters:** Experiment with threshold and window size
3. **Visualize results:** Check if advantages make sense
4. **Compare training:** Train with/without action chunk advantages

For more details, see `docs/ACTION_CHUNK_ADVANTAGE.md`

