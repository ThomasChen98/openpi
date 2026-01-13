# Action Chunk Advantage Labeling

## Overview

Action chunk advantage labeling provides **fine-grained, per-frame advantage labels** instead of labeling entire episodes. This allows the model to learn which specific action sequences lead to good outcomes, even within episodes that overall may not be successful.

## Comparison with Episode-Level Labeling

### Episode-Level Labeling (reward_labeling mode)
- **Granularity:** All frames in an episode get the same label
- **Label:** Based on final outcome or average reward
- **Prompt:** `"Fold the towel, Advantage=True"` for all frames in good episodes
- **Limitation:** Cannot distinguish good actions from bad actions within the same episode

### Action Chunk Advantage (action_chunk_advantage mode)
- **Granularity:** Each frame gets its own label
- **Label:** Based on future outcomes from similar states
- **Prompt:** Frame 50 might be `"Fold the towel, Advantage=True"` while frame 51 is `"Fold the towel, Advantage=False"`
- **Benefit:** Learns which specific action sequences are effective

## How It Works

### Algorithm

For each action chunk (starting from frame `t`):

1. **Compute Visual Embedding:** Extract DINOv3 embedding for frame `t`
2. **Find Similar States:** Search across all episodes for visually similar frames
3. **Evaluate Future Rewards:** For each similar state, compute mean reward over next 80 frames
4. **Rank Episodes:** Sort episodes by their future rewards from similar states
5. **Label Advantage:** If test episode is in top 33%, label as `Advantage=True`

### Example

```
Episode A, Frame 100:
  - Visual embedding: [0.23, -0.45, ...]
  - Find similar frames across all episodes:
    * Episode B, Frame 95: future reward = 75.2
    * Episode C, Frame 103: future reward = 45.8
    * Episode D, Frame 98: future reward = 82.1
    * Episode A, Frame 100: future reward = 78.5  ← Test frame
  - Ranking: [D(82.1), A(78.5), B(75.2), C(45.8)]
  - Episode A is in top 33% → Advantage=True
```

## Configuration

### Training Config (training_config.yaml)

```yaml
training:
  labeling_mode: "action_chunk_advantage"

reward:
  checkpoint_path: "/path/to/qwen/checkpoint"
  task_instruction: "Fold the towel into a small square."
  max_frames: 30
  advantage_threshold: 0.33      # Top 33% get Advantage=True
  look_ahead_window: 80          # Frames to look ahead for evaluation
```

### Parameters

- **checkpoint_path:** Path to fine-tuned Qwen3-VL reward model
- **task_instruction:** Detailed task description for reward prediction
- **max_frames:** Number of frames to sample for reward prediction (default: 30)
- **advantage_threshold:** Percentile threshold (0.33 = top 33%)
- **look_ahead_window:** How many future frames to consider (default: 80)

## Usage

### 1. Collect Data

```bash
# Run data collection as normal
./scripts/integrated_training.sh --config training_config_fold_towel.yaml
```

### 2. Compute Action Chunk Advantages

This happens automatically during the convert step, but you can run it manually:

```bash
python examples/h1_control_client/compute_action_chunk_advantages.py \
    --data-dir examples/h1_control_client/h1_data_lerobot/fold_towel/epoch_0/data/chunk-000 \
    --task-instruction "Fold the towel into a small square." \
    --checkpoint-path third_party/emboided_reward/checkpoint-750 \
    --look-ahead-window 80 \
    --advantage-threshold 0.33
```

This creates cache files:
- `episode_000000_reward.pkl` - Dense reward predictions
- `episode_000000_ego_image_embeddings.pkl` - Visual embeddings
- `episode_000000_action_chunk_advantages.pkl` - Per-frame advantage labels

### 3. Convert to LeRobot Format

```bash
./scripts/convert_h1_data.sh \
    --task-name fold_towel \
    --epoch 0 \
    --labeling-mode action_chunk_advantage
```

### 4. Train

Training proceeds as normal. The dataloader automatically loads per-frame advantage labels from the `task` field in the dataset.

## Implementation Details

### Files Modified

1. **compute_action_chunk_advantages.py** (NEW)
   - Pre-computes rewards using Qwen3-VL
   - Pre-computes visual embeddings using DINOv3
   - Computes per-frame advantages using similarity search

2. **convert_h1_data_to_lerobot.py**
   - Added `action_chunk_advantage` mode
   - Loads per-frame advantages from cache files
   - Assigns different prompts to each frame based on its advantage

3. **convert_h1_data.sh**
   - Calls `compute_action_chunk_advantages.py` before conversion
   - Passes through advantage parameters

4. **integrated_training.sh**
   - Added support for `action_chunk_advantage` mode
   - Exports required environment variables

### Data Flow

```
Raw HDF5 Episodes
    ↓
Convert to Parquet (LeRobot format)
    ↓
Compute Rewards (Qwen3-VL) → episode_*_reward.pkl
    ↓
Compute Embeddings (DINOv3) → episode_*_ego_image_embeddings.pkl
    ↓
Compute Advantages (Similarity Search) → episode_*_action_chunk_advantages.pkl
    ↓
Convert HDF5 to LeRobot with Per-Frame Advantages
    ↓
Training (each frame has its own advantage label)
```

## Performance Considerations

### Computation Time

For a dataset with 20 episodes × 600 frames each:
- **Reward Prediction:** ~10-15 minutes (with batch_size=1)
- **Embedding Extraction:** ~2-3 minutes (with batch_size=64)
- **Advantage Computation:** ~5-10 minutes (similarity search)
- **Total:** ~20-30 minutes per epoch

### Memory Usage

- **GPU Memory:** ~30GB for Qwen3-VL + DINOv3 (uses GPU 1 by default)
- **Disk Space:** ~50MB per episode for cache files

### Optimization Tips

1. **Caching:** Results are cached, so re-running is fast
2. **Batch Size:** Increase if you have more GPU memory
3. **Parallel Processing:** Run on multiple GPUs by setting `CUDA_VISIBLE_DEVICES`

## Troubleshooting

### Error: "No action chunk advantages found"

**Cause:** Cache files not generated

**Solution:**
```bash
# Manually run advantage computation
python examples/h1_control_client/compute_action_chunk_advantages.py \
    --data-dir path/to/parquet/files \
    --task-instruction "Your task" \
    --checkpoint-path path/to/checkpoint
```

### Error: "CUDA out of memory"

**Cause:** Qwen3-VL model is large

**Solution:**
- Use `--device cpu` (slower but works)
- Reduce batch size in the script
- Use a GPU with more memory

### Error: "Index out of bounds" during reward prediction

**Cause:** Batching issues with variable-length sequences

**Solution:** Already fixed by using `batch_size=1` in `run_inference()`

## Comparison with Original Notebooks

This implementation is based on:
- `pre_compute_reward_observation_embd.ipynb` - Reward and embedding computation
- `Qwen_action_chunck_advantage_inference_test.ipynb` - Advantage labeling logic

### Key Differences

1. **Integrated:** Runs automatically in the training pipeline
2. **Cached:** Results are saved and reused
3. **Robust:** Handles edge cases and errors gracefully
4. **Configurable:** All parameters exposed in training config

## Future Improvements

1. **Adaptive Window:** Automatically determine optimal look-ahead window
2. **Multi-GPU:** Parallelize across multiple GPUs
3. **Incremental Updates:** Only recompute for new episodes
4. **Visualization:** Generate videos showing advantage labels over time
5. **Threshold Tuning:** Automatically tune advantage threshold based on validation performance

## References

- DINOv3: [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
- Qwen3-VL: [unsloth/Qwen3-VL-8B-Instruct](https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct)
- LeRobot: [https://github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)

