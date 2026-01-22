# RoboDopamine Integration

## Overview

RoboDopamine has been successfully integrated as the third reward method option in the training system, alongside "Ours" (Qwen) and "GVL" (OpenAI GPT-5.2).

## What is RoboDopamine?

RoboDopamine is a **goal-conditioned reward model** (GRM-3B) that evaluates robot actions based on progress toward a goal image. Unlike the other methods that use task descriptions, RoboDopamine uses both a task description and a visual goal to provide reward signals.

Key features:
- Goal-conditioned: Requires a goal image showing the desired end state
- Uses vLLM for fast inference
- Returns progress scores (0-100) for each frame
- Model: `tanhuajie2001/Robo-Dopamine-GRM-3B` from HuggingFace

## Configuration

### Basic Setup

In your `training_config.yaml`:

```yaml
reward:
  method: "RoboDopamine"
  goal_image_path: "RoboDopamine/goal_images/insert_bottle_goal_image.png"
  task_instruction: "Pick up the bottle and insert it into the tray."
  max_frames: 30
  advantage_threshold: 0.3
  look_ahead_window: 80
  random_drop_rate: 0.0

training:
  labeling_mode: "action_chunk_advantage"  # Required for RoboDopamine
```

### Requirements

1. **Goal Image**: A PNG/JPEG image showing the successful completion state
   - Create task-specific goal images in `RoboDopamine/goal_images/`
   - Examples: `fold_towel_goal_image.png`, `insert_bottle_goal_image.png`

2. **RoboDopamine Package**: Must be installed and importable
   ```bash
   # Ensure RoboDopamine is in your Python path
   pip install -e RoboDopamine/  # or appropriate installation method
   ```

3. **Labeling Mode**: Currently only supports `action_chunk_advantage`
   - Episode-level `reward_labeling` mode not yet implemented

## How It Works

### 1. Episode Processing

For each episode, RoboDopamine:

1. **Converts to MP4**: Saves episode frames as `episode_XXXXXX_ego_images.mp4`
2. **Runs Pipeline**: Processes video with goal image using `run_pipeline()`
3. **Extracts Progress**: Gets progress scores (0-100) for each sampled frame
4. **Interpolates**: Creates dense rewards for all frames via interpolation

### 2. Advantage Computation

Like other methods, advantages are computed by:

1. Finding visually similar states across episodes
2. Comparing future rewards from those states
3. Labeling top-performing action chunks with `Advantage=True`

### 3. Training

The policy trains on prompts like:
- `"Pick up the bottle and insert it into the tray., Advantage=True"`
- `"Pick up the bottle and insert it into the tray., Advantage=False"`

## File Outputs

RoboDopamine creates several files during processing:

```
data/chunk-000/
├── episode_000000.parquet
├── episode_000000_ego_images.mp4              # Temporary video
├── episode_000000_RoboDopamine_reward.pkl     # Cached rewards
├── episode_000000_ego_image_embeddings.pkl    # DINOv3 embeddings (shared)
└── episode_000000_action_chunk_advantages.pkl # Advantage labels
```

### Cache System

- Rewards are cached per method: `_RoboDopamine_reward.pkl`
- Allows switching between methods without recomputing
- Use `--force-recompute` flag to regenerate caches

## Usage Examples

### Example 1: Bottle Insertion Task

```yaml
# training_config.yaml
task:
  name: "bottle_insertion"
  description: "Pick up the bottle and insert it into the tray"

training:
  labeling_mode: "action_chunk_advantage"

reward:
  method: "RoboDopamine"
  goal_image_path: "RoboDopamine/goal_images/insert_bottle_goal_image.png"
  task_instruction: "Pick up the bottle and insert it into the tray. Keep the bottle perfectly upright (vertical) and aligned straight."
  max_frames: 20
  advantage_threshold: 0.3
  look_ahead_window: 80
```

### Example 2: Towel Folding Task

```yaml
reward:
  method: "RoboDopamine"
  goal_image_path: "RoboDopamine/goal_images/fold_towel_goal_image.png"
  task_instruction: "Fold the towel into a small square."
  max_frames: 30
  advantage_threshold: 0.3
```

## Creating Goal Images

### Best Practices

1. **Clear Success State**: Image should clearly show the completed task
2. **Same Viewpoint**: Use the same camera perspective as robot's ego camera
3. **Good Lighting**: Similar lighting conditions to training environment
4. **High Resolution**: At least 224x224 pixels (higher is better)

### Directory Structure

```
RoboDopamine/
├── goal_images/
│   ├── insert_bottle_goal_image.png
│   ├── fold_towel_goal_image.png
│   ├── lift_lid_goal_image.png
│   └── ...
└── results/          # Pipeline outputs (auto-created)
```

## Comparison with Other Methods

| Feature | Ours (Qwen) | GVL (GPT-5.2) | RoboDopamine |
|---------|-------------|---------------|--------------|
| **Input** | Task description | Task description | Task + Goal image |
| **Model** | Fine-tuned Qwen3VL | OpenAI GPT-5.2 | RoboDopamine GRM-3B |
| **Setup** | Checkpoint path | API key | Goal image |
| **Cost** | Free (local) | API costs | Free (local) |
| **Speed** | Fast | Slower (API) | Fast (vLLM) |
| **Modes** | Both | Both | action_chunk only |
| **Parallelization** | Limited | High | Limited |

### When to Use RoboDopamine

**Use RoboDopamine when:**
- ✅ You have clear visual goal states
- ✅ Task success is visually verifiable
- ✅ You need goal-conditioned rewards
- ✅ You want local, fast inference

**Use Other Methods when:**
- ❌ Task has no clear visual goal state
- ❌ Success depends on non-visual factors
- ❌ You need episode-level labeling
- ❌ You want text-only specification

## Troubleshooting

### Error: "Could not import RoboDopamine"

```bash
# Install RoboDopamine package
cd RoboDopamine
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/RoboDopamine
```

### Error: "Goal image path does not exist"

```bash
# Check the path in your config
ls -la RoboDopamine/goal_images/

# Create goal images directory if needed
mkdir -p RoboDopamine/goal_images/
```

### Error: "RoboDopamine only supported for action_chunk_advantage"

Change your training config:

```yaml
training:
  labeling_mode: "action_chunk_advantage"  # Not "reward_labeling"
```

### Slow Processing

RoboDopamine uses vLLM which is fast, but:
- First episode may be slower (model loading)
- MP4 conversion adds overhead
- Consider reducing `max_frames` for faster processing

### Memory Issues

RoboDopamine GRM-3B requires significant GPU memory:
- Default: `gpu_memory_utilization=0.2` (20% of GPU)
- Adjust in `GRMInference` if needed
- Ensure no other models are loaded simultaneously

## Technical Details

### Pipeline Parameters

When calling `run_pipeline()`:

```python
reward_results = model.run_pipeline(
    cam_high_path=mp4_path,      # Main camera (ego view)
    cam_left_path=mp4_path,      # Left camera (same as ego)
    cam_right_path=mp4_path,     # Right camera (same as ego)
    out_root="RoboDopamine/results/",
    task=task_instruction,
    frame_interval=30,           # Process every 30th frame
    batch_size=30,               # Batch size for inference
    goal_image=goal_image_path,
    eval_mode="incremental",     # Evaluate incrementally
    visualize=False              # No visualization (faster)
)
```

### Reward Extraction

Progress scores are extracted and interpolated:

```python
# Extract progress from results
rewd_pred = []
for result in reward_results:
    id = int(result['id'].split('_')[-1])
    rewd_pred.append([id, result['progress']])

# Sort by frame ID
rewd_pred.sort(key=lambda x: x[0])

# Interpolate to all frames
dense_rewards = np.interp(
    np.arange(len(ego_images)),
    anchor_indices,
    progress_scores
)
```

## Implementation Files

Files modified to support RoboDopamine:

1. **`scripts/integrated_training_h1.sh`**
   - Added `REWARD_GOAL_IMAGE_PATH` configuration
   - Added validation for goal image path
   - Updated command building logic

2. **`scripts/convert_h1_data.sh`**
   - Added `--reward-goal-image-path` argument
   - Updated validation and logging

3. **`examples/h1_control_client/compute_action_chunk_advantages.py`**
   - Added RoboDopamine method to `ActionChunkAdvantageComputer`
   - Implemented MP4 conversion and pipeline execution
   - Added progress score extraction logic

4. **`examples/h1_control_client/convert_h1_data_to_lerobot.py`**
   - Added `reward_goal_image_path` parameter
   - Added validation (RoboDopamine requires action_chunk_advantage mode)

5. **Documentation**
   - Updated `reward_method_config_example.yaml`
   - Updated `QUICK_START_REWARD_METHOD.md`
   - Created this file

## Future Enhancements

Potential improvements:

1. **Episode-Level Labeling**: Implement `reward_labeling` mode support
2. **Multi-Goal Support**: Allow multiple goal images per task
3. **Dynamic Goals**: Generate goal images from successful episodes
4. **Goal Variations**: Support for goal image augmentation
5. **Visualization**: Add option to visualize progress predictions

## References

- RoboDopamine Paper: [Link to paper]
- Model Card: `tanhuajie2001/Robo-Dopamine-GRM-3B`
- Original Implementation: `RoboDopamine/examples/inference.py`
- Integration Notebook: `third_party/emboided_reward/pre_compute_reward_observation_embd.ipynb`

## Support

For issues or questions:
1. Check goal image exists and is readable
2. Verify RoboDopamine is properly installed
3. Check `labeling_mode` is set to `action_chunk_advantage`
4. Review logs in `logs/` directory
5. Check cache files in data directory

---

**Note**: RoboDopamine is particularly effective for manipulation tasks with clear visual goals. For tasks where success is better described linguistically, consider using "Ours" or "GVL" methods instead.
