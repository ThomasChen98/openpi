# Embodied Reward Labeling System

This document describes the reward labeling system for automatically labeling H1 robot episodes with advantage labels using a Vision-Language Model (VLM).

## Overview

The reward labeling system uses OpenAI's GPT-5.1 (or similar VLM) to automatically evaluate robot episodes and assign advantage labels (True/False) based on task completion quality. This eliminates the need for manual human labeling during training iterations.

## Components

### 1. `embodied_reward_labeling.py`

A standalone script that can:
- Test reward labeling on raw HDF5 data
- Label all episodes in a directory with advantage scores
- Generate JSON output with labeling results

**Usage:**
```bash
# Test on a directory of HDF5 files
uv run examples/h1_control_client/embodied_reward_labeling.py \
    --data_dir examples/h1_control_client/h1_data_auto/lift_lid_dec7/epoch_5/raw \
    --task_instruction "Lift the lid off the bowl and place it on the table" \
    --advantage_threshold 0.3 \
    --output_json labels.json

# With detailed task instruction
uv run examples/h1_control_client/embodied_reward_labeling.py \
    --data_dir /path/to/raw/hdf5s \
    --task_instruction "$(cat task_instruction.txt)" \
    --max_frames 30 \
    --image_rotation 0 \
    --advantage_threshold 0.4
```

**Key Features:**
- Converts HDF5 episodes to video for VLM processing
- Samples frames evenly across the episode
- Uses shuffling and reflection for more robust predictions
- Parallel processing with configurable workers
- Outputs detailed labeling statistics

### 2. `convert_h1_data_to_lerobot.py` (Modified)

Now supports three labeling modes:

1. **`none`**: No advantage labeling (original behavior)
2. **`human_labeling`**: Read advantage labels from HDF5 metadata (manual labels)
3. **`reward_labeling`**: Use VLM reward model to automatically label advantage

**New Parameters:**
- `--labeling_mode`: Choose labeling strategy
- `--reward_task_instruction`: Detailed task description for VLM
- `--reward_max_frames`: Maximum frames to sample (default: 30)
- `--reward_image_rotation`: Image rotation angle (0, 90, 180, 270)
- `--reward_advantage_threshold`: Percentage threshold for Advantage=True (0-100)

**Example:**
```bash
uv run examples/h1_control_client/convert_h1_data_to_lerobot.py \
    --data_dir ./h1_data_auto/lift_lid/epoch_5/raw \
    --task_description "lift the lid" \
    --labeling_mode reward_labeling \
    --reward_task_instruction "Lift the lid off the bowl with both hands..." \
    --reward_max_frames 30 \
    --reward_advantage_threshold 0.3
```

### 3. `training_config.yaml` (Updated)

Added new `reward` section for reward labeling configuration:

```yaml
# =============================================================================
# REWARD LABELING CONFIGURATION - For reward_labeling mode
# =============================================================================
reward:
  # Detailed task instruction for the VLM reward model
  # Should include success criteria and failure conditions
  task_instruction: |
    Lift the lid on top of the bowl with both hands, move it steadily away from the bowl, and place it flat on the table.
    The robot must use its two hands to do carefully put the lid on the table.
    Very important: The task fails if the lid slips, flipped, drops, or is knocked off during lifting or placement.
    The task fails if the lid is not placed flat on the table (e.g., still connected to the bowl).
    The task fails if the hands of the robot are colliding with the bowl or the table.
    If the task fails, all the frames after the failure should be scored 0.
  
  max_frames: 30                                # Maximum frames to sample for VLM
  image_rotation: 0                             # Image rotation (0, 90, 180, 270)
  advantage_threshold: 0.3                      # Percentile threshold: top 30% = Advantage=True
```

### 4. `integrated_training.sh` (Updated)

Now reads reward configuration from YAML and passes it to the conversion script. The reward labeling **only affects data conversion**, not data collection.

**Workflow with reward labeling:**
1. Robot collects data and humans can still provide labels during collection
2. Data is synced to training server
3. **During conversion:** VLM automatically evaluates episodes and assigns advantage labels
4. Training uses the VLM-generated labels (human labels during collection are ignored)

## Configuration

### Setting Up Reward Labeling

1. **Set OpenAI API Key (Required):**
   ```bash
   # Export in your shell session
   export OPENAI_API_KEY='your-openai-api-key-here'
   
   # Or add to your ~/.bashrc or ~/.zshrc for persistence
   echo 'export OPENAI_API_KEY="your-openai-api-key-here"' >> ~/.bashrc
   source ~/.bashrc
   ```
   
   **IMPORTANT**: Never commit your API key to git! The `.env` file is already in `.gitignore`.

2. **Edit `training_config.yaml`:**
   ```yaml
   training:
     labeling_mode: "reward_labeling"  # Change from "human_labeling"
   
   reward:
     task_instruction: |
       [Detailed task description with success/failure criteria]
     max_frames: 30
     image_rotation: 0
     advantage_threshold: 0.3
   ```

3. **Run Training Pipeline:**
   ```bash
   ./scripts/integrated_training.sh
   ```

### Key Parameters

- **`task_instruction`**: Detailed description of the task, including:
  - What the robot should do
  - Success criteria
  - Failure conditions
  - Special considerations (e.g., "If task fails, all frames after should score 0")

- **`max_frames`**: Number of frames to sample from the episode for VLM evaluation
  - More frames = more accurate but slower
  - Typical range: 20-40 frames

- **`image_rotation`**: Rotate images before sending to VLM
  - Use if camera is mounted at an angle
  - Options: 0, 90, 180, 270 (degrees clockwise)

- **`advantage_threshold`**: Percentile threshold for labeling as "good" (Advantage=True)
  - Value between 0.0 and 1.0 representing the top X% of episodes
  - VLM calculates total reward for each episode in the batch
  - Episodes are ranked by total reward
  - Top X% episodes get Advantage=True
  - Example: threshold=0.3 → top 30% episodes are labeled as good
  - Example: 100 episodes with threshold=0.3 → top 30 episodes get Advantage=True

## How It Works

### Reward Inference Pipeline

1. **Frame Sampling**: Extract evenly-spaced frames from episode video
2. **Shuffling**: Shuffle frame order to prevent temporal bias
3. **Initial Prediction**: VLM predicts task completion % for each frame
4. **Reflection**: VLM reviews and corrects its own predictions
5. **Scoring**: Calculate total score = sum of corrected predictions for each episode
6. **Ranking**: Sort all episodes in the batch by total score
7. **Percentile Labeling**: Top X% episodes (based on advantage_threshold) get Advantage=True

### VLM Prompting

The system uses a carefully designed prompt that:
- Shows the VLM the initial state (frame 1)
- Presents subsequent frames in random order
- Asks for task completion percentage (0-100) for each frame
- Includes task description with success/failure criteria
- Uses reflection to improve prediction quality

## Benefits

1. **Automated Labeling**: No need for manual review of hundreds of episodes
2. **Consistency**: VLM applies consistent criteria across all episodes
3. **Scalability**: Can label large datasets in parallel
4. **Transparency**: Detailed scores show why each episode was labeled
5. **Flexibility**: Easy to adjust criteria by changing task instruction

## Limitations

1. **API Cost**: VLM calls can be expensive for large datasets
2. **Speed**: Slower than instant human labels during collection
3. **API Dependency**: Requires OpenAI API access and internet connection
4. **VLM Errors**: VLM may occasionally make mistakes in evaluation

## Troubleshooting

### No API Key Error
```bash
export OPENAI_API_KEY='your-key-here'
```

### Import Error
Make sure `embodied_reward_util.py` is in `third_party/emboided_reward/`:
```bash
ls third_party/emboided_reward/embodied_reward_util.py
```

### Slow Performance
- Reduce `max_workers` if hitting API rate limits
- Reduce `max_frames` for faster processing
- Disable reflection with `--no_reflection` (less accurate)

### Wrong Labels
- Adjust `advantage_threshold` to change what percentage of episodes are labeled as good
  - Higher threshold (e.g., 0.5) → only top 50% are good (more selective)
  - Lower threshold (e.g., 0.2) → top 20% are good (very selective)
  - Default 0.3 → top 30% are good (balanced)
- Improve `task_instruction` with more specific criteria to improve ranking quality
- Review individual episode scores in output JSON to verify ranking makes sense

## Testing

Test reward labeling on a sample directory:
```bash
# Label a few episodes (top 30% will be marked as good)
uv run examples/h1_control_client/embodied_reward_labeling.py \
    --data_dir examples/h1_control_client/h1_data_auto/lift_lid/epoch_5/raw \
    --task_instruction "Lift lid off bowl and place on table" \
    --advantage_threshold 0.3 \
    --output_json test_labels.json

# Review results sorted by reward
cat test_labels.json | jq -r 'to_entries | sort_by(.value.corrected_total_reward) | reverse | .[] | "\(.value.advantage) \(.value.corrected_total_reward) \(.key)"'
```

## Integration with Training Pipeline

The reward labeling is fully integrated into the training pipeline:

1. **Data Collection** (manual or policy-driven)
   - Robot collects episodes
   - Human can optionally label during collection (labels saved to HDF5)
   - Data synced to training server

2. **Data Conversion** (automatic)
   - If `labeling_mode: reward_labeling`, VLM labels all episodes
   - Human labels from collection are ignored
   - Episodes converted to LeRobot format with VLM advantage labels

3. **Training** (automatic)
   - Policy trains on data with VLM-generated advantage labels
   - Prompt format: "{task_description}, Advantage=True/False"

4. **Next Epoch** (repeat)
   - New policy deployed for collection
   - Process repeats with new data

## Summary

The reward labeling system enables fully automated training pipelines by eliminating the need for manual episode labeling. It maintains compatibility with human labeling (you can still collect human labels during rollout for debugging/comparison) while providing a scalable, consistent alternative for production training.
