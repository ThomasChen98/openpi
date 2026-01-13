# Action Chunk Advantage - Technical Deep Dive

## The Problem You Identified

You correctly identified a critical flaw in my initial implementation! Let me explain:

### What I Did Wrong Initially ❌

**Stored per-frame advantages in the `task` field:**
```python
# Frame 0: task="Fold towel, Advantage=True"
# Frame 1: task="Fold towel, Advantage=False"  
# Frame 2: task="Fold towel, Advantage=True"
```

**Why this breaks:**
1. **Dataloader creates chunks:** Starting at frame `t`, takes frames `[t, t+1, ..., t+49]`
2. **Mixed prompts in one chunk:** A chunk starting at frame 0 would see 50 different task strings!
3. **Wrong semantics:** We want the chunk's advantage based on START frame, not individual frames

### The Correct Solution ✅

**Store advantage as a separate feature, apply based on chunk start frame:**

```python
# In LeRobot dataset:
Frame 0: advantage=True, task="Fold towel"
Frame 1: advantage=False, task="Fold towel"
Frame 2: advantage=True, task="Fold towel"
...

# During training (action_horizon=50):
Chunk starting at frame 0:
  → Read advantage[0] = True
  → Prompt for ALL 50 frames: "Fold towel, Advantage=True"

Chunk starting at frame 1:
  → Read advantage[1] = False
  → Prompt for ALL 50 frames: "Fold towel, Advantage=False"
```

## Implementation Architecture

### 1. Data Storage (convert_h1_data_to_lerobot.py)

```python
# Add 'advantage' as a separate feature
features = {
    "ego_cam": {...},
    "qpos": {...},
    "action": {...},
    "advantage": {  # NEW!
        "dtype": "bool",
        "shape": (1,),
        "names": ["advantage"],
    }
}

# Store per-frame advantages
for step_idx in range(len(actions)):
    frame_data = {
        "ego_cam": ...,
        "qpos": ...,
        "action": ...,
        "advantage": np.array([bool(frame_advantages[step_idx])], dtype=bool),  # Per-frame!
        "task": task_description  # Clean, no advantage suffix
    }
    dataset.add_frame(frame_data)
```

**Key points:**
- `advantage` is a **separate feature** (like `qpos`, `action`)
- `task` field stays clean (no advantage suffix)
- Each frame has its own advantage value

### 2. Data Loading (LeRobotDataset)

LeRobot's `delta_timestamps` mechanism creates action chunks:

```python
dataset = LeRobotDataset(
    repo_id=data_dir,
    delta_timestamps={
        "action": [0/30, 1/30, 2/30, ..., 49/30]  # 50 frames at 30 FPS
    }
)

# When you access dataset[i]:
# Returns frames [i, i+1, i+2, ..., i+49]
# Including advantage values for all 50 frames
```

**What you get:**
```python
data = dataset[100]  # Chunk starting at frame 100
data["advantage"]  # Shape: (50, 1) - advantages for frames [100, 101, ..., 149]
data["action"]     # Shape: (50, 26) - actions for frames [100, 101, ..., 149]
data["task"]       # String: "Fold towel" (no advantage yet)
```

### 3. Transform (ActionChunkAdvantagePrompt)

This transform reads advantage from **chunk start frame** and augments the prompt:

```python
@dataclasses.dataclass(frozen=True)
class ActionChunkAdvantagePrompt(DataTransformFn):
    def __call__(self, data: DataDict) -> DataDict:
        # Get advantage for START frame of chunk
        advantage_values = data["advantage"]  # Shape: (50, 1)
        start_advantage = bool(advantage_values[0, 0])  # Only use first frame!
        
        # Augment prompt
        base_prompt = data["prompt"]  # "Fold towel"
        advantage_str = "True" if start_advantage else "False"
        augmented_prompt = f"{base_prompt}, Advantage={advantage_str}"
        
        return {**data, "prompt": augmented_prompt}
```

**Key insight:** Only `advantage[0]` (start frame) determines the prompt for the entire chunk!

### 4. Automatic Application (data_loader.py)

The transform is automatically applied when the dataset has an `advantage` feature:

```python
def create_torch_dataset(data_config, action_horizon, model_config):
    dataset = LeRobotDataset(...)
    
    transforms = []
    if data_config.prompt_from_task:
        transforms.append(PromptFromLeRobotTask(dataset_meta.tasks))
    
    # Auto-detect advantage feature
    if 'advantage' in dataset_meta.features:
        transforms.append(ActionChunkAdvantagePrompt())  # Automatically added!
    
    return TransformedDataset(dataset, transforms)
```

## Data Flow Example

### Input: Episode with 600 frames

```
Frame 0:   advantage=False
Frame 1:   advantage=False
...
Frame 50:  advantage=True   ← Good actions start here
Frame 51:  advantage=True
...
Frame 150: advantage=False  ← Mistake happened
...
Frame 599: advantage=False
```

### Training: 20,000 Chunks (action_horizon=50)

```
Chunk 0 (frames 0-49):
  - Start frame: 0
  - advantage[0] = False
  - Prompt: "Fold towel, Advantage=False"
  - Actions: 50 action vectors

Chunk 1 (frames 1-50):
  - Start frame: 1
  - advantage[1] = False
  - Prompt: "Fold towel, Advantage=False"
  - Actions: 50 action vectors

...

Chunk 50 (frames 50-99):
  - Start frame: 50
  - advantage[50] = True  ← Changed!
  - Prompt: "Fold towel, Advantage=True"
  - Actions: 50 action vectors

...

Chunk 150 (frames 150-199):
  - Start frame: 150
  - advantage[150] = False  ← Changed again!
  - Prompt: "Fold towel, Advantage=False"
  - Actions: 50 action vectors
```

**Result:** Each of the 20,000 chunks gets its own advantage label based on its start frame!

## Why This Works

### Semantic Correctness

**Question:** "Are the actions starting from frame `t` good?"

**Answer:** Look at `advantage[t]` (computed via similarity search + future rewards)

**Application:** When training on chunk starting at `t`, use `advantage[t]` for the prompt

### Computational Efficiency

- **Storage:** Only store per-frame advantages once (in LeRobot dataset)
- **Lookup:** O(1) to read `advantage[start_frame]`
- **No recomputation:** Advantages pre-computed and cached

### Training Signal

The model learns:
- **Input:** State at frame `t`, prompt "Task, Advantage=True/False"
- **Output:** Next 50 actions
- **Signal:** Advantage tells model if these actions lead to good outcomes

## Comparison: Episode-Level vs Action Chunk

### Episode-Level (reward_labeling)

```python
# All frames get same label
Episode 1 (600 frames): Advantage=True
  → All 600 chunks: "Fold towel, Advantage=True"

Episode 2 (600 frames): Advantage=False
  → All 600 chunks: "Fold towel, Advantage=False"
```

**Total unique prompts:** 2 (True/False)

### Action Chunk (action_chunk_advantage)

```python
# Each frame gets own label
Episode 1 (600 frames):
  → Chunk 0-49: "Fold towel, Advantage=False"
  → Chunk 50-149: "Fold towel, Advantage=True"
  → Chunk 150-599: "Fold towel, Advantage=False"

Episode 2 (600 frames):
  → Chunk 0-99: "Fold towel, Advantage=True"
  → Chunk 100-599: "Fold towel, Advantage=False"
```

**Total unique prompts:** Still 2, but applied at chunk level!

## Implementation Files

### Modified Files

1. **`examples/h1_control_client/convert_h1_data_to_lerobot.py`**
   - Added `advantage` feature to dataset schema
   - Store per-frame advantages as separate feature
   - Keep `task` field clean

2. **`src/openpi/transforms.py`**
   - Added `ActionChunkAdvantagePrompt` transform
   - Reads advantage from chunk start frame
   - Augments prompt with advantage

3. **`src/openpi/training/data_loader.py`**
   - Auto-detects `advantage` feature
   - Automatically applies `ActionChunkAdvantagePrompt`

### No Changes Needed

- **Training loop:** Works unchanged
- **Model:** Sees prompts as before
- **Inference:** Standard prompt format

## Testing

### Verify Data Storage

```python
import pandas as pd

# Load episode
df = pd.read_parquet('episode_000000.parquet')

# Check advantage column
print(df.columns)  # Should include 'advantage'
print(df['advantage'].dtype)  # Should be bool or int
print(df['advantage'].value_counts())  # Distribution of True/False
```

### Verify Transform

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Load dataset
dataset = LeRobotDataset('path/to/dataset')

# Get a chunk
data = dataset[100]  # Chunk starting at frame 100

# Check advantage
print(data['advantage'].shape)  # Should be (50, 1) for action_horizon=50
print(data['advantage'][0])     # Advantage of start frame
print(data['prompt'])           # Should include ", Advantage=True/False"
```

### Verify Training

```python
# During training, check batch
for batch in dataloader:
    obs, actions = batch
    print(obs['prompt'])  # Should see different advantages
    # Example output:
    # ["Fold towel, Advantage=True", "Fold towel, Advantage=False", ...]
```

## Performance

### Storage Overhead

- **Per frame:** 1 bool = 1 byte
- **Per episode (600 frames):** 600 bytes = 0.6 KB
- **Negligible!**

### Computation Overhead

- **Transform:** O(1) per chunk (just read one value)
- **No impact on training speed**

### Memory Overhead

- **Advantage tensor:** (batch_size, action_horizon, 1)
- **Example:** (32, 50, 1) = 1,600 bools = 1.6 KB
- **Negligible compared to images!**

## Summary

✅ **Correct:** Advantages stored per-frame, applied per-chunk
✅ **Efficient:** O(1) lookup, minimal overhead
✅ **Automatic:** Transform auto-applied when feature detected
✅ **Flexible:** Works with existing training pipeline

The implementation correctly addresses your concern about chunk-level advantage labeling!

