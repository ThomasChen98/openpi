# Action Chunk Advantage Ablation Study Guide

This guide explains how to run the ablation study comparing **Action Chunk Advantage** vs **SFT Baseline**.

---

## 🎯 Goal

Validate that Action Chunk Advantage (learning from both good and bad trajectories with per-frame labels) outperforms standard SFT (filtering to keep only good trajectories).

---

## 📊 Two Approaches

### Approach 1: Action Chunk Advantage (Your Main Method)

**Script**: `scripts/integrated_training_h1.sh`  
**Config**: `training_config_fold_towel_jan12.yaml`

```yaml
labeling_mode: action_chunk_advantage
```

**Behavior:**
- ✅ Uses ALL 50 episodes (100% of data)
- ✅ Computes per-frame advantage labels using Qwen reward model
- ✅ Each 50-frame chunk labeled as `Advantage=True` or `Advantage=False`
- ✅ Model sees prompts: `"Fold the towel..., Advantage=True/False"`
- ✅ ~30% chunks labeled True, ~70% labeled False
- ✅ ~33,800 training chunks

**Training Data:**
```
50 episodes → 35,000 frames → 33,800 chunks
├─ 10,140 chunks: Advantage=True (good trajectories)
└─ 23,660 chunks: Advantage=False (suboptimal)
```

---

### Approach 2: SFT Baseline (Ablation Study)

**Script**: `scripts/integrated_training_h1_SFT.sh`  
**Config**: `training_config_fold_towel_jan12_SFT.yaml`

```yaml
labeling_mode: human_labeling
```

**Behavior:**
- ⚠️ Filters to keep ONLY good episodes
- ⚠️ Uses episode-level labels from HDF5 metadata (human labels: 'g' or 'b')
- ⚠️ Discards any episode labeled as "bad"
- ⚠️ Model sees prompts: `"Fold the towel..."`  (no advantage suffix)
- ⚠️ Expected: ~9 episodes kept (18% of data)
- ⚠️ ~6,000 training chunks

**Training Data:**
```
50 episodes → Filter → 9 good episodes
├─ 9 episodes kept (labeled 'g' by human)
└─ 41 episodes discarded (labeled 'b' by human)
```

---

## 🚀 How to Run

### Step 1: Prepare Data

Collect 50 episodes for both methods. During data collection, label each episode as:
- **'g' (good)**: Episode successfully completes the task
- **'b' (bad)**: Episode fails or is suboptimal

The human labels are stored in HDF5 metadata automatically by the robot client.

---

### Step 2: Train SFT Baseline

```bash
cd /home/yuxin/Projects/openpi

# Start SFT training
./scripts/integrated_training_h1_SFT.sh \
  --config examples/h1_control_client/training_config_fold_towel_jan12_SFT.yaml
```

**Expected Output:**
```
Converting Data (Epoch 1)
  Labeling mode: human_labeling (SFT mode)
  Filtering: Keep only episodes marked as 'good'
  
Advantage label statistics:
  Good episodes (Advantage=True): 9
  Bad episodes (Advantage=False): 41

Filtering episodes (filter_good_only=True):
  Original episodes: 50
  Kept (Advantage=True): 9
  Removed (Advantage=False): 41

Dataset created successfully!
Total episodes: 9
Total frames: ~6,300
```

**Checkpoints saved to:**
```
checkpoints/pi05_h1_auto/fold_towel_jan12_SFT/epoch_1/1199/
```

---

### Step 3: Train with Action Chunk Advantage

```bash
cd /home/yuxin/Projects/openpi

# Start action chunk advantage training
./scripts/integrated_training_h1.sh \
  --config examples/h1_control_client/training_config_fold_towel_jan12.yaml
```

**Expected Output:**
```
Converting Data (Epoch 1)
  Labeling mode: action_chunk_advantage
  Phase 1: Creating parquet files...
  Phase 2: Computing action chunk advantages...
    
SUMMARY STATISTICS
Total episodes: 50
Total frames: 35,100
Frames with Advantage=True: 10,983 (31.3%)
Frames with Advantage=False: 24,117 (68.7%)

  Phase 3: Re-converting with advantages...

Dataset created successfully!
Total episodes: 50
Total frames: 35,100

[During Training]
[ActionChunkAdvantage] Processed 1000 chunks: True=297 (29.7%), False=703 (70.3%)
```

**Checkpoints saved to:**
```
checkpoints/pi05_h1_auto/fold_towel_jan12/epoch_1/1199/
```

---

## 📈 Comparison Metrics

### Data Usage

| Metric | Action Chunk Advantage | SFT Baseline | Ratio |
|--------|----------------------|--------------|-------|
| Episodes used | 50 | 9 | 5.6x |
| Total frames | 35,100 | 6,300 | 5.6x |
| Training chunks | 33,800 | 6,000 | 5.6x |
| Data kept | 100% | 18% | - |

### Training Distribution

| Metric | Action Chunk Advantage | SFT Baseline |
|--------|----------------------|--------------|
| Good examples | 30% (~10,000 chunks) | 100% (6,000 chunks) |
| Bad examples | 70% (~24,000 chunks) | 0% (filtered out) |
| Diversity | High | Low |
| Contrastive signal | Yes | No |

---

## 🎯 Expected Results

### Hypothesis: Action Chunk Advantage Outperforms SFT

**Why?**

1. **More data**: 5.6x more training examples
2. **Contrastive learning**: Model learns what NOT to do from bad examples
3. **Better generalization**: Understands success vs failure patterns
4. **Robustness**: Can recover from suboptimal states

### Evaluation Metrics

After training both models, evaluate on held-out test episodes:

**1. Success Rate**
```bash
# Test both policies on same 10 test episodes
# Count how many successfully complete the task
```

**2. Qwen Reward Score**
```bash
# Run Qwen reward model on test trajectories
# Higher reward = better quality execution
```

**3. Efficiency**
```bash
# Measure average episode length for successful runs
# Fewer steps = more efficient
```

**4. Robustness**
```bash
# Apply perturbations (move towel slightly)
# Check recovery ability
```

---

## 📁 Files

### Scripts
- `scripts/integrated_training_h1.sh` - Action chunk advantage training
- `scripts/integrated_training_h1_SFT.sh` - SFT baseline training
- `scripts/convert_h1_data.sh` - Data conversion (shared)
- `examples/h1_control_client/compute_action_chunk_advantages.py` - Advantage computation

### Configs
- `training_config_fold_towel_jan12.yaml` - Action chunk advantage config
- `training_config_fold_towel_jan12_SFT.yaml` - SFT baseline config

### Code
- `src/openpi/transforms.py` - `ActionChunkAdvantagePrompt` transform
- `src/openpi/training/data_loader.py` - Dataset loading with advantage support
- `examples/h1_control_client/convert_h1_data_to_lerobot.py` - Data conversion with advantages

---

## 💡 Key Differences Summary

### Data Processing

**Action Chunk Advantage:**
```
HDF5 files (50 episodes)
  ↓
Phase 1: Convert to parquet
  ↓
Phase 2: Compute per-frame advantages (Qwen + DINOv3)
  ↓
Phase 3: Re-convert with advantage labels
  ↓
Dataset: All 50 episodes, each chunk labeled
  ↓
Training: 33,800 chunks (30% True, 70% False)
```

**SFT Baseline:**
```
HDF5 files (50 episodes with human labels)
  ↓
Filter: Keep only episodes labeled 'g'
  ↓
Dataset: 9 good episodes only
  ↓
Training: 6,000 chunks (100% good)
```

### Training Prompts

**Action Chunk Advantage:**
- `"Fold the towel in half with both hands, Advantage=True"` (30%)
- `"Fold the towel in half with both hands, Advantage=False"` (70%)

**SFT Baseline:**
- `"Fold the towel in half with both hands"` (100%)

### Learning Signal

**Action Chunk Advantage:**
- Contrastive: Model learns to distinguish good vs bad
- Rich signal: "Do this (True), not that (False)"

**SFT Baseline:**
- Imitative: Model learns to copy good examples
- Simple signal: "Do this"

---

## 🔬 Running the Full Ablation Study

```bash
# 1. Collect 50 episodes (label each as 'g' or 'b' during collection)
./scripts/integrated_training_h1.sh --config training_config_fold_towel_jan12.yaml
# Press 'd' when done collecting

# 2. Train SFT baseline
./scripts/integrated_training_h1_SFT.sh --config training_config_fold_towel_jan12_SFT.yaml

# 3. Wait for both trainings to complete

# 4. Evaluate both checkpoints on test set

# 5. Compare metrics:
#    - Success rate
#    - Qwen reward scores
#    - Episode efficiency
#    - Robustness to perturbations
```

---

## 📝 Notes

- Both methods use the **same** warmup checkpoint (epoch 0)
- Both methods use the **same** data collection
- The **only** difference is how they process and use the data
- This ensures a fair comparison

---

## ✅ Quick Checklist

**For SFT Baseline:**
- [ ] Use `integrated_training_h1_SFT.sh`
- [ ] Config: `training_config_fold_towel_jan12_SFT.yaml`
- [ ] Mode: `human_labeling`
- [ ] Filter: Keep only good episodes
- [ ] Expected: ~9 episodes, ~6,000 chunks

**For Action Chunk Advantage:**
- [ ] Use `integrated_training_h1.sh`
- [ ] Config: `training_config_fold_towel_jan12.yaml`
- [ ] Mode: `action_chunk_advantage`
- [ ] Filter: None (keep all)
- [ ] Expected: 50 episodes, ~33,800 chunks

**For Evaluation:**
- [ ] Test both on same held-out episodes
- [ ] Measure success rate
- [ ] Compute Qwen rewards
- [ ] Check episode efficiency
- [ ] Test robustness

---

Good luck with your ablation study! 🚀

