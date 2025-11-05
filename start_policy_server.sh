#!/bin/bash

# Memory-safe policy server startup script
# Limits GPU memory allocation to approximately 25GB

# Set checkpoint configuration
POLICY_CONFIG="pi05_h1_finetune"
CHECKPOINT_DIR="checkpoints/pi05_h1_finetune/pi05_h1_H50/1999"

# XLA memory management settings
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35  # ~25GB on a 70GB GPU

# Optional: Reduce memory fragmentation
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Optional: Limit JAX to use bfloat16 where possible to save memory
export JAX_DEFAULT_DTYPE_BITS=32

echo "Starting policy server with memory-safe settings..."
echo "Target GPU memory: ~25GB"
echo "Policy config: $POLICY_CONFIG"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo ""

cd /home/yuxin/Projects/openpi

uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=$POLICY_CONFIG \
  --policy.dir=$CHECKPOINT_DIR

