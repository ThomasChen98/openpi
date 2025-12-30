#!/bin/bash
# Setup script for Qwen-based reward labeling
# This installs the necessary dependencies for using the Qwen reward model
# on systems with CUDA 12.8+ (e.g., Blackwell GPUs)

set -e

echo "=========================================="
echo "Setting up Qwen Reward Model Dependencies"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

echo "Step 1/3: Installing PyTorch with CUDA 12.8 support..."
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 "numpy<2.0"

echo ""
echo "Step 2/3: Installing Unsloth from source..."
uv pip install git+https://github.com/unslothai/unsloth.git

echo ""
echo "Step 3/3: Installing Transformers from source (for Qwen3-VL support)..."
uv pip install git+https://github.com/huggingface/transformers.git

echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

# Verify versions
python -c "
import torch
import transformers
import unsloth

print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ Transformers: {transformers.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA version: {torch.version.cuda}')
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
print(f'✓ Unsloth installed')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "You can now run the Qwen reward model:"
echo "  python examples/h1_control_client/test_qwen_reward.py \\"
echo "    --video_path third_party/videos/fold_towel_epoch_2_episode_000000_bad.mp4 \\"
echo "    --task_instruction \"Fold the towel into a small square\" \\"
echo "    --checkpoint_path third_party/emboided_reward/checkpoint-1130"
echo ""
echo "IMPORTANT: Use 'python' directly instead of 'uv run' to ensure"
echo "           you're using these manually installed dependencies."
echo ""

