#!/bin/bash
# Setup script for Qwen-based reward labeling
# This installs the necessary dependencies for using the Qwen reward model
# on systems with CUDA 12.8+ (e.g., Blackwell GPUs)

set -e

# Function to check if setup is needed
check_setup_needed() {
    # Check if transformers is at least version 5.0
    python -c "import transformers; v = transformers.__version__; exit(0 if v.startswith('5.') or v.startswith('6.') else 1)" 2>/dev/null
    return $?
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

# Check if setup is already done
if check_setup_needed; then
    echo "✓ Qwen reward dependencies already installed (transformers $(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null))"
    exit 0
fi

echo "=========================================="
echo "Setting up Qwen Reward Model Dependencies"
echo "=========================================="
echo ""

echo "Step 1/3: Installing PyTorch with CUDA 12.8 support..."
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 "numpy<2.0" --quiet

echo ""
echo "Step 2/3: Installing Unsloth from source..."
uv pip install git+https://github.com/unslothai/unsloth.git --quiet

echo ""
echo "Step 3/3: Installing Transformers from source (for Qwen3-VL support)..."
uv pip install git+https://github.com/huggingface/transformers.git --quiet

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
