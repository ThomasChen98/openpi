#!/bin/bash
# Setup script for Qwen-based reward labeling
# This installs the necessary dependencies for using the Qwen reward model
# on systems with CUDA 12.8+ (e.g., Blackwell GPUs)

set -e

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

# Function to check if setup is needed (using venv Python)
check_setup_needed() {
    # Check PyTorch version (must be cu128 for Blackwell)
    .venv/bin/python -c "import torch; exit(0 if '+cu128' in torch.__version__ else 1)" 2>/dev/null || return 1
    
    # Check transformers version (must be 5.0+ for Qwen3-VL)
    .venv/bin/python -c "import transformers; v = transformers.__version__; exit(0 if v.startswith('5.') or v.startswith('6.') else 1)" 2>/dev/null || return 1
    
    return 0
}

# Check if setup is already done
if check_setup_needed; then
    PYTORCH_VER=$(.venv/bin/python -c 'import torch; print(torch.__version__)' 2>/dev/null)
    TRANSFORMERS_VER=$(.venv/bin/python -c 'import transformers; print(transformers.__version__)' 2>/dev/null)
    echo "✓ Qwen reward dependencies already installed in venv:"
    echo "  PyTorch: $PYTORCH_VER"
    echo "  Transformers: $TRANSFORMERS_VER"
    exit 0
fi

echo "=========================================="
echo "Setting up Qwen Reward Model Dependencies"
echo "=========================================="
echo ""

echo "Step 1/3: Installing PyTorch with Blackwell/CUDA 12.8 support..."
# Check if we have the right PyTorch version for Blackwell (sm_120)
NEEDS_PYTORCH_UPDATE=0
.venv/bin/python -c "import torch; exit(0 if '+cu128' in torch.__version__ else 1)" 2>/dev/null || NEEDS_PYTORCH_UPDATE=1

if [ $NEEDS_PYTORCH_UPDATE -eq 1 ]; then
    echo "  Current PyTorch doesn't support Blackwell GPU (sm_120)"
    echo "  Installing PyTorch nightly with CUDA 12.8..."
    uv pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
else
    echo "  PyTorch with CUDA 12.8 already installed"
fi

echo ""
echo "Step 2/3: Installing Unsloth from source..."
uv pip install --upgrade --force-reinstall git+https://github.com/unslothai/unsloth.git || {
    echo "Error: Failed to install Unsloth"
    exit 1
}

echo ""
echo "Step 3/3: Installing Transformers from source (for Qwen3-VL support)..."
echo "  Current version: $(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo 'not installed')"
echo "  Installing latest dev version to project venv..."
# Uninstall old version first to avoid conflicts
uv pip uninstall transformers -y 2>/dev/null || true
# Install new version
uv pip install --reinstall git+https://github.com/huggingface/transformers.git || {
    echo "Error: Failed to install Transformers from source"
    exit 1
}
# Fix numpy version conflict
uv pip install "numpy<2.3.0,>=2.0" || true

echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

# Verify versions (using venv Python)
.venv/bin/python -c "
import torch
import transformers
import unsloth

print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ Transformers: {transformers.__version__}')

# Check if transformers is the dev version
if '5.0.0' in transformers.__version__:
    print('✓ Transformers dev version (with Qwen3-VL support)')
else:
    print('⚠ Warning: Transformers might not have Qwen3-VL support')

print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA version: {torch.version.cuda}')
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
    compute_cap = torch.cuda.get_device_capability(0)
    print(f'✓ Compute capability: {compute_cap[0]}.{compute_cap[1]}')
    if compute_cap[0] >= 12:
        print('✓ Blackwell GPU support enabled')
print(f'✓ Unsloth installed')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
