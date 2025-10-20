#!/usr/bin/bash
#
# Setup script for H1-2 Remote Policy Client
# Run this once to install all dependencies
#

set -e  # Exit on error

echo "=========================================="
echo "  H1-2 Remote Client Setup"
echo "=========================================="

# Check if in correct directory
if [ ! -f "h1_remote_client.py" ]; then
    echo "❌ Error: Please run this script from the h1_control_client directory"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "📍 Python version: $python_version"

# 1. Install conda dependencies (pinocchio)
echo ""
echo "Step 1: Installing pinocchio via conda..."
if command -v conda &> /dev/null; then
    echo "  Found conda, checking if pinocchio is installed..."
    if python3 -c "import pinocchio" 2>/dev/null; then
        echo "  ✅ pinocchio already installed"
    else
        echo "  Installing pinocchio..."
        conda install -y pinocchio=3.1.0 -c conda-forge
    fi
else
    echo "  ⚠️  conda not found. Please install pinocchio manually:"
    echo "     conda create -n h1_client python=3.10 pinocchio=3.1.0 -c conda-forge"
    echo "  Or install via pip (may have issues):"
    echo "     pip install pin"
fi

# 2. Install Python dependencies
echo ""
echo "Step 2: Installing Python dependencies..."
pip install -r requirements.txt

# 3. Install OpenPi client
echo ""
echo "Step 3: Installing OpenPi client..."
if [ -d "../../packages/openpi-client" ]; then
    cd ../../packages/openpi-client
    pip install -e .
    cd - > /dev/null
    echo "  ✅ OpenPi client installed"
else
    echo "  ❌ Error: Could not find openpi-client package"
    echo "     Make sure you cloned the full openpi repository"
    exit 1
fi

# 4. Install Unitree SDK
echo ""
echo "Step 4: Installing Unitree SDK..."
if [ -d "libraries/unitree_sdk2_python" ]; then
    cd libraries/unitree_sdk2_python
    pip install -e .
    cd - > /dev/null
    echo "  ✅ Unitree SDK installed"
else
    echo "  ❌ Error: Unitree SDK not found in libraries/"
    exit 1
fi

# 5. Install Inspire Hand SDK
echo ""
echo "Step 5: Installing Inspire Hand SDK..."
if [ -d "libraries/inspire_hand_sdk" ]; then
    cd libraries/inspire_hand_sdk
    pip install -e .
    cd - > /dev/null
    echo "  ✅ Inspire Hand SDK installed"
else
    echo "  ❌ Error: Inspire Hand SDK not found in libraries/"
    exit 1
fi

echo ""
echo "=========================================="
echo "  ✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Start the policy server on your GPU machine:"
echo "     cd /path/to/openpi"
echo "     uv run scripts/mock_policy_server.py --port 8000"
echo ""
echo "  2. Run the H1-2 client:"
echo "     python h1_remote_client.py --server-host <gpu-ip> --server-port 8000"
echo ""

