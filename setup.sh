#!/bin/bash
# Phase 1: Environment setup for SAE exploration on remote instance
# Run this on the remote: bash setup.sh

set -e

echo "=== Phase 1: SAE Exploration Environment Setup ==="

# Step 1: Kill VLLM to free GPU memory
echo ""
echo "--- Step 1: Freeing GPU memory ---"
if nvidia-smi | grep -q "VLLM"; then
    echo "Found VLLM process, killing it..."
    pkill -f "vllm" || true
    sleep 3
    # Force kill if still running
    if nvidia-smi | grep -q "VLLM"; then
        echo "Force killing..."
        pkill -9 -f "vllm" || true
        sleep 2
    fi
else
    echo "No VLLM process found."
fi
echo "GPU status after cleanup:"
nvidia-smi

# Step 2: Create project directory
echo ""
echo "--- Step 2: Creating project directory ---"
mkdir -p /workspace/sae_mad/results
cd /workspace/sae_mad
echo "Working directory: $(pwd)"

# Step 3: Create virtual environment
echo ""
echo "--- Step 3: Setting up Python environment ---"
python3 -m venv .venv
source .venv/bin/activate
echo "Python: $(python3 --version)"
echo "Pip: $(pip --version)"

# Step 4: Install dependencies
echo ""
echo "--- Step 4: Installing dependencies ---"

# PyTorch -- try CUDA 13.x index first, fall back to default
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu131 2>/dev/null \
    || pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 2>/dev/null \
    || pip install torch torchvision torchaudio
echo "PyTorch installed."

# HuggingFace transformers (need 4.50+ for Gemma 3)
echo "Installing transformers..."
pip install "transformers>=4.50.0" accelerate
echo "Transformers installed."

# SAELens (need 6.30.1+ for Gemma Scope 2 hook name fixes)
echo "Installing SAELens..."
pip install sae-lens
echo "SAELens installed."

# Analysis dependencies
echo "Installing analysis deps..."
pip install numpy pandas matplotlib plotly scikit-learn ipython
echo "Analysis deps installed."

# Step 5: Verify installation
echo ""
echo "--- Step 5: Verification ---"
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    print(f'VRAM free: {torch.cuda.mem_get_info(0)[0] / 1e9:.1f} GB')

import transformers
print(f'Transformers: {transformers.__version__}')

import sae_lens
print(f'SAELens: {sae_lens.__version__}')

# Check if TransformerLens is available and what version
try:
    import transformer_lens
    print(f'TransformerLens: {transformer_lens.__version__}')
except ImportError:
    print('TransformerLens: NOT INSTALLED (SAELens may have pulled it in or not)')

print()
print('=== All checks passed! Ready for hello worlds. ===')
"

echo ""
echo "=== Setup complete! ==="
echo "To activate the environment in future sessions:"
echo "  cd /workspace/sae_mad && source .venv/bin/activate"
echo ""
echo "Next: run the hello world scripts:"
echo "  python3 exploratory/hw1_load_model.py"
