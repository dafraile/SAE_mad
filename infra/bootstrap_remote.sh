#!/bin/bash
# Bootstrap script: sets up a new vast.ai instance for SAE exploration.
# Run from your local machine:
#   bash bootstrap_remote.sh <port> <ip>
# Example:
#   bash bootstrap_remote.sh 25791 140.150.159.2

set -e

PORT="${1:?Usage: bootstrap_remote.sh <port> <ip>}"
IP="${2:?Usage: bootstrap_remote.sh <port> <ip>}"
SSH="ssh -o IdentityFile=~/.ssh/id_rsa -p $PORT root@$IP"
SCP="scp -o IdentityFile=~/.ssh/id_rsa -P $PORT"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Bootstrapping remote instance at $IP:$PORT ==="

# Step 1: Kill any existing GPU processes and create project dir
echo "--- Step 1: Setup ---"
$SSH bash -c '"pkill -9 -f vllm 2>/dev/null || true; mkdir -p /workspace/sae_mad/results"'

# Step 2: Upload all project files
echo "--- Step 2: Uploading files ---"
$SCP "$LOCAL_DIR"/*.py "$LOCAL_DIR"/*.json "$LOCAL_DIR"/*.sh root@$IP:/workspace/sae_mad/

# Step 3: Upload cached activations if they exist (saves re-running inference)
if [ -f "$LOCAL_DIR/remote_cache/cached_activations.pt" ]; then
    echo "--- Step 3: Uploading cached activations (502MB, may take a minute) ---"
    $SCP "$LOCAL_DIR/remote_cache/cached_activations.pt" root@$IP:/workspace/sae_mad/
else
    echo "--- Step 3: No cached activations to upload ---"
fi

# Step 4: Set up HuggingFace token
echo "--- Step 4: Setting up HF token ---"
HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
if [ -n "$HF_TOKEN" ]; then
    $SSH bash -c "\"mkdir -p ~/.cache/huggingface && echo -n '$HF_TOKEN' > ~/.cache/huggingface/token\""
    echo "HF token set."
else
    echo "WARNING: No local HF token found. You'll need to set it manually."
fi

# Step 5: Install SAELens (PyTorch + transformers should already be on vast.ai images)
echo "--- Step 5: Installing SAELens ---"
$SSH bash -c '"pip3 install sae-lens matplotlib scikit-learn 2>&1 | tail -3"'

# Step 6: Verify
echo "--- Step 6: Verification ---"
$SSH bash -c '"python3 -c \"import torch, transformers, sae_lens; print(f\\\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Transformers: {transformers.__version__}, SAELens: {sae_lens.__version__}\\\"); print(f\\\"GPU: {torch.cuda.get_device_name(0)}\\\")\"" '

echo ""
echo "=== Bootstrap complete! ==="
echo "SSH in with: ssh -p $PORT root@$IP"
echo "Then: cd /workspace/sae_mad"
echo ""
echo "To re-run analysis only (using cached activations):"
echo "  python3 v1_exploration.py --analyze"
echo ""
echo "To re-run everything:"
echo "  python3 v1_exploration.py"
