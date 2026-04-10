#!/bin/bash
# vast_gpu.sh -- Manage vast.ai GPU instances for research experiments
#
# Usage:
#   bash vast_gpu.sh search [profile]     Search for offers matching a GPU profile
#   bash vast_gpu.sh launch [profile]     Launch the cheapest matching instance
#   bash vast_gpu.sh status               Show running instances
#   bash vast_gpu.sh ssh [instance_id]    SSH into an instance
#   bash vast_gpu.sh setup [instance_id]  Bootstrap an instance with project files
#   bash vast_gpu.sh pull [instance_id]   Pull results/data from instance
#   bash vast_gpu.sh destroy [instance_id] Destroy an instance
#   bash vast_gpu.sh list-profiles        List available GPU profiles
#
# GPU Profiles (pick the cheapest that fits your task):
#   tiny    - 10GB+ VRAM (3060, 2080Ti) -- Gemma 3 1B, small SAE experiments
#   medium  - 24GB+ VRAM (3090, 4090)   -- Gemma 3 4B, larger SAEs, training
#   large   - 48GB+ VRAM (A6000, L40S)  -- Multiple models, big SAE widths
#   huge    - 80GB+ VRAM (A100, H100)   -- Frontier experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SSH_KEY="$HOME/.ssh/id_rsa"

# GPU profiles: min VRAM, reliability, and optional GPU name filter
declare -A PROFILE_VRAM=(
    [tiny]=10
    [medium]=22
    [large]=45
    [huge]=75
)

declare -A PROFILE_DESC=(
    [tiny]="10GB+ VRAM (RTX 3060, 2080Ti, etc.) - Good for 1B models"
    [medium]="24GB+ VRAM (RTX 3090, 4090, etc.) - Good for 4B models"
    [large]="48GB+ VRAM (A6000, L40S, etc.) - Good for multi-model work"
    [huge]="80GB+ VRAM (A100, H100) - Frontier experiments"
)

# Default image -- PyTorch with CUDA, good for ML work
DEFAULT_IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DEFAULT_DISK=30

cmd="${1:-help}"
arg="${2:-tiny}"

case "$cmd" in
    list-profiles)
        echo "Available GPU profiles:"
        for p in tiny medium large huge; do
            echo "  $p  -- ${PROFILE_DESC[$p]}"
        done
        ;;

    search)
        profile="$arg"
        vram="${PROFILE_VRAM[$profile]:-10}"
        echo "Searching for '$profile' instances (${vram}GB+ VRAM)..."
        vastai search offers \
            "reliability>0.95 num_gpus=1 gpu_ram>=${vram} cuda_vers>=12.0 direct_port_count>=1 rentable=true rented=false" \
            -o 'dph' --limit 15 --storage "$DEFAULT_DISK"
        ;;

    launch)
        profile="$arg"
        vram="${PROFILE_VRAM[$profile]:-10}"
        echo "Finding cheapest '$profile' instance (${vram}GB+ VRAM)..."

        # Get the cheapest offer ID
        OFFER_ID=$(vastai search offers \
            "reliability>0.95 num_gpus=1 gpu_ram>=${vram} cuda_vers>=12.0 direct_port_count>=1 rentable=true rented=false" \
            -o 'dph' --limit 1 --storage "$DEFAULT_DISK" --raw \
            | python3 -c "import sys,json; offers=json.load(sys.stdin); print(offers[0]['id']) if offers else sys.exit(1)")

        if [ -z "$OFFER_ID" ]; then
            echo "No matching offers found."
            exit 1
        fi

        echo "Launching offer $OFFER_ID..."
        vastai create instance "$OFFER_ID" \
            --image "$DEFAULT_IMAGE" \
            --disk "$DEFAULT_DISK" \
            --ssh --direct \
            --onstart-cmd "mkdir -p /workspace/sae_mad"

        echo ""
        echo "Instance creating. Run 'bash vast_gpu.sh status' to check when it's ready."
        echo "Once running, run 'bash vast_gpu.sh setup <instance_id>' to bootstrap."
        ;;

    status)
        echo "Current instances:"
        vastai show instances
        ;;

    ssh)
        instance_id="$arg"
        if [ "$instance_id" = "tiny" ] || [ -z "$instance_id" ]; then
            echo "Usage: bash vast_gpu.sh ssh <instance_id>"
            echo "Run 'bash vast_gpu.sh status' to find your instance ID."
            exit 1
        fi

        # Get SSH connection info
        SSH_INFO=$(vastai show instance "$instance_id" --raw | python3 -c "
import sys, json
info = json.load(sys.stdin)
host = info.get('ssh_host', '')
port = info.get('ssh_port', '')
if host and port:
    print(f'{host} {port}')
else:
    print('NOT_READY')
")
        HOST=$(echo "$SSH_INFO" | cut -d' ' -f1)
        PORT=$(echo "$SSH_INFO" | cut -d' ' -f2)

        if [ "$HOST" = "NOT_READY" ]; then
            echo "Instance not ready yet. Check 'bash vast_gpu.sh status'."
            exit 1
        fi

        echo "Connecting to $HOST:$PORT..."
        ssh -i "$SSH_KEY" -p "$PORT" root@"$HOST" -L 8080:localhost:8080
        ;;

    setup)
        instance_id="$arg"
        if [ "$instance_id" = "tiny" ] || [ -z "$instance_id" ]; then
            echo "Usage: bash vast_gpu.sh setup <instance_id>"
            exit 1
        fi

        # Get SSH connection info
        SSH_INFO=$(vastai show instance "$instance_id" --raw | python3 -c "
import sys, json
info = json.load(sys.stdin)
host = info.get('ssh_host', '')
port = info.get('ssh_port', '')
if host and port:
    print(f'{host} {port}')
else:
    print('NOT_READY')
")
        HOST=$(echo "$SSH_INFO" | cut -d' ' -f1)
        PORT=$(echo "$SSH_INFO" | cut -d' ' -f2)

        if [ "$HOST" = "NOT_READY" ]; then
            echo "Instance not ready yet."
            exit 1
        fi

        SSH_CMD="ssh -i $SSH_KEY -p $PORT root@$HOST"
        SCP_CMD="scp -i $SSH_KEY -P $PORT"

        echo "=== Bootstrapping instance $instance_id ==="

        # Upload project files
        echo "--- Uploading project files ---"
        $SCP_CMD "$SCRIPT_DIR"/*.py "$SCRIPT_DIR"/*.json root@"$HOST":/workspace/sae_mad/ 2>/dev/null || true

        # Upload cached activations if they exist
        if [ -f "$SCRIPT_DIR/remote_cache/cached_activations.pt" ]; then
            echo "--- Uploading cached activations (may take a minute) ---"
            $SCP_CMD "$SCRIPT_DIR/remote_cache/cached_activations.pt" root@"$HOST":/workspace/sae_mad/
        fi

        # Set up HF token
        HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
        if [ -n "$HF_TOKEN" ]; then
            echo "--- Setting HF token ---"
            $SSH_CMD "mkdir -p ~/.cache/huggingface && echo -n '$HF_TOKEN' > ~/.cache/huggingface/token"
        fi

        # Install SAELens
        echo "--- Installing SAELens ---"
        $SSH_CMD "pip install sae-lens matplotlib scikit-learn 2>&1 | tail -3"

        # Create results dir
        $SSH_CMD "mkdir -p /workspace/sae_mad/results"

        # Verify
        echo "--- Verifying ---"
        $SSH_CMD 'python3 -c "import torch, sae_lens; print(f\"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB, SAELens: {sae_lens.__version__}\")"'

        echo ""
        echo "=== Setup complete! ==="
        echo "SSH in: bash vast_gpu.sh ssh $instance_id"
        echo "Then:   cd /workspace/sae_mad && python3 v1_exploration.py"
        ;;

    pull)
        instance_id="$arg"
        if [ "$instance_id" = "tiny" ] || [ -z "$instance_id" ]; then
            echo "Usage: bash vast_gpu.sh pull <instance_id>"
            exit 1
        fi

        SSH_INFO=$(vastai show instance "$instance_id" --raw | python3 -c "
import sys, json
info = json.load(sys.stdin)
print(f\"{info.get('ssh_host','')} {info.get('ssh_port','')}\")
")
        HOST=$(echo "$SSH_INFO" | cut -d' ' -f1)
        PORT=$(echo "$SSH_INFO" | cut -d' ' -f2)

        SCP_CMD="scp -i $SSH_KEY -P $PORT"

        echo "=== Pulling data from instance $instance_id ==="
        mkdir -p "$SCRIPT_DIR/remote_cache" "$SCRIPT_DIR/results"

        # Pull cached activations
        echo "--- Pulling cached activations ---"
        $SCP_CMD root@"$HOST":/workspace/sae_mad/cached_activations.pt "$SCRIPT_DIR/remote_cache/" 2>/dev/null && echo "OK" || echo "Not found"

        # Pull results
        echo "--- Pulling results ---"
        $SCP_CMD root@"$HOST":/workspace/sae_mad/results/* "$SCRIPT_DIR/results/" 2>/dev/null && echo "OK" || echo "Not found"

        echo "=== Pull complete ==="
        ;;

    destroy)
        instance_id="$arg"
        if [ "$instance_id" = "tiny" ] || [ -z "$instance_id" ]; then
            echo "Usage: bash vast_gpu.sh destroy <instance_id>"
            exit 1
        fi
        echo "Destroying instance $instance_id..."
        vastai destroy instance "$instance_id"
        echo "Instance destroyed. No more charges."
        ;;

    help|*)
        echo "vast_gpu.sh -- Manage vast.ai GPU instances for research"
        echo ""
        echo "Commands:"
        echo "  search [profile]      Search for matching GPU offers"
        echo "  launch [profile]      Launch cheapest matching instance"
        echo "  status                Show running instances"
        echo "  ssh <id>              SSH into an instance"
        echo "  setup <id>            Bootstrap instance with project files + deps"
        echo "  pull <id>             Download results/cache from instance"
        echo "  destroy <id>          Destroy instance (stop charges)"
        echo "  list-profiles         Show GPU profiles"
        echo ""
        echo "Profiles: tiny (1B models), medium (4B), large (multi-model), huge (frontier)"
        echo ""
        echo "Typical workflow:"
        echo "  bash vast_gpu.sh search tiny          # Find cheap GPUs"
        echo "  bash vast_gpu.sh launch tiny          # Launch cheapest one"
        echo "  bash vast_gpu.sh status               # Wait for 'running'"
        echo "  bash vast_gpu.sh setup <id>           # Upload files + install deps"
        echo "  bash vast_gpu.sh ssh <id>             # SSH in and work"
        echo "  bash vast_gpu.sh pull <id>            # Save results locally"
        echo "  bash vast_gpu.sh destroy <id>         # Clean up"
        ;;
esac
