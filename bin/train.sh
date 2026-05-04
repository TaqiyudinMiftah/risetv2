#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/caers_data.yaml"

# W&B Configuration
WANDB_API_KEY="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"
WANDB_PROJECT="caers-emotion-recognition"
WANDB_ENTITY="Tim-1"
WANDB_RUN_NAME="CAER-Dual Stream-$(date +%Y%m%d-%H%M%S)"
WANDB_OFFLINE=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --run-name)
            WANDB_RUN_NAME="$2"
            shift 2
            ;;
        --offline)
            WANDB_OFFLINE="--wandb-offline"
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --resume PATH      Resume from checkpoint"
            echo "  --run-name NAME    W&B run name"
            echo "  --offline          Run W&B in offline mode"
            echo "  --config PATH      Custom config file path"
            echo "  --help, -h         Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "================================"
echo "Training CAER-Net"
echo "================================"
echo "Config: $CONFIG"
echo "W&B Project: $WANDB_PROJECT"
[ -n "$WANDB_RUN_NAME" ] && echo "W&B Run Name: $WANDB_RUN_NAME"
[ -n "${RESUME_CHECKPOINT:-}" ] && echo "Resume: $RESUME_CHECKPOINT"
echo ""

cd "$PROJECT_ROOT"

# Build command
CMD=(
    python scripts/train_caers.py
    --config "$CONFIG"
    --wandb-api-key "$WANDB_API_KEY"
    --wandb-project "$WANDB_PROJECT"
)

[ -n "$WANDB_ENTITY" ] && CMD+=(--wandb-entity "$WANDB_ENTITY")
[ -n "$WANDB_RUN_NAME" ] && CMD+=(--wandb-run-name "$WANDB_RUN_NAME")
[ -n "$WANDB_OFFLINE" ] && CMD+=($WANDB_OFFLINE)
[ -n "${RESUME_CHECKPOINT:-}" ] && CMD+=(--resume "$RESUME_CHECKPOINT")

# Run training
echo "Running: ${CMD[*]}"
echo ""
"${CMD[@]}"

echo ""
echo "Training complete!"
echo "Checkpoints saved to: checkpoints/caers/"
