#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/caers_data.yaml"
CHECKPOINT="${PROJECT_ROOT}/checkpoints/caers/best_model.pt"
SPLIT="test"

# W&B Configuration
WANDB_API_KEY="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"
WANDB_PROJECT="caers-emotion-recognition"
WANDB_ENTITY=""
WANDB_RUN_NAME=""
WANDB_OFFLINE=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
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
            echo "  --checkpoint PATH  Path to model checkpoint (default: checkpoints/caers/best_model.pt)"
            echo "  --split SPLIT      Split to evaluate: test or val (default: test)"
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
echo "Evaluating CAER-Net"
echo "================================"
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Split: $SPLIT"
echo "W&B Project: $WANDB_PROJECT"
[ -n "$WANDB_RUN_NAME" ] && echo "W&B Run Name: $WANDB_RUN_NAME"
echo ""

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Train first with: ./bin/train.sh"
    exit 1
fi

cd "$PROJECT_ROOT"

# Build command
CMD=(
    python scripts/evaluate_caers.py
    --config "$CONFIG"
    --checkpoint "$CHECKPOINT"
    --split "$SPLIT"
    --wandb-api-key "$WANDB_API_KEY"
    --wandb-project "$WANDB_PROJECT"
)

[ -n "$WANDB_ENTITY" ] && CMD+=(--wandb-entity "$WANDB_ENTITY")
[ -n "$WANDB_RUN_NAME" ] && CMD+=(--wandb-run-name "$WANDB_RUN_NAME")
[ -n "$WANDB_OFFLINE" ] && CMD+=($WANDB_OFFLINE)

# Run evaluation
echo "Running: ${CMD[*]}"
echo ""
"${CMD[@]}"

echo ""
echo "Evaluation complete!"
