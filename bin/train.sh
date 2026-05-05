#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/caernet.yaml"

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
        --augment)
            AUGMENT="--augment"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --resume PATH      Resume from checkpoint"
            echo "  --run-name NAME    W&B run name"
            echo "  --offline          Run W&B in offline mode"
            echo "  --config PATH      Custom config file path"
            echo "  --augment          Enable data augmentation"
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

WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-caers-emotion-recognition}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_OFFLINE="${WANDB_OFFLINE:-}"
AUGMENT="${AUGMENT:-}"

echo "================================"
echo "Training Emotion Recognition"
echo "================================"
echo "Config: $CONFIG"
echo "W&B Project: $WANDB_PROJECT"
[ -n "$WANDB_RUN_NAME" ] && echo "W&B Run Name: $WANDB_RUN_NAME"
[ -n "${RESUME_CHECKPOINT:-}" ] && echo "Resume: $RESUME_CHECKPOINT"
echo ""

cd "$PROJECT_ROOT"

# Build command
CMD=(
    python scripts/train.py
    --config "$CONFIG"
)

[ -n "$WANDB_API_KEY" ] && CMD+=(--wandb-api-key "$WANDB_API_KEY")
[ -n "$WANDB_PROJECT" ] && CMD+=(--wandb-project "$WANDB_PROJECT")
[ -n "$WANDB_ENTITY" ] && CMD+=(--wandb-entity "$WANDB_ENTITY")
[ -n "$WANDB_RUN_NAME" ] && CMD+=(--wandb-run-name "$WANDB_RUN_NAME")
[ -n "$WANDB_OFFLINE" ] && CMD+=($WANDB_OFFLINE)
[ -n "${RESUME_CHECKPOINT:-}" ] && CMD+=(--resume "$RESUME_CHECKPOINT")
[ -n "$AUGMENT" ] && CMD+=($AUGMENT)

# Run training
echo "Running: ${CMD[*]}"
echo ""
"${CMD[@]}"

echo ""
echo "Training complete!"
