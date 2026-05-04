#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/caers_data.yaml"

echo "================================"
echo "Smoke Test Data Pipeline"
echo "================================"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

cd "$PROJECT_ROOT"

python scripts/smoke_data_pipeline.py --config "$CONFIG" --batch-size 4

echo ""
echo "Smoke test complete!"
