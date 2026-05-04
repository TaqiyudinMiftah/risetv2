#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/caers_data.yaml"

echo "================================"
echo "Building CAER-S Manifest"
echo "================================"

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

cd "$PROJECT_ROOT"
python scripts/build_caers_manifest.py --config "$CONFIG"

echo ""
echo "Manifest build complete!"
echo "  Manifest: artifacts/caers/manifest_caers.jsonl"
echo "  Diagnostics: artifacts/caers/diagnostics_caers.json"
