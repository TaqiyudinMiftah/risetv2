#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/caernet.yaml"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config PATH      Custom config file path (default: configs/caernet.yaml)"
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
echo "Building CAER-S Manifest"
echo "================================"
echo "Config: $CONFIG"

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
