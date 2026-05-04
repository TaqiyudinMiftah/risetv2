#!/usr/bin/env bash
set -euo pipefail

echo "================================"
echo "CAER-S Pipeline UV Setup"
echo "================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed. Install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Python version: $(uv python --version 2>/dev/null || echo 'not managed by uv')"
echo ""

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with Python 3.12..."
    uv venv --python 3.12
else
    echo "Virtual environment already exists."
fi

echo ""
echo "Installing dependencies with uv..."
uv pip install -e ".[dev]"

echo ""
echo "Setup complete! Activate with:"
echo "  source .venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Update configs/caers_data.yaml with your dataset path"
echo "  2. Run: ./bin/build_manifest.sh"
echo "  3. Run: ./bin/train.sh"
