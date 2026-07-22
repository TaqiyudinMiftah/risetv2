#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <compatible-pytorch-rocm-index-url>" >&2
    echo "Select the URL for the exact AMD GPU, OS, Python, and ROCm versions." >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
index_url="$1"
python_version="${PYTHON_VERSION:-3.12}"
torch_spec="${TORCH_SPEC:-torch}"
torchvision_spec="${TORCHVISION_SPEC:-torchvision}"

cd "$repo_root"
command -v uv >/dev/null || {
    echo "uv is required. Install it before running this script." >&2
    exit 2
}

uv venv --python "$python_version"
uv pip install --python .venv/bin/python --index-url "$index_url" "$torch_spec" "$torchvision_spec"
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/check_accelerator.py --require-backend rocm --min-devices 1
