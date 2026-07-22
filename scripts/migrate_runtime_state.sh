#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <ssh-host> [remote-repo-root]" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remote_host="$1"
remote_root="${2:-riset/risetv2}"
rsync_args=(-a --partial --info=progress2)

ssh "$remote_host" "mkdir -p '$remote_root'"

for path in CAER-S artifacts checkpoints detectors paper; do
    if [[ -e "$repo_root/$path" ]]; then
        rsync "${rsync_args[@]}" "$repo_root/$path/" "$remote_host:$remote_root/$path/"
    fi
done

models_root="$repo_root/third_party/CAER/CAER/official_runs/models"
if [[ -d "$models_root" ]]; then
    rsync "${rsync_args[@]}" \
        --include='*/' --include='model_best.pth' --include='config.json' --exclude='*' \
        "$models_root/" \
        "$remote_host:$remote_root/third_party/CAER/CAER/official_runs/models/"
fi

echo "Runtime state copied. Large upstream logs, W&B media, .venv, and caches were excluded."
