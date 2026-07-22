#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
session="${TMUX_SESSION:-caer-clean-s42}"
device_ids="${DEVICE_IDS:-0}"
n_gpu="${N_GPU:-1}"
seed="${SEED:-42}"
config="${CONFIG:-configs/experiments/caernet_clean_content_disjoint_exploratory_seed42.json}"
run_id="${RUN_ID:-caernet__clean_inrepo__seed${seed}__$(date -u +%Y%m%d_%H%M%S)}"
log_path="artifacts/launch_logs/${run_id}.log"
hsa_override="${HSA_OVERRIDE_GFX_VERSION:-}"

command -v tmux >/dev/null || {
    echo "tmux is required." >&2
    exit 2
}
if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 2
fi

mkdir -p "$repo_root/artifacts/launch_logs"
accelerator_env=""
if [[ -n "$hsa_override" ]]; then
    printf -v accelerator_env 'HSA_OVERRIDE_GFX_VERSION=%q ' "$hsa_override"
fi
printf -v launch_command \
    'cd %q && %sPYTHONUNBUFFERED=1 .venv/bin/python run_caer_clean.py train --config %q --seed %q --run-id %q --device %q --n-gpu %q --wandb-mode offline > %q 2>&1' \
    "$repo_root" "$accelerator_env" "$config" "$seed" "$run_id" "$device_ids" "$n_gpu" "$log_path"
tmux new-session -d -s "$session" "$launch_command"

echo "Session: $session"
echo "Run ID: $run_id"
echo "Log: $log_path"
