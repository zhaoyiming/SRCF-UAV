#!/usr/bin/env bash
# Usage: bash tools/dist_train.sh <config> [train.py args...]
# GPU count: STREAMRC_GPUS, else SLURM_GPUS_ON_NODE, else all visible GPUs.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/env.sh"
cd "$STREAMRC_ROOT"
config="${STREAMRC_CONFIG:-$1}"
shift
gpu_num="${STREAMRC_GPUS:-${SLURM_GPUS_ON_NODE:-}}"
if [[ -z "$gpu_num" ]]; then
    gpu_num=$("$STREAMRC_PYTHON" -c 'import torch; print(torch.cuda.device_count())')
fi
if ! [[ "$gpu_num" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid GPU count: $gpu_num" >&2
    exit 2
fi
test -f "$config" || { echo "Config not found: $config" >&2; exit 2; }
# Annotation files store media paths relative to the repo root (data/nuscenes/...).
test -d data/nuscenes/samples || { echo "Missing data/nuscenes/samples (see README: Data)" >&2; exit 2; }
printf 'Project: %s\nPython: %s\nData: %s\nConfig: %s\nGPUs: %s\n' \
    "$STREAMRC_ROOT" "$STREAMRC_PYTHON" "$STREAMRC_DATA_ROOT" "$config" "$gpu_num"
exec "$STREAMRC_PYTHON" -m torch.distributed.run --standalone \
    --nnodes=1 --nproc_per_node="$gpu_num" train.py --config "$config" "$@"
