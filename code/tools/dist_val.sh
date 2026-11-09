#!/usr/bin/env bash
# Usage: bash tools/dist_val.sh <config> <checkpoint.pth> [val.py args...]
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/env.sh"
cd "$STREAMRC_ROOT"
config=$1; weights=$2; shift 2
gpu_num="${STREAMRC_GPUS:-${SLURM_GPUS_ON_NODE:-}}"
if [[ -z "$gpu_num" ]]; then
    gpu_num=$("$STREAMRC_PYTHON" -c 'import torch; print(torch.cuda.device_count())')
fi
test -d data/nuscenes/samples || { echo "Missing data/nuscenes/samples (see README: Data)" >&2; exit 2; }
exec "$STREAMRC_PYTHON" -m torch.distributed.run --standalone \
    --nnodes=1 --nproc_per_node="$gpu_num" val.py --config "$config" --weights "$weights" "$@"
