#!/usr/bin/env bash
# Usage: bash val.sh <checkpoint.pth> [val.py args...]
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
weights=$1; shift
exec bash "$script_dir/tools/dist_val.sh" configs/r50_fusion_streaming_full.py "$weights" "$@"
