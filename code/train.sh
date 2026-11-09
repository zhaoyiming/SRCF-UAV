#!/usr/bin/env bash
# Train the paper R50 camera-radar model. Extra args go to train.py.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/tools/dist_train.sh" configs/r50_fusion_streaming_full.py "$@"
