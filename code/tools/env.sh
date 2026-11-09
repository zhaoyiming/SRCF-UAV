#!/usr/bin/env bash
# Source this file before training/evaluation. All values can be overridden.
#   STREAMRC_ENV        conda env prefix (optional; default: python on PATH)
#   STREAMRC_CUDA_HOME  CUDA toolkit used to build models/csrc (optional)
#   STREAMRC_DATA_ROOT  dataset root (default: <repo>/data/nuscenes)
#   STREAMRC_OUTPUT_ROOT output root (default: <repo>/outputs)
export STREAMRC_ROOT="${STREAMRC_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -n "${STREAMRC_ENV:-}" ]]; then
    export STREAMRC_PYTHON="${STREAMRC_PYTHON:-$STREAMRC_ENV/bin/python}"
    export PATH="$STREAMRC_ENV/bin:$PATH"
    export LD_LIBRARY_PATH="$STREAMRC_ENV/lib:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="$STREAMRC_ENV/lib:${LIBRARY_PATH:-}"
else
    export STREAMRC_PYTHON="${STREAMRC_PYTHON:-python}"
fi
if [[ -n "${STREAMRC_CUDA_HOME:-}" ]]; then
    export CUDA_HOME="$STREAMRC_CUDA_HOME"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
export STREAMRC_DATA_ROOT="${STREAMRC_DATA_ROOT:-$STREAMRC_ROOT/data/nuscenes}"
export STREAMRC_OUTPUT_ROOT="${STREAMRC_OUTPUT_ROOT:-$STREAMRC_ROOT/outputs}"
# Project root first (custom models/loaders), then the bundled UAV nuScenes devkit.
export PYTHONPATH="$STREAMRC_ROOT:$STREAMRC_ROOT/nuscenes-devkit/python-sdk:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
