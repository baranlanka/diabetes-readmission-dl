#!/usr/bin/env bash
# Launches the overnight hyperparameter-tuning run
# (notebooks/03_hyperparameter_tuning.ipynb).
#
# Usage (safe to background - survives the terminal closing):
#   nohup ./scripts/run_tuning.sh > /dev/null 2>&1 &
#
# The script itself redirects all jupyter/nbconvert output to a timestamped
# log file under logs/, so closing the terminal neither stops the run nor
# loses the log; `nohup` additionally detaches the process from the
# terminal's SIGHUP.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/.venv"
# Defaults to the real tuning notebook. An optional first argument overrides it,
# which allows a rehearsal run against a SMOKE_TEST=True copy without touching
# the deliverable notebook.
NOTEBOOK="${1:-$PROJECT_ROOT/notebooks/03_hyperparameter_tuning.ipynb}"
LOG_DIR="$PROJECT_ROOT/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/tuning_${TIMESTAMP}.log"

# Generous timeout for the whole `nbconvert --execute` process (seconds).
# The budgeted run is 2-4h (docs/04-master-plan.md Phase 4); 6h leaves
# headroom without risking nbconvert killing a legitimately slow run.
NBCONVERT_TIMEOUT=21600

mkdir -p "$LOG_DIR"

# CUDA libraries shipped via the `nvidia-*` pip wheels are not on the
# loader path by default, so a bare `python`/`jupyter` reports "GPUs: []"
# (docs/04-master-plan.md Section 1, D3 gotcha). These are the same
# directories baked into the python3 Jupyter kernel's env block and
# appended to .venv/bin/activate.
NVIDIA_LIB_DIR="$VENV_DIR/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB_DIR/cublas/lib:$NVIDIA_LIB_DIR/cuda_cupti/lib:$NVIDIA_LIB_DIR/cuda_nvrtc/lib:$NVIDIA_LIB_DIR/cuda_runtime/lib:$NVIDIA_LIB_DIR/cudnn/lib:$NVIDIA_LIB_DIR/cufft/lib:$NVIDIA_LIB_DIR/curand/lib:$NVIDIA_LIB_DIR/cusolver/lib:$NVIDIA_LIB_DIR/cusparse/lib:$NVIDIA_LIB_DIR/nccl/lib:$NVIDIA_LIB_DIR/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

{
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting overnight hyperparameter-tuning run."
    echo "Notebook       : $NOTEBOOK"
    echo "Kernel         : python3"
    echo "nbconvert timeout: ${NBCONVERT_TIMEOUT}s"
    echo "Log file       : $LOG_FILE"
} | tee -a "$LOG_FILE"

echo "Started. Tail this file to follow progress:"
echo "  tail -f $LOG_FILE"

"$VENV_DIR/bin/jupyter" nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout="$NBCONVERT_TIMEOUT" \
    "$NOTEBOOK" >> "$LOG_FILE" 2>&1
STATUS=$?

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Run finished with exit code $STATUS." | tee -a "$LOG_FILE"
exit "$STATUS"
