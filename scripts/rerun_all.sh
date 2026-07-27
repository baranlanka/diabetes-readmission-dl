#!/usr/bin/env bash
# Full pipeline re-run after the A1Cresult / max_glu_serum "Not tested" fix.
# Runs notebooks 01 -> 04 in order, stopping immediately if any one fails.
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
VENV="$PROJECT_ROOT/.venv"
LOG_DIR="$PROJECT_ROOT/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/rerun_$(date +%Y%m%d_%H%M%S).log"

NV="$VENV/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$(find "$NV" -name lib -type d | sort | tr '\n' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# A finished nbconvert process does not release the GPU instantly - the kernel is torn
# down asynchronously. Starting the next notebook immediately produces
#   InternalError: cudaSetDevice() on GPU:0 failed ... device(s) is/are busy
# so wait for the card to actually come free before each notebook.
wait_for_gpu() {
  for _ in $(seq 1 60); do
    if ! pgrep -f "ipykernel_launcher" >/dev/null 2>&1; then
      sleep 5   # settle time after the kernel process disappears
      return 0
    fi
    sleep 5
  done
  echo "WARNING: kernels still alive after 5 min; proceeding anyway" | tee -a "$LOG"
}

echo "Log: $LOG"
for nb in ${NOTEBOOKS:-01_eda_and_preprocessing 02_model_building 03_hyperparameter_tuning 04_evaluation_and_ensemble}; do
  wait_for_gpu
  echo "[$(date '+%H:%M:%S')] ===== START $nb =====" | tee -a "$LOG"
  "$VENV/bin/jupyter" nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.kernel_name=python3 \
      --ExecutePreprocessor.timeout=7200 \
      "notebooks/$nb.ipynb" >> "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date '+%H:%M:%S')] ===== FAILED $nb (rc=$rc) - stopping cascade =====" | tee -a "$LOG"
    exit $rc
  fi
  echo "[$(date '+%H:%M:%S')] ===== DONE $nb =====" | tee -a "$LOG"
done
echo "[$(date '+%H:%M:%S')] ===== CASCADE COMPLETE =====" | tee -a "$LOG"
