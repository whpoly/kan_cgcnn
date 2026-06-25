#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-matbench_phonons}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-50}"
EPOCH_PAUSE_SECONDS="${EPOCH_PAUSE_SECONDS:-0.0}"
CONV_KAN_IMPL="${CONV_KAN_IMPL:-fastkan}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/${DATASET}-5fold-${RUN_ID}}"

mkdir -p "${OUTPUT_DIR}"

for FOLD in 0 1 2 3 4; do
  echo "=== ${DATASET} fold ${FOLD}/4, ${EPOCHS} epochs ==="
  DATASET="${DATASET}" \
  FOLD="${FOLD}" \
  EPOCHS="${EPOCHS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  LOG_EVERY_STEPS="${LOG_EVERY_STEPS}" \
  EPOCH_PAUSE_SECONDS="${EPOCH_PAUSE_SECONDS}" \
  CONV_KAN_IMPL="${CONV_KAN_IMPL}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  bash scripts/run_matbench_cuda.sh
done

python scripts/summarize_matbench_results.py \
  --input-dir "${OUTPUT_DIR}" \
  --dataset "${DATASET}" \
  --expect-folds 5

echo "5-fold outputs: ${OUTPUT_DIR}"
