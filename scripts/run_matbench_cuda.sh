#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-matbench_phonons}"
FOLD="${FOLD:-0}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmarks}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
NUM_CONVS="${NUM_CONVS:-4}"
CONV_KAN_HIDDEN_DIM="${CONV_KAN_HIDDEN_DIM:-16}"
CONV_KAN_IMPL="${CONV_KAN_IMPL:-fastkan}"
CONV_KAN_GRID_SIZE="${CONV_KAN_GRID_SIZE:-8}"
EDGE_DIM="${EDGE_DIM:-41}"
FORWARD_ITERS="${FORWARD_ITERS:-10}"
WARMUP_ITERS="${WARMUP_ITERS:-2}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-50}"
EPOCH_PAUSE_SECONDS="${EPOCH_PAUSE_SECONDS:-0.0}"

python scripts/benchmark_matbench.py \
  --dataset "${DATASET}" \
  --fold "${FOLD}" \
  --conv-nets mlp kan \
  --device cuda \
  --require-cuda \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --head-hidden-dims 32 \
  --num-convs "${NUM_CONVS}" \
  --conv-kan-impl "${CONV_KAN_IMPL}" \
  --conv-kan-hidden-dim "${CONV_KAN_HIDDEN_DIM}" \
  --conv-kan-grid-size "${CONV_KAN_GRID_SIZE}" \
  --edge-dim "${EDGE_DIM}" \
  --num-workers "${NUM_WORKERS}" \
  --forward-iters "${FORWARD_ITERS}" \
  --warmup-iters "${WARMUP_ITERS}" \
  --log-every-steps "${LOG_EVERY_STEPS}" \
  --epoch-pause-seconds "${EPOCH_PAUSE_SECONDS}" \
  --output-dir "${OUTPUT_DIR}" \
  --pin-memory \
  --persistent-workers
