#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-matbench_phonons}"
FOLD="${FOLD:-0}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmarks}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
KAN_HEAD_HIDDEN_DIMS="${KAN_HEAD_HIDDEN_DIMS:-8}"
MLP_HEAD_NET="${MLP_HEAD_NET:-mlp}"
KAN_HEAD_NET="${KAN_HEAD_NET:-kan}"
NUM_CONVS="${NUM_CONVS:-4}"
CONV_KAN_HIDDEN_DIM="${CONV_KAN_HIDDEN_DIM:-16}"
CONV_KAN_IMPL="${CONV_KAN_IMPL:-fastkan}"
CONV_KAN_GRID_SIZE="${CONV_KAN_GRID_SIZE:-3}"
EDGE_DIM="${EDGE_DIM:-41}"
CUTOFF="${CUTOFF:-6.0}"
ATOM_FEATURES="${ATOM_FEATURES:-}"
EDGE_FEATURES="${EDGE_FEATURES:-}"
MLP_ATOM_FEATURES="${MLP_ATOM_FEATURES:-cgcnn}"
MLP_EDGE_FEATURES="${MLP_EDGE_FEATURES:-gaussian}"
KAN_ATOM_FEATURES="${KAN_ATOM_FEATURES:-cgcnn}"
KAN_EDGE_FEATURES="${KAN_EDGE_FEATURES:-gaussian}"
LR="${LR:-3e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
MLP_LR="${MLP_LR:-}"
KAN_LR="${KAN_LR:-0.003}"
MLP_WEIGHT_DECAY="${MLP_WEIGHT_DECAY:-}"
KAN_WEIGHT_DECAY="${KAN_WEIGHT_DECAY:-0.0}"
FORWARD_ITERS="${FORWARD_ITERS:-10}"
WARMUP_ITERS="${WARMUP_ITERS:-2}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-50}"
EPOCH_PAUSE_SECONDS="${EPOCH_PAUSE_SECONDS:-0.0}"

CMD=(
  python scripts/benchmark_matbench.py
  --dataset "${DATASET}"
  --fold "${FOLD}"
  --conv-nets mlp kan
  --device cuda
  --require-cuda
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --hidden-dim "${HIDDEN_DIM}"
  --head-hidden-dims 32
  --kan-head-hidden-dims "${KAN_HEAD_HIDDEN_DIMS}"
  --mlp-head-net "${MLP_HEAD_NET}"
  --kan-head-net "${KAN_HEAD_NET}"
  --num-convs "${NUM_CONVS}"
  --conv-kan-impl "${CONV_KAN_IMPL}"
  --conv-kan-hidden-dim "${CONV_KAN_HIDDEN_DIM}"
  --conv-kan-grid-size "${CONV_KAN_GRID_SIZE}"
  --edge-dim "${EDGE_DIM}"
  --cutoff "${CUTOFF}"
  --mlp-atom-features "${MLP_ATOM_FEATURES}"
  --mlp-edge-features "${MLP_EDGE_FEATURES}"
  --kan-atom-features "${KAN_ATOM_FEATURES}"
  --kan-edge-features "${KAN_EDGE_FEATURES}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --num-workers "${NUM_WORKERS}"
  --forward-iters "${FORWARD_ITERS}"
  --warmup-iters "${WARMUP_ITERS}"
  --log-every-steps "${LOG_EVERY_STEPS}"
  --epoch-pause-seconds "${EPOCH_PAUSE_SECONDS}"
  --output-dir "${OUTPUT_DIR}"
  --pin-memory
  --persistent-workers
)

if [[ -n "${MLP_LR}" ]]; then
  CMD+=(--mlp-lr "${MLP_LR}")
fi
if [[ -n "${ATOM_FEATURES}" ]]; then
  CMD+=(--atom-features "${ATOM_FEATURES}")
fi
if [[ -n "${EDGE_FEATURES}" ]]; then
  CMD+=(--edge-features "${EDGE_FEATURES}")
fi
if [[ -n "${KAN_LR}" ]]; then
  CMD+=(--kan-lr "${KAN_LR}")
fi
if [[ -n "${MLP_WEIGHT_DECAY}" ]]; then
  CMD+=(--mlp-weight-decay "${MLP_WEIGHT_DECAY}")
fi
if [[ -n "${KAN_WEIGHT_DECAY}" ]]; then
  CMD+=(--kan-weight-decay "${KAN_WEIGHT_DECAY}")
fi

"${CMD[@]}"
