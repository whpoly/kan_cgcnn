#!/usr/bin/env bash
set -euo pipefail

TASK_SET="all"
TASKS=()
OFFICIAL_ENV="modnet-v012-matbench"
KAN_ENV="kan-cgcnn-cuda"
OFFICIAL_OUTPUT_DIR=""
KAN_OUTPUT_ROOT=""
N_JOBS=4
EXPORT_MAX_FEATURES=512
N_MODELS=5
NESTED_FOLDS=5
FAST_OFFICIAL=0
SKIP_OFFICIAL=0
MODELS=(mlp fastkan spline)
FOLDS=(0 1 2 3 4)
N_FEATURES=512
COMMON_DIMS=(512)
GROUP_DIMS=(256)
PROPERTY_DIMS=(64)
TARGET_DIMS=(64)
MLP_COMMON_DIMS=()
MLP_GROUP_DIMS=()
MLP_PROPERTY_DIMS=()
MLP_TARGET_DIMS=()
KAN_COMMON_DIMS=()
KAN_GROUP_DIMS=()
KAN_PROPERTY_DIMS=()
KAN_TARGET_DIMS=()
KAN_LOSS="auto"
KAN_GRID_SIZE=3
KAN_SPLINE_ORDER=3
EPOCHS=300
BATCH_SIZE=64
VAL_RATIO=0.1
EARLY_STOPPING_PATIENCE=60
LR=0.001
WEIGHT_DECAY=0.000001
SCALER="minmax"
TARGET_SCALE="none"
IMPUTE_STRATEGY="median"
DROPOUT=0
PRUNE_KAN_FRACTION=0.5
DEVICE="cuda"
REQUIRE_CUDA=0
EXPORT_FORMULAS=1
FORMULA_TOP_K=20
FORMULA_MIN_ABS=0
LOG_EVERY_EPOCHS=10
FORWARD_ITERS=20
WARMUP_ITERS=5
NO_MATBENCH_RECORDS=0
SETUP_ENVS=0
DRY_RUN=0

ALL_TASKS=(
  matbench_dielectric
  matbench_expt_gap
  matbench_expt_is_metal
  matbench_glass
  matbench_jdft2d
  matbench_elastic
  matbench_mp_e_form
  matbench_mp_gap
  matbench_mp_is_metal
  matbench_perovskites
  matbench_phonons
  matbench_steels
)

SMALL_TASKS=(
  matbench_dielectric
  matbench_expt_gap
  matbench_expt_is_metal
  matbench_glass
  matbench_jdft2d
  matbench_elastic
  matbench_perovskites
  matbench_phonons
  matbench_steels
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_modnet_kan_matbench.sh --tasks matbench_phonons --require-cuda

Main options:
  --setup-envs                 Create missing conda envs before running.
  --tasks <task...>            Matbench task names. Default: --task-set all.
  --task-set small|all         Task preset when --tasks is omitted.
  --skip-official              Reuse an existing --official-output-dir.
  --official-output-dir <dir>  Official MODNet output/features directory.
  --kan-output-root <dir>      KAN/MLP output root.
  --models <name...>           Default: mlp fastkan spline.
  --folds <fold...>            Default: 0 1 2 3 4.
  --epochs <n>                 KAN/MLP epochs. Default: 300.
  --kan-common-dims <dims...>  KAN-only common dims. Use smaller widths for interpretability.
  --patience <n>               Early stopping patience. Default: 60.
  --lr <x>                     Learning rate. Default: 0.001.
  --weight-decay <x>           AdamW weight decay. Default: 1e-6.
  --prune-kan-fraction <x>     Post-training global pruning fraction. Default: 0.5.
  --formula-top-k <n>          Terms per neuron in readable formula. Use 0 for exact.
  --no-export-formulas         Disable formula files.
  --require-cuda               Fail if CUDA is unavailable.
  --dry-run                    Print commands without running them.
EOF
}

read_values() {
  local -n target=$1
  shift
  target=()
  while [[ $# -gt 0 && $1 != --* ]]; do
    target+=("$1")
    shift
  done
  REMAINING_ARGS=("$@")
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --setup-envs) SETUP_ENVS=1; shift ;;
    --task-set) TASK_SET="$2"; shift 2 ;;
    --tasks) shift; read_values TASKS "$@"; set -- "${REMAINING_ARGS[@]}" ;;
    --official-env) OFFICIAL_ENV="$2"; shift 2 ;;
    --kan-env) KAN_ENV="$2"; shift 2 ;;
    --official-output-dir) OFFICIAL_OUTPUT_DIR="$2"; shift 2 ;;
    --kan-output-root) KAN_OUTPUT_ROOT="$2"; shift 2 ;;
    --n-jobs) N_JOBS="$2"; shift 2 ;;
    --export-max-features) EXPORT_MAX_FEATURES="$2"; shift 2 ;;
    --n-models) N_MODELS="$2"; shift 2 ;;
    --nested-folds) NESTED_FOLDS="$2"; shift 2 ;;
    --fast-official) FAST_OFFICIAL=1; shift ;;
    --skip-official) SKIP_OFFICIAL=1; shift ;;
    --models) shift; read_values MODELS "$@"; set -- "${REMAINING_ARGS[@]}" ;;
    --folds) shift; read_values FOLDS "$@"; set -- "${REMAINING_ARGS[@]}" ;;
    --n-features) N_FEATURES="$2"; shift 2 ;;
    --common-dims) shift; read_values COMMON_DIMS "$@"; set -- "${REMAINING_ARGS[@]}" ;;
    --group-dims) shift; read_values GROUP_DIMS "$@"; set -- "${REMAINING_ARGS[@]}" ;;
    --property-dims) shift; read_values PROPERTY_DIMS "$@"; set -- "${REMAINING_ARGS[@]}" ;;
    --target-dims) shift; read_values TARGET_DIMS "$@"; set -- "${REMAINING_ARGS[@]}" ;;
    --mlp-common-dims) shift; read_values MLP_COMMON_DIMS "$@"; set -- "${REMAINING_ARGS[@]}" ;;
    --mlp-group-dims) shift; read_values MLP_GROUP_DIMS "$@"; set -- "${REMAINING_ARGS[@]}" ;;
    --mlp-property-dims) shift; read_values MLP_PROPERTY_DIMS "$@"; set -- "${REMAINING_ARGS[@]}" ;;
    --mlp-target-dims) shift; read_values MLP_TARGET_DIMS "$@"; set -- "${REMAINING_ARGS[@]}" ;;
    --kan-common-dims) shift; read_values KAN_COMMON_DIMS "$@"; set -- "${REMAINING_ARGS[@]}" ;;
    --kan-group-dims) shift; read_values KAN_GROUP_DIMS "$@"; set -- "${REMAINING_ARGS[@]}" ;;
    --kan-property-dims) shift; read_values KAN_PROPERTY_DIMS "$@"; set -- "${REMAINING_ARGS[@]}" ;;
    --kan-target-dims) shift; read_values KAN_TARGET_DIMS "$@"; set -- "${REMAINING_ARGS[@]}" ;;
    --kan-loss) KAN_LOSS="$2"; shift 2 ;;
    --kan-grid-size) KAN_GRID_SIZE="$2"; shift 2 ;;
    --kan-spline-order) KAN_SPLINE_ORDER="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --val-ratio) VAL_RATIO="$2"; shift 2 ;;
    --patience|--early-stopping-patience) EARLY_STOPPING_PATIENCE="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --scaler) SCALER="$2"; shift 2 ;;
    --target-scale) TARGET_SCALE="$2"; shift 2 ;;
    --impute-strategy) IMPUTE_STRATEGY="$2"; shift 2 ;;
    --dropout) DROPOUT="$2"; shift 2 ;;
    --prune-kan-fraction) PRUNE_KAN_FRACTION="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --require-cuda) REQUIRE_CUDA=1; shift ;;
    --no-export-formulas) EXPORT_FORMULAS=0; shift ;;
    --formula-top-k) FORMULA_TOP_K="$2"; shift 2 ;;
    --formula-min-abs) FORMULA_MIN_ABS="$2"; shift 2 ;;
    --log-every-epochs) LOG_EVERY_EPOCHS="$2"; shift 2 ;;
    --forward-iters) FORWARD_ITERS="$2"; shift 2 ;;
    --warmup-iters) WARMUP_ITERS="$2"; shift 2 ;;
    --no-matbench-records) NO_MATBENCH_RECORDS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ${#TASKS[@]} -eq 0 ]]; then
  if [[ "$TASK_SET" == "all" ]]; then
    TASKS=("${ALL_TASKS[@]}")
  elif [[ "$TASK_SET" == "small" ]]; then
    TASKS=("${SMALL_TASKS[@]}")
  else
    echo "--task-set must be small or all" >&2
    exit 2
  fi
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
if [[ -z "$OFFICIAL_OUTPUT_DIR" ]]; then
  OFFICIAL_OUTPUT_DIR="benchmarks/official-modnet-v012-${TASK_SET}-${STAMP}"
fi
if [[ -z "$KAN_OUTPUT_ROOT" ]]; then
  KAN_OUTPUT_ROOT="benchmarks/kan-on-official-modnet-features-${TASK_SET}-${STAMP}"
fi

run_step() {
  local title="$1"
  shift
  echo
  echo "=== ${title} ==="
  printf '%q ' "$@"
  echo
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

env_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "$1"
}

if [[ "$SETUP_ENVS" -eq 1 ]]; then
  if ! env_exists "$KAN_ENV"; then
    run_step "Create KAN conda env" bash scripts/setup_conda_cuda.sh "$KAN_ENV"
  fi
  if [[ "$SKIP_OFFICIAL" -eq 0 ]] && ! env_exists "$OFFICIAL_ENV"; then
    run_step "Create official MODNet conda env" conda env create -n "$OFFICIAL_ENV" -f environment-modnet-v012.yml
  fi
fi

mkdir -p "$KAN_OUTPUT_ROOT"
{
  echo "task_set=${TASK_SET}"
  echo "tasks=${TASKS[*]}"
  echo "official_env=${OFFICIAL_ENV}"
  echo "kan_env=${KAN_ENV}"
  echo "official_output_dir=${OFFICIAL_OUTPUT_DIR}"
  echo "kan_output_root=${KAN_OUTPUT_ROOT}"
  echo "models=${MODELS[*]}"
  echo "folds=${FOLDS[*]}"
  echo "n_features=${N_FEATURES}"
  echo "common_dims=${COMMON_DIMS[*]}"
  echo "group_dims=${GROUP_DIMS[*]}"
  echo "property_dims=${PROPERTY_DIMS[*]}"
  echo "target_dims=${TARGET_DIMS[*]}"
  echo "mlp_common_dims=${MLP_COMMON_DIMS[*]}"
  echo "mlp_group_dims=${MLP_GROUP_DIMS[*]}"
  echo "mlp_property_dims=${MLP_PROPERTY_DIMS[*]}"
  echo "mlp_target_dims=${MLP_TARGET_DIMS[*]}"
  echo "kan_common_dims=${KAN_COMMON_DIMS[*]}"
  echo "kan_group_dims=${KAN_GROUP_DIMS[*]}"
  echo "kan_property_dims=${KAN_PROPERTY_DIMS[*]}"
  echo "kan_target_dims=${KAN_TARGET_DIMS[*]}"
  echo "kan_grid_size=${KAN_GRID_SIZE}"
  echo "epochs=${EPOCHS}"
  echo "batch_size=${BATCH_SIZE}"
  echo "val_ratio=${VAL_RATIO}"
  echo "early_stopping_patience=${EARLY_STOPPING_PATIENCE}"
  echo "lr=${LR}"
  echo "weight_decay=${WEIGHT_DECAY}"
  echo "prune_kan_fraction=${PRUNE_KAN_FRACTION}"
  echo "formula_top_k=${FORMULA_TOP_K}"
  echo "started_at=$(date -Iseconds)"
} > "${KAN_OUTPUT_ROOT}/combined-run-metadata.txt"

if [[ "$SKIP_OFFICIAL" -eq 0 ]]; then
  official_args=(
    conda run --no-capture-output
    -n "$OFFICIAL_ENV"
    python -u scripts/run_official_modnet_matbench.py
    --tasks "${TASKS[@]}"
    --n-jobs "$N_JOBS"
    --nested-folds "$NESTED_FOLDS"
    --n-models "$N_MODELS"
    --export-feature-folds
    --export-max-features "$EXPORT_MAX_FEATURES"
    --output-dir "$OFFICIAL_OUTPUT_DIR"
  )
  if [[ "$FAST_OFFICIAL" -eq 1 ]]; then
    official_args+=(--fast)
  fi
  run_step "Official MODNet benchmark and feature export" "${official_args[@]}"
else
  echo "Skipping official MODNet. Using existing features under: ${OFFICIAL_OUTPUT_DIR}"
fi

for task in "${TASKS[@]}"; do
  feature_dir="${OFFICIAL_OUTPUT_DIR}/${task}/official_feature_folds"
  if [[ "$DRY_RUN" -eq 0 && ! -d "$feature_dir" ]]; then
    echo "Missing official feature directory: ${feature_dir}" >&2
    exit 1
  fi

  task_out="${KAN_OUTPUT_ROOT}/${task}"
  mkdir -p "$task_out"
  kan_args=(
    conda run --no-capture-output
    -n "$KAN_ENV"
    python -u scripts/benchmark_modnet_kan.py
    --dataset "$task"
    --folds "${FOLDS[@]}"
    --models "${MODELS[@]}"
    --precomputed-feature-dir "$feature_dir"
    --n-features "$N_FEATURES"
    --common-dims "${COMMON_DIMS[@]}"
    --group-dims "${GROUP_DIMS[@]}"
    --property-dims "${PROPERTY_DIMS[@]}"
  )
  if [[ ${#TARGET_DIMS[@]} -gt 0 ]]; then
    kan_args+=(--target-dims "${TARGET_DIMS[@]}")
  fi
  if [[ ${#MLP_COMMON_DIMS[@]} -gt 0 ]]; then
    kan_args+=(--mlp-common-dims "${MLP_COMMON_DIMS[@]}")
  fi
  if [[ ${#MLP_GROUP_DIMS[@]} -gt 0 ]]; then
    kan_args+=(--mlp-group-dims "${MLP_GROUP_DIMS[@]}")
  fi
  if [[ ${#MLP_PROPERTY_DIMS[@]} -gt 0 ]]; then
    kan_args+=(--mlp-property-dims "${MLP_PROPERTY_DIMS[@]}")
  fi
  if [[ ${#MLP_TARGET_DIMS[@]} -gt 0 ]]; then
    kan_args+=(--mlp-target-dims "${MLP_TARGET_DIMS[@]}")
  fi
  if [[ ${#KAN_COMMON_DIMS[@]} -gt 0 ]]; then
    kan_args+=(--kan-common-dims "${KAN_COMMON_DIMS[@]}")
  fi
  if [[ ${#KAN_GROUP_DIMS[@]} -gt 0 ]]; then
    kan_args+=(--kan-group-dims "${KAN_GROUP_DIMS[@]}")
  fi
  if [[ ${#KAN_PROPERTY_DIMS[@]} -gt 0 ]]; then
    kan_args+=(--kan-property-dims "${KAN_PROPERTY_DIMS[@]}")
  fi
  if [[ ${#KAN_TARGET_DIMS[@]} -gt 0 ]]; then
    kan_args+=(--kan-target-dims "${KAN_TARGET_DIMS[@]}")
  fi
  kan_args+=(
    --loss "$KAN_LOSS"
    --kan-grid-size "$KAN_GRID_SIZE"
    --kan-spline-order "$KAN_SPLINE_ORDER"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --val-ratio "$VAL_RATIO"
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
    --lr "$LR"
    --weight-decay "$WEIGHT_DECAY"
    --scaler "$SCALER"
    --target-scale "$TARGET_SCALE"
    --impute-strategy "$IMPUTE_STRATEGY"
    --dropout "$DROPOUT"
    --prune-kan-fraction "$PRUNE_KAN_FRACTION"
    --log-every-epochs "$LOG_EVERY_EPOCHS"
    --forward-iters "$FORWARD_ITERS"
    --warmup-iters "$WARMUP_ITERS"
    --output-dir "$task_out"
    --device "$DEVICE"
  )
  if [[ "$REQUIRE_CUDA" -eq 1 ]]; then
    kan_args+=(--require-cuda)
  fi
  if [[ "$EXPORT_FORMULAS" -eq 1 ]]; then
    kan_args+=(--export-formulas --formula-top-k "$FORMULA_TOP_K" --formula-min-abs "$FORMULA_MIN_ABS")
  fi
  if [[ "$NO_MATBENCH_RECORDS" -eq 1 ]]; then
    kan_args+=(--no-matbench-records)
  fi
  run_step "KAN/MLP on official MODNet features: ${task}" "${kan_args[@]}"
done

echo
echo "Combined run complete."
echo "Official MODNet outputs: ${OFFICIAL_OUTPUT_DIR}"
echo "KAN outputs: ${KAN_OUTPUT_ROOT}"
