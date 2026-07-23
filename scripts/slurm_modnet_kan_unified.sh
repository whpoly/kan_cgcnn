#!/bin/bash
#SBATCH --job-name=modnet_kan_oneenv
#SBATCH --output=./job_logs/modnet_kan_oneenv_%j.out
#SBATCH --error=./job_logs/modnet_kan_oneenv_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=1000:00:00

set -euo pipefail

CONDA_PATH="${CONDA_PATH:-/home/wuhao/miniconda3/etc/profile.d/conda.sh}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ENV_NAME="${ENV_NAME:-modnet-kan}"
TASK_SET="${TASK_SET:-small}"
TASKS_OVERRIDE="${TASKS_OVERRIDE:-}"
RUN_MODE="${RUN_MODE:-full}"
RUN_ID="${RUN_ID:-${TASK_SET}-oneenv-$(date +%Y%m%d-%H%M%S)}"
OFFICIAL_N_JOBS="${OFFICIAL_N_JOBS:-4}"
OFFICIAL_RETRY_N_JOBS="${OFFICIAL_RETRY_N_JOBS:-1}"
OFFICIAL_MAX_TASK_ATTEMPTS="${OFFICIAL_MAX_TASK_ATTEMPTS:-2}"
OFFICIAL_TASK_TIMEOUT_MINUTES="${OFFICIAL_TASK_TIMEOUT_MINUTES:-1440}"
OFFICIAL_HEARTBEAT_SECONDS="${OFFICIAL_HEARTBEAT_SECONDS:-60}"
TRIAL_TIMEOUT_MINUTES="${TRIAL_TIMEOUT_MINUTES:-720}"
MAX_TRIALS_PER_FAMILY="${MAX_TRIALS_PER_FAMILY:-20}"
FIXED_N_FEATURES="${FIXED_N_FEATURES:-32}"
FIXED_KAN_TARGET_DIM="${FIXED_KAN_TARGET_DIM:-8}"
FIXED_KAN_GRID_SIZE="${FIXED_KAN_GRID_SIZE:-3}"
FIXED_KAN_SPLINE_ORDER="${FIXED_KAN_SPLINE_ORDER:-3}"
FIXED_EPOCHS="${FIXED_EPOCHS:-1000}"
FIXED_BATCH_SIZE="${FIXED_BATCH_SIZE:-64}"
FIXED_LR="${FIXED_LR:-0.001}"
FIXED_EARLY_STOPPING_PATIENCE="${FIXED_EARLY_STOPPING_PATIENCE:-100}"
FIXED_EARLY_STOPPING_MIN_DELTA="${FIXED_EARLY_STOPPING_MIN_DELTA:-0.001}"
FIXED_SEED="${FIXED_SEED:-7}"
FIXED_MIN_RELATIVE_IMPROVEMENT="${FIXED_MIN_RELATIVE_IMPROVEMENT:-0.02}"
FIXED_MIN_FOLD_WINS="${FIXED_MIN_FOLD_WINS:-3}"
FIXED_SPLINE_SYMBOLIC_INPUT_EDGES="${FIXED_SPLINE_SYMBOLIC_INPUT_EDGES:-5}"
FIXED_SPLINE_SYMBOLIC_OUTPUT_EDGES="${FIXED_SPLINE_SYMBOLIC_OUTPUT_EDGES:-4}"
FIXED_SPLINE_SYMBOLIC_FUNCTIONS="${FIXED_SPLINE_SYMBOLIC_FUNCTIONS:-identity square cube sin cos tan tanh exp log sqrt reciprocal abs arctan gaussian}"
FIXED_SYMBOLIC_HIDDEN_DIMS="${FIXED_SYMBOLIC_HIDDEN_DIMS:-4}"
FIXED_SYMBOLIC_EDGES_PER_UNIT="${FIXED_SYMBOLIC_EDGES_PER_UNIT:-3}"
FIXED_SYMBOLIC_PRIMITIVES="${FIXED_SYMBOLIC_PRIMITIVES:-zero one identity square cube sin cos tanh exp log1p_abs lorentz gaussian sinh cosh}"
FIXED_SYMBOLIC_HARDENING_EPOCHS="${FIXED_SYMBOLIC_HARDENING_EPOCHS:-100}"
FIXED_SYMBOLIC_HARDENING_LR="${FIXED_SYMBOLIC_HARDENING_LR:-0.0001}"
FIXED_SYMBOLIC_PROJECTION_TOP_K="${FIXED_SYMBOLIC_PROJECTION_TOP_K:-3}"
FIXED_MIN_FORMULA_FIDELITY_R2="${FIXED_MIN_FORMULA_FIDELITY_R2:-0.90}"
FIXED_MIN_FEATURE_JACCARD="${FIXED_MIN_FEATURE_JACCARD:-0.40}"
FIXED_MIN_OPERATOR_JACCARD="${FIXED_MIN_OPERATOR_JACCARD:-0.40}"
FIXED_MAX_IMPROVEMENT_STD="${FIXED_MAX_IMPROVEMENT_STD:-0.10}"
FIXED_MAX_TEACHER_DEGRADATION="${FIXED_MAX_TEACHER_DEGRADATION:-0.05}"
FIXED_MAX_FORMULA_DEGRADATION="${FIXED_MAX_FORMULA_DEGRADATION:-0.05}"

ALL_TASKS=(
  matbench_dielectric
  matbench_elastic
  matbench_expt_gap
  matbench_glass
  matbench_jdft2d
  matbench_mp_e_form
  matbench_mp_gap
  matbench_mp_is_metal
  matbench_perovskites
  matbench_phonons
  matbench_steels
)

SMALL_TASKS=(
  matbench_dielectric
  matbench_elastic
  matbench_expt_gap
  matbench_perovskites
  matbench_phonons
)

if [[ -n "$TASKS_OVERRIDE" ]]; then
  read -r -a TASKS <<< "$TASKS_OVERRIDE"
elif [[ "$TASK_SET" == "all" ]]; then
  TASKS=("${ALL_TASKS[@]}")
elif [[ "$TASK_SET" == "small" ]]; then
  TASKS=("${SMALL_TASKS[@]}")
else
  echo "TASK_SET must be all or small when TASKS_OVERRIDE is empty, got: ${TASK_SET}" >&2
  exit 2
fi

source "$CONDA_PATH"
cd "$PROJECT_DIR"
mkdir -p job_logs benchmarks

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

env_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "$1"
}

if ! env_exists "$ENV_NAME"; then
  bash scripts/setup_conda_modnet_kan.sh "$ENV_NAME"
fi

if [[ "$RUN_MODE" == "fixed5fold" ]]; then
  if [[ -z "${OFFICIAL_OUTPUT_DIR:-}" ]]; then
    echo "RUN_MODE=fixed5fold requires OFFICIAL_OUTPUT_DIR to point to completed official results." >&2
    exit 2
  fi
  KAN_OUTPUT_ROOT="${KAN_OUTPUT_ROOT:-benchmarks/interpretable-kan-5fold-${RUN_ID}}"
  read -r -a FIXED_SPLINE_FUNCTION_ARRAY <<< "$FIXED_SPLINE_SYMBOLIC_FUNCTIONS"
  read -r -a FIXED_SYMBOLIC_HIDDEN_ARRAY <<< "$FIXED_SYMBOLIC_HIDDEN_DIMS"
  read -r -a FIXED_SYMBOLIC_PRIMITIVE_ARRAY <<< "$FIXED_SYMBOLIC_PRIMITIVES"

  echo "Run mode: fixed5fold"
  echo "Tasks: ${TASKS[*]}"
  echo "Official results: ${OFFICIAL_OUTPUT_DIR}"
  echo "KAN outputs: ${KAN_OUTPUT_ROOT}"
  echo "Spline KAN: ${FIXED_N_FEATURES} descriptors -> ${FIXED_KAN_TARGET_DIM} -> output, grid=${FIXED_KAN_GRID_SIZE}, order=${FIXED_KAN_SPLINE_ORDER}"
  echo "Spline auto-symbolic library: ${FIXED_SPLINE_FUNCTION_ARRAY[*]}"
  echo "Paper Symbolic-KAN: ${FIXED_N_FEATURES} -> ${FIXED_SYMBOLIC_HIDDEN_ARRAY[*]} -> sum, edges/unit=${FIXED_SYMBOLIC_EDGES_PER_UNIT}"
  echo "Paper symbolic primitive library: ${FIXED_SYMBOLIC_PRIMITIVE_ARRAY[*]}"
  echo "Official fit rule: full outer train+validation, loss early stopping, min_delta=${FIXED_EARLY_STOPPING_MIN_DELTA}, patience=${FIXED_EARLY_STOPPING_PATIENCE}, restore_best_state=false"

  for task in "${TASKS[@]}"; do
    feature_dir="${OFFICIAL_OUTPUT_DIR}/${task}/official_feature_folds"
    official_fold_csv="${OFFICIAL_OUTPUT_DIR}/${task}/official-modnet-fold-results-${task}.csv"
    out_dir="${KAN_OUTPUT_ROOT}/${task}"
    spline_dir="${out_dir}/direct-spline"
    symbolic_dir="${out_dir}/symbolic-kan"

    if [[ ! -s "$official_fold_csv" ]]; then
      echo "Missing official fold results: ${official_fold_csv}" >&2
      exit 1
    fi
    for fold in 0 1 2 3 4; do
      metadata_path="${feature_dir}/fold_${fold}/metadata.json"
      if [[ ! -s "$metadata_path" ]]; then
        echo "Missing official feature fold: ${metadata_path}" >&2
        exit 1
      fi
    done
    mkdir -p "$spline_dir" "$symbolic_dir" \
      "$out_dir/compare-direct-spline" "$out_dir/compare-symbolic-kan"

    conda run --no-capture-output -n "$ENV_NAME" \
      python -u scripts/benchmark_modnet_kan.py \
      --dataset "$task" \
      --precomputed-feature-dir "$feature_dir" \
      --folds 0 1 2 3 4 \
      --models direct-spline \
      --n-features "$FIXED_N_FEATURES" \
      --kan-target-dims "$FIXED_KAN_TARGET_DIM" \
      --kan-grid-size "$FIXED_KAN_GRID_SIZE" \
      --kan-spline-order "$FIXED_KAN_SPLINE_ORDER" \
      --epochs "$FIXED_EPOCHS" \
      --batch-size "$FIXED_BATCH_SIZE" \
      --val-ratio 0 \
      --early-stopping-monitor loss \
      --early-stopping-patience "$FIXED_EARLY_STOPPING_PATIENCE" \
      --early-stopping-min-delta "$FIXED_EARLY_STOPPING_MIN_DELTA" \
      --no-restore-best-state \
      --lr "$FIXED_LR" \
      --weight-decay 0 \
      --loss auto \
      --activation elu \
      --scaler minmax \
      --target-scale none \
      --kan-l1-lambda 0 \
      --prune-kan-fraction 0 \
      --seed "$FIXED_SEED" \
      --device cuda \
      --require-cuda \
      --forward-iters 5 \
      --warmup-iters 1 \
      --log-every-epochs 20 \
      --no-kan-layernorm \
      --export-formulas \
      --formula-top-k 0 \
      --symbolify-spline-kan \
      --spline-symbolic-functions "${FIXED_SPLINE_FUNCTION_ARRAY[@]}" \
      --spline-symbolic-input-edges "$FIXED_SPLINE_SYMBOLIC_INPUT_EDGES" \
      --spline-symbolic-output-edges "$FIXED_SPLINE_SYMBOLIC_OUTPUT_EDGES" \
      --output-dir "$spline_dir"

    conda run --no-capture-output -n "$ENV_NAME" \
      python -u scripts/benchmark_modnet_kan.py \
      --dataset "$task" \
      --precomputed-feature-dir "$feature_dir" \
      --folds 0 1 2 3 4 \
      --models symbolic-kan \
      --n-features "$FIXED_N_FEATURES" \
      --symbolic-hidden-dims "${FIXED_SYMBOLIC_HIDDEN_ARRAY[@]}" \
      --symbolic-edges-per-unit "$FIXED_SYMBOLIC_EDGES_PER_UNIT" \
      --symbolic-primitives "${FIXED_SYMBOLIC_PRIMITIVE_ARRAY[@]}" \
      --symbolic-hardening-epochs "$FIXED_SYMBOLIC_HARDENING_EPOCHS" \
      --symbolic-hardening-lr "$FIXED_SYMBOLIC_HARDENING_LR" \
      --symbolic-projection-top-k "$FIXED_SYMBOLIC_PROJECTION_TOP_K" \
      --epochs "$FIXED_EPOCHS" \
      --batch-size "$FIXED_BATCH_SIZE" \
      --val-ratio 0 \
      --early-stopping-monitor loss \
      --early-stopping-patience "$FIXED_EARLY_STOPPING_PATIENCE" \
      --early-stopping-min-delta "$FIXED_EARLY_STOPPING_MIN_DELTA" \
      --no-restore-best-state \
      --lr "$FIXED_LR" \
      --weight-decay 0 \
      --loss rmse \
      --scaler minmax \
      --target-scale standard \
      --seed "$FIXED_SEED" \
      --device cuda \
      --require-cuda \
      --forward-iters 5 \
      --warmup-iters 1 \
      --log-every-epochs 20 \
      --output-dir "$symbolic_dir"

    conda run --no-capture-output -n "$ENV_NAME" \
      python -u scripts/compare_official_modnet_kan.py \
      --dataset "$task" \
      --official-fold-csv "$official_fold_csv" \
      --kan-output-dir "$spline_dir" \
      --model direct-spline \
      --comparison-target spline-symbolic \
      --expected-folds 5 \
      --min-relative-improvement "$FIXED_MIN_RELATIVE_IMPROVEMENT" \
      --min-fold-wins "$FIXED_MIN_FOLD_WINS" \
      --min-formula-fidelity-r2 "$FIXED_MIN_FORMULA_FIDELITY_R2" \
      --min-feature-jaccard "$FIXED_MIN_FEATURE_JACCARD" \
      --min-operator-jaccard "$FIXED_MIN_OPERATOR_JACCARD" \
      --max-improvement-std "$FIXED_MAX_IMPROVEMENT_STD" \
      --max-teacher-relative-degradation "$FIXED_MAX_TEACHER_DEGRADATION" \
      --max-formula-relative-degradation "$FIXED_MAX_FORMULA_DEGRADATION" \
      --output-dir "$out_dir/compare-direct-spline"

    conda run --no-capture-output -n "$ENV_NAME" \
      python -u scripts/compare_official_modnet_kan.py \
      --dataset "$task" \
      --official-fold-csv "$official_fold_csv" \
      --kan-output-dir "$symbolic_dir" \
      --model symbolic-kan \
      --comparison-target symbolic-kan \
      --expected-folds 5 \
      --min-relative-improvement "$FIXED_MIN_RELATIVE_IMPROVEMENT" \
      --min-fold-wins "$FIXED_MIN_FOLD_WINS" \
      --min-formula-fidelity-r2 "$FIXED_MIN_FORMULA_FIDELITY_R2" \
      --min-feature-jaccard "$FIXED_MIN_FEATURE_JACCARD" \
      --min-operator-jaccard "$FIXED_MIN_OPERATOR_JACCARD" \
      --max-improvement-std "$FIXED_MAX_IMPROVEMENT_STD" \
      --max-teacher-relative-degradation "$FIXED_MAX_TEACHER_DEGRADATION" \
      --max-formula-relative-degradation "$FIXED_MAX_FORMULA_DEGRADATION" \
      --output-dir "$out_dir/compare-symbolic-kan"

    conda run --no-capture-output -n "$ENV_NAME" \
      python -u scripts/summarize_interpretable_kan.py \
      --dataset "$task" \
      --spline-comparison-csv "$out_dir/compare-direct-spline/fixed5fold-comparison-${task}.csv" \
      --symbolic-comparison-csv "$out_dir/compare-symbolic-kan/fixed5fold-comparison-${task}.csv" \
      --output-dir "$out_dir"
  done

  echo "Unified conda env: ${ENV_NAME}"
  echo "Reused official MODNet outputs: ${OFFICIAL_OUTPUT_DIR}"
  echo "Spline KAN and paper Symbolic-KAN five-fold outputs: ${KAN_OUTPUT_ROOT}"
  exit 0
fi

if [[ "$RUN_MODE" != "full" ]]; then
  echo "RUN_MODE must be full or fixed5fold, got: ${RUN_MODE}" >&2
  exit 2
fi

OFFICIAL_OUTPUT_DIR="${OFFICIAL_OUTPUT_DIR:-benchmarks/official-modnet-v012-${RUN_ID}}"
TUNE_OUTPUT_ROOT="${TUNE_OUTPUT_ROOT:-benchmarks/tune-modnet-kan-${RUN_ID}}"

conda run --no-capture-output -n "$ENV_NAME" \
  python -u scripts/run_official_modnet_matbench.py \
  --tasks "${TASKS[@]}" \
  --n-jobs "$OFFICIAL_N_JOBS" \
  --retry-n-jobs "$OFFICIAL_RETRY_N_JOBS" \
  --max-task-attempts "$OFFICIAL_MAX_TASK_ATTEMPTS" \
  --task-timeout-minutes "$OFFICIAL_TASK_TIMEOUT_MINUTES" \
  --heartbeat-seconds "$OFFICIAL_HEARTBEAT_SECONDS" \
  --hp-strategy fit_preset \
  --random-state 7 \
  --nested-folds 5 \
  --n-models 5 \
  --skip-existing \
  --export-feature-folds \
  --export-max-features 512 \
  --output-dir "$OFFICIAL_OUTPUT_DIR"

for task in "${TASKS[@]}"; do
  feature_dir="${OFFICIAL_OUTPUT_DIR}/${task}/official_feature_folds"
  out_dir="${TUNE_OUTPUT_ROOT}/${task}"

  conda run --no-capture-output -n "$ENV_NAME" \
    python -u scripts/tune_modnet_kan.py \
    --dataset "$task" \
    --precomputed-feature-dir "$feature_dir" \
    --model-families mlp fastkan spline \
    --protocol matbench-nested \
    --inner-folds 5 \
    --final-folds 0 1 2 3 4 \
    --search-space compact \
    --strategy successive-halving \
    --halving-factor 3 \
    --rung-epochs 200 500 1000 \
    --rung-fold-counts 1 3 5 \
    --max-trials-per-family "$MAX_TRIALS_PER_FAMILY" \
    --metric auto \
    --n-feature-candidates 16 32 64 128 \
    --kan-grid-size-candidates 2 3 5 \
    --kan-spline-order-candidates 2 3 \
    --lr-candidates 0.001 0.005 0.01 \
    --weight-decay-candidates 0 \
    --dropout-candidates 0 \
    --tune-epochs 1000 \
    --final-epochs 1000 \
    --batch-size 64 \
    --val-ratio 0.1 \
    --early-stopping-patience 100 \
    --early-stopping-monitor loss \
    --early-stopping-min-delta 0.001 \
    --loss-candidates mae \
    --activation elu \
    --kan-l1-lambda 1e-6 \
    --kan-l1-lambda-candidates 0 1e-6 \
    --kan-sparsity-mode edge-group \
    --prune-kan-fraction-candidates 0 \
    --posthoc-prune-kan-fraction 0 \
    --posthoc-kan-sparsity-lambda 0 \
    --prune-mode edge \
    --prune-finetune-epochs 20 \
    --scaler minmax \
    --target-scale none \
    --impute-strategy median \
    --device cuda \
    --require-cuda \
    --log-every-epochs 0 \
    --trial-timeout-minutes "$TRIAL_TIMEOUT_MINUTES" \
    --resume \
    --formula-top-k 20 \
    --formula-min-abs 0 \
    --simple-formula-min-inputs 5 \
    --simple-formula-max-inputs 10 \
    --simple-formula-max-terms 10 \
    --simple-formula-coverage 0.95 \
    --simple-formula-calibration-ratio 0.1 \
    --output-dir "$out_dir"
done

echo "Unified conda env: ${ENV_NAME}"
echo "Official MODNet descriptor outputs: ${OFFICIAL_OUTPUT_DIR}"
echo "Tuning/final benchmark outputs: ${TUNE_OUTPUT_ROOT}"
