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
RUN_ID="${RUN_ID:-${TASK_SET}-oneenv-$(date +%Y%m%d-%H%M%S)}"
OFFICIAL_N_JOBS="${OFFICIAL_N_JOBS:-4}"
OFFICIAL_RETRY_N_JOBS="${OFFICIAL_RETRY_N_JOBS:-1}"
OFFICIAL_MAX_TASK_ATTEMPTS="${OFFICIAL_MAX_TASK_ATTEMPTS:-2}"
OFFICIAL_TASK_TIMEOUT_MINUTES="${OFFICIAL_TASK_TIMEOUT_MINUTES:-1440}"
OFFICIAL_HEARTBEAT_SECONDS="${OFFICIAL_HEARTBEAT_SECONDS:-60}"
TRIAL_TIMEOUT_MINUTES="${TRIAL_TIMEOUT_MINUTES:-180}"

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
  matbench_glass
  matbench_jdft2d
  matbench_perovskites
  matbench_phonons
  matbench_steels
)

if [[ "$TASK_SET" == "all" ]]; then
  TASKS=("${ALL_TASKS[@]}")
elif [[ "$TASK_SET" == "small" ]]; then
  TASKS=("${SMALL_TASKS[@]}")
else
  echo "TASK_SET must be all or small, got: ${TASK_SET}" >&2
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

OFFICIAL_OUTPUT_DIR="benchmarks/official-modnet-v012-${RUN_ID}"
TUNE_OUTPUT_ROOT="benchmarks/tune-modnet-kan-${RUN_ID}"

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
    --model-families mlp hybrid-fastkan hybrid-spline \
    --protocol matbench-nested \
    --inner-folds 5 \
    --final-folds 0 1 2 3 4 \
    --search-space random \
    --num-random-trials 12 \
    --max-trials-per-family 12 \
    --metric auto \
    --tune-epochs 80 \
    --final-epochs 300 \
    --batch-size 64 \
    --val-ratio 0.1 \
    --early-stopping-patience 60 \
    --loss-candidates mae rmse \
    --activation elu \
    --kan-l1-lambda 0 \
    --prune-kan-fraction-candidates 0 \
    --posthoc-prune-kan-fraction 0.3 \
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
