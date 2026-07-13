#!/bin/bash
#SBATCH --job-name=modnet_kan_tune
#SBATCH --output=./job_logs/modnet_kan_tune_%j.out
#SBATCH --error=./job_logs/modnet_kan_tune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=1000:00:00

set -euo pipefail

# Edit these two paths if your server layout differs.
CONDA_PATH="${CONDA_PATH:-/home/wuhao/miniconda3/etc/profile.d/conda.sh}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"

OFFICIAL_ENV="${OFFICIAL_ENV:-modnet-v012-matbench}"
KAN_ENV="${KAN_ENV:-kan-cgcnn-cuda}"
TASK_SET="${TASK_SET:-small}"
RUN_ID="${RUN_ID:-${TASK_SET}-tune-$(date +%Y%m%d-%H%M%S)}"
OFFICIAL_N_JOBS="${OFFICIAL_N_JOBS:-4}"

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

if ! env_exists "$KAN_ENV"; then
  bash scripts/setup_conda_cuda.sh "$KAN_ENV"
fi

if ! env_exists "$OFFICIAL_ENV"; then
  conda env create -n "$OFFICIAL_ENV" -f environment-modnet-v012.yml
fi

OFFICIAL_OUTPUT_DIR="benchmarks/official-modnet-v012-${RUN_ID}"
TUNE_OUTPUT_ROOT="benchmarks/tune-modnet-kan-${RUN_ID}"

# Step 1: official MODNet descriptors and feature selection.
conda run --no-capture-output -n "$OFFICIAL_ENV" \
  python -u scripts/run_official_modnet_matbench.py \
  --tasks "${TASKS[@]}" \
  --n-jobs "$OFFICIAL_N_JOBS" \
  --hp-strategy fit_preset \
  --random-state 7 \
  --nested-folds 5 \
  --n-models 5 \
  --skip-existing \
  --export-feature-folds \
  --export-max-features 512 \
  --output-dir "$OFFICIAL_OUTPUT_DIR"

# Step 2: tune MLP, FastKAN, and Spline-KAN on the same official descriptors.
# Default tuner search is family-aware:
# - MLP tries wider blocks, e.g. 256/512 common dims.
# - KAN tries smaller blocks, e.g. 32/64/128 common dims.
# - target_dim=0 is included, so shallower formulas are tested.
for task in "${TASKS[@]}"; do
  feature_dir="${OFFICIAL_OUTPUT_DIR}/${task}/official_feature_folds"
  out_dir="${TUNE_OUTPUT_ROOT}/${task}"

  conda run --no-capture-output -n "$KAN_ENV" \
    python -u scripts/tune_modnet_kan.py \
    --dataset "$task" \
    --precomputed-feature-dir "$feature_dir" \
    --model-families mlp fastkan spline \
    --tune-folds 0 1 \
    --final-folds 0 1 2 3 4 \
    --search-space random \
    --num-random-trials 12 \
    --max-trials-per-family 12 \
    --metric auto \
    --tune-epochs 80 \
    --final-epochs 300 \
    --tune-train-size 1024 \
    --batch-size 64 \
    --val-ratio 0.1 \
    --early-stopping-patience 60 \
    --loss-candidates mae rmse \
    --prune-kan-fraction-candidates 0.3 0.5 \
    --scaler minmax \
    --target-scale none \
    --impute-strategy median \
    --device cuda \
    --require-cuda \
    --log-every-epochs 0 \
    --formula-top-k 20 \
    --formula-min-abs 0 \
    --output-dir "$out_dir"
done

echo "Official MODNet descriptor outputs: ${OFFICIAL_OUTPUT_DIR}"
echo "Tuning/final benchmark outputs: ${TUNE_OUTPUT_ROOT}"
