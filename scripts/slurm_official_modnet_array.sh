#!/usr/bin/env bash
#SBATCH --job-name=official_modnet
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --array=0-10%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00

set -euo pipefail

TASKS=(
  matbench_dielectric
  matbench_expt_gap
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

TASK_INDEX="${SLURM_ARRAY_TASK_ID:?This script must be submitted as a Slurm array}"
if (( TASK_INDEX < 0 || TASK_INDEX >= ${#TASKS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_INDEX}" >&2
  exit 2
fi
TASK="${TASKS[$TASK_INDEX]}"

CONDA_PATH="${CONDA_PATH:-/home/wuhao/miniconda3/etc/profile.d/conda.sh}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
OFFICIAL_ENV="${OFFICIAL_ENV:-modnet-v012-matbench}"
OFFICIAL_OUTPUT_DIR="${OFFICIAL_OUTPUT_DIR:-${PROJECT_DIR}/benchmarks/official-modnet-v012-resumable}"
OFFICIAL_N_JOBS="${OFFICIAL_N_JOBS:-4}"
# Two 22-hour attempts fit within the 48-hour allocation; override for clusters
# where the large MP tasks need a longer single attempt.
TASK_TIMEOUT_MINUTES="${TASK_TIMEOUT_MINUTES:-1320}"

source "$CONDA_PATH"
cd "$PROJECT_DIR"
mkdir -p "$OFFICIAL_OUTPUT_DIR"

export PYTHONUNBUFFERED=1
export MODNET_THREADS_PER_PROCESS=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

echo "task=${TASK}"
echo "official_output_dir=${OFFICIAL_OUTPUT_DIR}"

conda run --no-capture-output -n "$OFFICIAL_ENV" \
  python -u scripts/run_official_modnet_matbench.py \
  --tasks "$TASK" \
  --output-dir "$OFFICIAL_OUTPUT_DIR" \
  --n-jobs "$OFFICIAL_N_JOBS" \
  --retry-n-jobs 1 \
  --max-task-attempts 2 \
  --task-timeout-minutes "$TASK_TIMEOUT_MINUTES" \
  --heartbeat-seconds 60 \
  --hp-strategy fit_preset \
  --random-state 7 \
  --nested-folds 5 \
  --n-models 5 \
  --save-folds \
  --skip-existing \
  --export-feature-folds \
  --export-max-features 512
