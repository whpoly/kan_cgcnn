#!/usr/bin/env bash
#SBATCH --job-name=modnet_kan
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --array=0-10%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

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
KAN_ENV="${KAN_ENV:-kan-cgcnn-cuda}"
OFFICIAL_OUTPUT_DIR="${OFFICIAL_OUTPUT_DIR:-${PROJECT_DIR}/benchmarks/official-modnet-v012-resumable}"
KAN_OUTPUT_ROOT="${KAN_OUTPUT_ROOT:-${PROJECT_DIR}/benchmarks/modnet-kan-resumable}"
TRIAL_TIMEOUT_MINUTES="${TRIAL_TIMEOUT_MINUTES:-180}"
MODEL_FAMILIES_STRING="${MODEL_FAMILIES:-mlp hybrid-fastkan hybrid-spline}"
read -r -a MODEL_FAMILIES_ARRAY <<< "$MODEL_FAMILIES_STRING"

source "$CONDA_PATH"
cd "$PROJECT_DIR"

FEATURE_DIR="${OFFICIAL_OUTPUT_DIR}/${TASK}/official_feature_folds"
TASK_OUTPUT_DIR="${KAN_OUTPUT_ROOT}/${TASK}"
for fold in 0 1 2 3 4; do
  metadata_path="${FEATURE_DIR}/fold_${fold}/metadata.json"
  if [[ ! -s "$metadata_path" ]]; then
    echo "Missing completed official feature fold: ${metadata_path}" >&2
    echo "Set OFFICIAL_OUTPUT_DIR to the directory containing your completed official run." >&2
    exit 1
  fi
done
mkdir -p "$TASK_OUTPUT_DIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "task=${TASK}"
echo "feature_dir=${FEATURE_DIR}"
echo "output_dir=${TASK_OUTPUT_DIR}"
echo "model_families=${MODEL_FAMILIES_ARRAY[*]}"

conda run --no-capture-output -n "$KAN_ENV" \
  python -u scripts/tune_modnet_kan.py \
  --dataset "$TASK" \
  --precomputed-feature-dir "$FEATURE_DIR" \
  --model-families "${MODEL_FAMILIES_ARRAY[@]}" \
  --tune-folds 0 1 \
  --final-folds 0 1 2 3 4 \
  --search-space random \
  --num-random-trials 8 \
  --max-trials-per-family 8 \
  --metric auto \
  --tune-epochs 80 \
  --final-epochs 300 \
  --tune-train-size 1024 \
  --batch-size 64 \
  --val-ratio 0.1 \
  --early-stopping-patience 60 \
  --loss-candidates mae rmse \
  --activation elu \
  --kan-l1-lambda 1e-5 \
  --prune-kan-fraction-candidates 0 0.3 0.5 \
  --prune-mode edge \
  --prune-finetune-epochs 20 \
  --scaler minmax \
  --target-scale none \
  --impute-strategy median \
  --device cuda \
  --require-cuda \
  --log-every-epochs 10 \
  --trial-timeout-minutes "$TRIAL_TIMEOUT_MINUTES" \
  --formula-top-k 20 \
  --formula-min-abs 0 \
  --resume \
  --output-dir "$TASK_OUTPUT_DIR"
