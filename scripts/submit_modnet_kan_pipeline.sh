#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
OFFICIAL_OUTPUT_DIR="${OFFICIAL_OUTPUT_DIR:-${PROJECT_DIR}/benchmarks/official-modnet-v012-resumable}"
KAN_OUTPUT_ROOT="${KAN_OUTPUT_ROOT:-${PROJECT_DIR}/benchmarks/modnet-kan-resumable}"

official_job_id="$({ sbatch --parsable \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",OFFICIAL_OUTPUT_DIR="$OFFICIAL_OUTPUT_DIR" \
  scripts/slurm_official_modnet_array.sh; } | cut -d';' -f1)"

kan_job_id="$({ sbatch --parsable \
  --dependency="aftercorr:${official_job_id}" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",OFFICIAL_OUTPUT_DIR="$OFFICIAL_OUTPUT_DIR",KAN_OUTPUT_ROOT="$KAN_OUTPUT_ROOT" \
  scripts/slurm_modnet_kan_array.sh; } | cut -d';' -f1)"

echo "official_array_job_id=${official_job_id}"
echo "kan_array_job_id=${kan_job_id}"
echo "official_output_dir=${OFFICIAL_OUTPUT_DIR}"
echo "kan_output_root=${KAN_OUTPUT_ROOT}"
