#!/bin/bash
#SBATCH --job-name=modnet_kan_all
#SBATCH --output=./job_logs/modnet_kan_all_%j.out
#SBATCH --error=./job_logs/modnet_kan_all_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=1000:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK_SET="${TASK_SET:-all}"
exec bash "${SCRIPT_DIR}/slurm_modnet_kan_small.sh"
