#!/bin/bash
#SBATCH --job-name=trad
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=outputs/traditional_%j.log
#SBATCH --account=<FILL_IN_ACCOUNT>
#SBATCH --partition=<FILL_IN_PARTITION>

# ── User-specific settings ─────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ROOT="${HOME}/anaconda3"          # change if miniconda or different path
CONDA_ENV="sparsekgc"
GPU=0
# ──────────────────────────────────────────────────────────────────────────

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export SPARSEKGC_OUTPUT_DIR="${PROJECT_DIR}/outputs"
export SPARSEKGC_DATA_DIR="${PROJECT_DIR}/datasets"
mkdir -p "${SPARSEKGC_OUTPUT_DIR}"

cd "${PROJECT_DIR}"
python run_baseline.py traditional --gpu "${GPU}"
