#!/bin/bash
#SBATCH --job-name=probcbr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=outputs/probcbr_%j.log
#SBATCH --account=<FILL_IN_ACCOUNT>
#SBATCH --partition=<FILL_IN_PARTITION>

# ── User-specific settings ─────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ROOT="${HOME}/anaconda3"
CONDA_ENV="sparsekgc"
# ──────────────────────────────────────────────────────────────────────────

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export SPARSEKGC_OUTPUT_DIR="${PROJECT_DIR}/outputs"
export SPARSEKGC_DATA_DIR="${PROJECT_DIR}/datasets"
mkdir -p "${SPARSEKGC_OUTPUT_DIR}"

cd "${PROJECT_DIR}"
python run_baseline.py probcbr
