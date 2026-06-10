#!/bin/bash
#SBATCH --job-name=dackgr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-6              # 7 datasets in parallel, each gets 1 GPU
#SBATCH --output=outputs/dackgr_%A_%a.log
#SBATCH --account=<FILL_IN_ACCOUNT>
#SBATCH --partition=<FILL_IN_PARTITION>

# ── User-specific settings ─────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ROOT="${HOME}/anaconda3"
CONDA_ENV="sparsekgc"
GPU=0
# ──────────────────────────────────────────────────────────────────────────

DATASETS=(
    "WD-singer"
    "FB15K-237-10"
    "WN18RR"
    "FB15K-237-20"
    "FB15K-237-50"
    "NELL23K"
    "FB15K-237"
)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export SPARSEKGC_OUTPUT_DIR="${PROJECT_DIR}/outputs"
export SPARSEKGC_DATA_DIR="${PROJECT_DIR}/datasets"
mkdir -p "${SPARSEKGC_OUTPUT_DIR}"

echo "Array task ${SLURM_ARRAY_TASK_ID}: running DacKGR on ${DATASET}"

cd "${PROJECT_DIR}"
python run_baseline.py dackgr --datasets "${DATASET}" --gpu "${GPU}"
