#!/bin/bash
# One-time HPC environment setup script.
# Run this ONCE after uploading the project to HPC.
#
# Usage: bash setup_hpc.sh
#
# What it does:
#   1. Creates a single conda env "sparsekgc" (Python 3.10)
#   2. Installs PyTorch 2.1 + CUDA 12.1
#   3. Installs torch-geometric and related packages
#   4. Installs remaining Python dependencies
#   5. Installs prob_cbr from this project's source

set -e

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ROOT="${HOME}/anaconda3"           # change to miniconda3 if needed
ENV_NAME="sparsekgc"
PROB_CBR_SRC="${SCRIPT_DIR}/baselines/Prob-CBR"
# ──────────────────────────────────────────────────────────────────────────

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

echo "=== Creating conda env: ${ENV_NAME} (Python 3.10) ==="
conda create -n "${ENV_NAME}" python=3.10 -y

conda activate "${ENV_NAME}"

echo "=== Installing PyTorch 2.1 (CUDA 12.1) ==="
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing torch-geometric ==="
pip install torch-geometric==2.6.1
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

echo "=== Installing other dependencies ==="
pip install \
    numpy==1.26.4 \
    scipy==1.15.3 \
    scikit-learn \
    tqdm \
    wandb \
    networkx \
    matplotlib \
    pandas \
    pyyaml

echo "=== Installing prob_cbr from source ==="
pip install -e "${PROB_CBR_SRC}"

echo "=== Verifying installation ==="
python -c "
import torch
import torch_geometric
import prob_cbr
print('torch:', torch.__version__)
print('torch_geometric:', torch_geometric.__version__)
print('prob_cbr: ok')
print('CUDA available:', torch.cuda.is_available())
"

echo ""
echo "=== Done! Update sbatch scripts: CONDA_ENV=\"${ENV_NAME}\" ==="
