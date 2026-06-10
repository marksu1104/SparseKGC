#!/bin/bash
# Submit all baseline jobs to HPC.
# Run this script from the SparseKGC/ directory.

set -e
cd "$(dirname "$0")"

mkdir -p outputs

echo "Submitting Traditional..."
sbatch exp_traditional.sh

echo "Submitting HoGRN..."
sbatch exp_hogrn.sh

echo "Submitting Prob-CBR..."
sbatch exp_probcbr.sh

echo "Submitting DacKGR (7-dataset array)..."
sbatch exp_dackgr.sh

echo "All jobs submitted. Check status with: squeue -u \$USER"
