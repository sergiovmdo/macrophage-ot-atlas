#!/bin/bash

#SBATCH --partition=epyc2
#SBATCH --job-name=clustree_analysis
#SBATCH --output=clustree_%j.out
#SBATCH --error=clustree_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16           # Increased to 16 to speed up Neighbor calculation
#SBATCH --mem=256G                   # 256GB is plenty for clustering 400k cells
#SBATCH --time=10:00:00              # This job should finish in <1 hour, but 6h is safe

echo "Job started on $(hostname) at $(date)"

# --- 1. Environment Setup ---
export PYTHONUNBUFFERED=1

# Bind math libraries to the CPUs we requested to speed up Leiden/Neighbors
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Activate Conda
# (Ensure this path matches your system setup)
eval "$(conda shell.bash hook)"

echo "Activating Conda environment: py311_r_env"
conda activate py311_r_env

# --- 2. Run the Script ---
echo "Running Clustree Analysis..."
python run_clustree.py

echo "Job finished at $(date)"