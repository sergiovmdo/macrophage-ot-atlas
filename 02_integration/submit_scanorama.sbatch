#!/bin/bash

#SBATCH --partition=epyc2
#SBATCH --job-name=scanorama_integration
#SBATCH --output=scanorama_integration_%j.out   # %j adds the Job ID to filename (prevents overwriting)
#SBATCH --error=scanorama_integration_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            
#SBATCH --mem=512G                   # 512GB is excellent for approx=False
#SBATCH --time=24:00:00              # 24h is likely enough, but if queue allows 24h, it's safer.

echo "Job started on $(hostname) at $(date)"

# --- 1. OPTIMIZATION: Prevent Python Output Buffering ---
# This ensures you see your "print()" statements in the .out file immediately
export PYTHONUNBUFFERED=1

# --- 2. OPTIMIZATION: Bind Math Libraries to CPUs ---
# Scanorama uses NumPy/Scikit-learn. We must tell them to use the 8 cores we requested.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load Conda
# (Ensure this path is correct for your specific cluster config)
eval "$(conda shell.bash hook)"

echo "Activating Conda environment: py311_r_env"
conda activate py311_r_env

echo "Python environment activated."
echo "Resources: $SLURM_CPUS_PER_TASK CPUs, $SLURM_MEM_PER_NODE memory."

# Run the Python script
# Make sure your python file is actually named 'run_scanorama.py'
python run_scanorama_v2.py

echo "Job finished at $(date)"