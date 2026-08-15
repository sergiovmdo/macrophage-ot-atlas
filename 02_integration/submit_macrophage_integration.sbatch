#!/bin/bash

#SBATCH --partition=epyc2
#SBATCH --job-name=macro_filter_integ
#SBATCH --output=macro_filter_%j.out
#SBATCH --error=macro_filter_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16           
#SBATCH --mem=256G                   
#SBATCH --time=24:00:00              

echo "Job started on $(hostname) at $(date)"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate py311_r_env

echo "Running Macrophage Filter & Re-Integration..."
python filter_and_reintegrate_macrophages.py

echo "Job finished at $(date)"