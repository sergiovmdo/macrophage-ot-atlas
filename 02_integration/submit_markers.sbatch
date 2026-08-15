#!/bin/bash

#SBATCH --partition=epyc2

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16              # rank_genes_groups can use a few cores
#SBATCH --mem=256G                       # Needs to load the full object
#SBATCH --time=02:00:00                  # 2 hours

#SBATCH --job-name=find_markers
#SBATCH --output=find_markers.out
#SBATCH --error=find_markers.err

echo "Job started on $(hostname) at $(date)"

# Activate your specific Conda environment
eval "$(conda shell.bash hook)"
echo "Activating Conda environment: py311_r_env"
conda activate py311_r_env

echo "Python environment activated. Finding markers..."

# Run the Python script
python find_macrophage_markers.py

echo "Job finished at $(date)"