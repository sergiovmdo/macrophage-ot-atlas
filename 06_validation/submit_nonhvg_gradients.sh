#!/bin/bash
#SBATCH --partition=epyc2
#SBATCH --job-name=non-hvg-anal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=05:00:00
#SBATCH --output=logs/non-hvg-anal_%j.log
#SBATCH --error=logs/non-hvg-anal_%j.err

mkdir -p logs
eval "$(conda shell.bash hook)"
conda activate py311_r_env

python /storage/homefs/sv24v923/MPI_data/clean_pipeline/run_nonHVG_SRC_all.py
