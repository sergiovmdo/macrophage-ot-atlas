#!/bin/bash
#SBATCH --job-name=velocity_perm
#SBATCH --time=05:30:00
#SBATCH --nodes=1
#SBATCH --account=gratis
#SBATCH --partition=cpu-invest
#SBATCH --qos=job_cpu_preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G
#SBATCH --output=/storage/homefs/sv24v923/MPI_data/clean_pipeline/logs/velocity_perm_%j.log
#SBATCH --error=/storage/homefs/sv24v923/MPI_data/clean_pipeline/logs/velocity_perm_%j.err
#SBATCH --chdir=/storage/homefs/sv24v923/MPI_data/clean_pipeline

eval "$(conda shell.bash hook)"
conda activate py311_r_env

python -u run_velocity_permutation.py