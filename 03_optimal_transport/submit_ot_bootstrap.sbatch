#!/bin/bash
#SBATCH --job-name=boot_sensitivity
#SBATCH --account=gratis
#SBATCH --partition=cpu-invest
#SBATCH --qos=job_cpu_preemptable
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=boot_sensitivity_%j.log
#SBATCH --error=boot_sensitivity_%j.err

eval "$(conda shell.bash hook)"
conda activate py311_r_env
echo "Conda environment activated."

python /storage/homefs/sv24v923/MPI_data/clean_pipeline/ot_boostrap_vallidation_final.py