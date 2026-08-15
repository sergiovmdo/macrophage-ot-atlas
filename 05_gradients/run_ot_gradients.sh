#!/bin/bash

#SBATCH --partition=gpu-invest
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --job-name=ot_gradient
#SBATCH --nodes=1
#SBATCH --array=1-1
#SBATCH --output=/storage/homefs/sv24v923/MPI_data/clean_pipeline/logs_v6/ot_v6_%A_%a.log
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G

PIPELINE_DIR="/storage/homefs/sv24v923/MPI_data/clean_pipeline"

mkdir -p "${PIPELINE_DIR}/logs"
eval "$(conda shell.bash hook)"
conda activate py311_r_env

python ${PIPELINE_DIR}/OT_gradients.py    