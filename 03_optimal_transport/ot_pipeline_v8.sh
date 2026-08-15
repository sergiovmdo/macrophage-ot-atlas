#!/bin/bash

#SBATCH --time=19:59:59
#SBATCH --mem-per-cpu=11G
#SBATCH --account=invest
#SBATCH --partition=gpu-invest
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --qos=job_gpu_igmp-tru
#SBATCH --job-name=ot_v8_B500
#SBATCH --array=1-4
#SBATCH --output=/storage/homefs/sv24v923/MPI_data/clean_pipeline/logs_v8/ot_v8_B500_%A_%a.log
#SBATCH --error=/storage/homefs/sv24v923/MPI_data/clean_pipeline/logs_v8/ot_v8_B500_%A_%a.err

PIPELINE_DIR="/storage/homefs/sv24v923/MPI_data/clean_pipeline"
PYTHON_SCRIPT="${PIPELINE_DIR}/ot_pipeline_v8.py"

echo "--- Starting SLURM Array Task ${SLURM_ARRAY_TASK_ID} ---"

case ${SLURM_ARRAY_TASK_ID} in
    1) MODE="full"         ;;
    2) MODE="loco_alsaigh" ;;
    3) MODE="loco_bashore" ;;
    4) MODE="loco_jaiswal" ;;
    *) echo "Error: unexpected task ID ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac

echo "Mode: ${MODE}"
mkdir -p "${PIPELINE_DIR}/logs_v8"

eval "$(conda shell.bash hook)"
conda activate py311_r_env
echo "Conda environment activated."

echo "Running OT pipeline v8 in mode: ${MODE}"
python $PYTHON_SCRIPT --mode "$MODE"

echo "Python script finished. Job complete."