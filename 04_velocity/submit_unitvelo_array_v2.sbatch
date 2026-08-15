#!/bin/bash

#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --account=gratis
#SBATCH --partition=gpu-invest
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --job-name=unitvelo_v3
#SBATCH --array=1-5
#SBATCH --output=/storage/homefs/sv24v923/MPI_data/clean_pipeline/logs_v8/unitvelo_%A_%a.log
#SBATCH --error=/storage/homefs/sv24v923/MPI_data/clean_pipeline/logs_v8/unitvelo_%A_%a.err

PIPELINE_DIR="/storage/homefs/sv24v923/MPI_data/clean_pipeline"
PYTHON_SCRIPT="${PIPELINE_DIR}/run_unitvelo_pairwise_array_v4.py"
CONNECTIONS_FILE="${PIPELINE_DIR}/connections.txt"

echo "--- Starting SLURM Array Task ${SLURM_ARRAY_TASK_ID} ---"

if [ ! -f "$CONNECTIONS_FILE" ]; then
    echo "Error: connections_v3.txt not found at $CONNECTIONS_FILE"
    exit 1
fi

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $CONNECTIONS_FILE)
TYPE_A=$(echo "$LINE" | cut -d'"' -f2)
TYPE_B=$(echo "$LINE" | cut -d'"' -f4)

if [ -z "$TYPE_A" ] || [ -z "$TYPE_B" ]; then
    echo "Error: Could not parse line ${SLURM_ARRAY_TASK_ID}: $LINE"
    exit 1
fi

echo "Parameters found: TYPE_A='${TYPE_A}', TYPE_B='${TYPE_B}'"

mkdir -p "${PIPELINE_DIR}/logs_v8"

eval "$(conda shell.bash hook)"
conda activate unitvelo
echo "Conda environment activated."

echo "Running Python script..."
python $PYTHON_SCRIPT --type_a "$TYPE_A" --type_b "$TYPE_B"

echo "Python script finished. Job complete."