#!/bin/bash
# -----------------------------------------------------------------------------
# SBATCH options
#=================================================================
#SBATCH --partition=gpu-invest
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --qos=job_gpu_igmp-tru
#SBATCH --cpus-per-task=4 
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --job-name=sc_prep
#SBATCH --output=pipeline_%j.out
#SBATCH --error=pipeline_%j.err

#=================================================================
# INPUT ARGUMENT CHECK
#=================================================================
MODE=$1
DATASET_NAME=$2
OPTIONAL_FLAG=$3  # Capture the third argument (if it exists)

#=================================================================
# ENVIRONMENT SETUP
#=================================================================
echo "Job started on $(hostname) at $(date)"
echo "Mode: $MODE"
echo "Dataset: $DATASET_NAME"
echo "Extra Flags: $OPTIONAL_FLAG"

# Load correct CUDA modules for RTX4090 (Restored from your original)
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

# Activate Environment
echo "Activating conda environment: sc_pipeline..."
eval "$(conda shell.bash hook)"
conda activate sc_pipeline

#=================================================================
# RUN THE PIPELINE
#=================================================================
echo "--- Starting Python Pipeline ---"

# Pass arguments to Python.
# If $OPTIONAL_FLAG is empty, Python simply ignores it.
python run_preprocessing_pipeline.py "$MODE" "$DATASET_NAME" $OPTIONAL_FLAG

echo "--- Pipeline Finished at $(date) ---"