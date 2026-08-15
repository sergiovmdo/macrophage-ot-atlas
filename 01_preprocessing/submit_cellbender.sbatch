#!/bin/bash

# --- 1. SBATCH Resource Configuration ---
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1      # <-- I SET THIS TO 1 (see note below)
#SBATCH --cpus-per-task=4
#SBATCH --mem=59G
#SBATCH --time=24:00:00           # 24 hours for the full loop
#SBATCH --job-name=cellbender_all
#SBATCH --output=cellbender_all_%j.out  # %j = Job ID
#SBATCH --error=cellbender_all_%j.err

# --- 2. SETUP THE ENVIRONMENT ---
echo "Job started on $(hostname)"
echo "Loading Conda environment: cellbender"
date

eval "$(conda shell.bash hook)"
conda activate cellbender

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'cellbender'"
    exit 1
fi
echo "Environment activated successfully."

# --- 3. RUN THE PYTHON SCRIPT ---
echo "--- Starting the main Python script (run_cellbender.py) ---"

python run_cellbender.py

echo "--- Python script finished ---"
date