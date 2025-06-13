#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=21_add_perspective_generated
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=job_outputs/21_add_perspective_generated_%j.out

# Load the required modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Activate your virtual environment if needed
source /home/tchakravorty/.bashrc
conda activate code-switch

# pip install -r requirements.txt

# Run the script
python new_python_scripts/get_perspective_scores_full.py