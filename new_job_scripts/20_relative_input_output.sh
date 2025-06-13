#!/bin/bash

#SBATCH --partition=rome
#SBATCH --job-name=20_relative_input_output
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=job_outputs/20_relative_input_output_%j.out

# Load the required modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Activate your virtual environment if needed
source /home/tchakravorty/.bashrc
conda activate code-switch

# Run the script
python new_python_scripts/analyze_toxicity_differences.py