#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --job-name=19_run_perplexity_on_srctgt_full_gen
#SBATCH --mem=128G
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --output=job_outputs/19_run_perplexity_on_srctgt_full_gen_%j.out

# Load the required modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Activate your virtual environment if needed
source /home/tchakravorty/.bashrc
conda activate code-switch

pip install numpy pandas tqdm

# Run the script
python new_python_scripts/run_perspective_on_continuations.py