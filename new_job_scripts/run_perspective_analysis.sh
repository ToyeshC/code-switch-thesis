#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=run_perspective_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=48:00:00
#SBATCH --output=job_outputs/run_perspective_analysis_%j.out

# Activate conda environment
source /home/tchakravorty/.bashrc
conda activate code-switch

# Install required packages
pip install --quiet pandas numpy requests tqdm

# Run the perspective analysis script
python new_python_scripts/run_perspective_on_continuations.py 