#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=analyze
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=job_outputs/analyze_%j.out

source /home/tchakravorty/.bashrc
conda activate code-switch

python temp_scripts/analyze_toxicity_scores.py