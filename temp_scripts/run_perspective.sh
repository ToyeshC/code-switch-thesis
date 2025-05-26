#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=perspective
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=job_outputs/perspective_%j.out

source /home/tchakravorty/.bashrc
conda activate code-switch

python temp_scripts/analyze_perspective_metrics.py