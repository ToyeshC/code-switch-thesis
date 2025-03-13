#!/bin/bash

#SBATCH --partition=rome
#SBATCH --job-name=10_evaluation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=3:00:00
#SBATCH --output=/home/tchakravorty/tchakravorty/code-switch-thesis/outputs/10_evaluation.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

python src/evaluation_pipeline.py --input_file data/output/small_generated_sentences.csv --output data/output/small_evaluation_metrics.csv