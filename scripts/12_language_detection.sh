#!/bin/bash

#SBATCH --partition=rome
#SBATCH --job-name=12_language_detection
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=3:00:00
#SBATCH --output=/home/tchakravorty/tchakravorty/code-switch-thesis/outputs/12_language_detection.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

python src/language_detection.py --input_file data/output/small_generated_sentences.csv --output_file data/output/language_detection_small.csv