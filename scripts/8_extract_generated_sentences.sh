#!/bin/bash

#SBATCH --partition=rome
#SBATCH --job-name=8_extract_generated_sentences
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=3:00:00
#SBATCH --output=/home/tchakravorty/tchakravorty/code-switch-thesis/outputs/8_extract_generated_sentences.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

python src/extract_generated_sentences.py