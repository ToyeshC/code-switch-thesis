#!/bin/bash

#SBATCH --partition=rome
#SBATCH --job-name=13_filter_language_mix
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=3:00:00
#SBATCH --output=/home/tchakravorty/tchakravorty/code-switch-thesis/outputs/13_filter_language_mix.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

python src/filter_language_mix.py --input data/output/language_detection_full.csv --output data/output/filtered_output.csv