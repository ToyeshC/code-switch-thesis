#!/bin/bash

#SBATCH --partition=rome
#SBATCH --job-name=15_get_generation_and_toxicity_and_histogram_aya
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=15:00:00
#SBATCH --output=/home/tchakravorty/tchakravorty/code-switch-thesis/outputs/15_get_generation_and_toxicity_and_histogram_aya.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

python src/get_full_text.py --input data/output/continued_output_aya.csv --output data/output/full_text_aya.csv
python src/analyse_comments.py --input data/output/full_text_aya.csv --output data/output/perspective_analysis_aya.csv
python src/analyze_toxicity.py --input data/output/perspective_analysis_aya.csv --output_dir data/output/aya