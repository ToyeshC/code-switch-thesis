#!/bin/bash

#SBATCH --partition=rome
#SBATCH --job-name=2_get_alignment_hi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=/home/tchakravorty/tchakravorty/code-switch-thesis/outputs/2_get_alignment_hi.out

module purge

module load 2024
module load Boost/1.85.0-GCC-13.3.0

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# Generate Alignment Files from gold translations
python ezswitch/alignment/giza-py/giza.py \
 --bin ezswitch/alignment/mgiza/mgizapp/bin \
 --source data/extracted_prompts/train_en.txt \
 --target data/extracted_prompts/train_hi.txt \
 --alignments data/alignments/en-hi_align_gold.txt

# Generate Alignment Files from silver translations
 python ezswitch/alignment/giza-py/giza.py \
 --bin ezswitch/alignment/mgiza/mgizapp/bin \
 --source data/extracted_prompts/train_en.txt \
 --target data/translate_api_outputs/train_hi.txt \
 --alignments data/alignments/en-hi_align_silver.txt