#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --partition=rome
#SBATCH --job-name=alignments
#SBATCH --mem=16G
#SBATCH --output=outputs/03_generate_alignments.out

# Load necessary modules
module purge
module load 2024
module load Boost/1.85.0-GCC-13.3.0

module load 2023
module load Miniconda3/23.5.2-0
source ~/.bashrc
conda activate code-switch

# Define languages - can be modified for other language pairs
BASE_LANG="hindi"
SOURCE_LANG="english"

# Define language codes for alignments
BASE_LANG_CODE="hi"
SOURCE_LANG_CODE="en"

# Define paths
INPUT_DIR="data/extracted_prompts"
TRANSLATE_DIR="data/translate_api_outputs"
OUTPUT_DIR="data/alignments"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Generating alignments for gold translations (original parallel texts)..."
python ezswitch/alignment/giza-py/giza.py \
    --bin ezswitch/alignment/mgiza/mgizapp/bin \
    --source $INPUT_DIR/train_${SOURCE_LANG}.txt \
    --target $INPUT_DIR/train_${BASE_LANG}.txt \
    --alignments $OUTPUT_DIR/${SOURCE_LANG_CODE}-${BASE_LANG_CODE}_align_gold.txt

echo "Generating alignments for silver translations (machine translations)..."
python ezswitch/alignment/giza-py/giza.py \
    --bin ezswitch/alignment/mgiza/mgizapp/bin \
    --source $INPUT_DIR/train_${SOURCE_LANG}.txt \
    --target $TRANSLATE_DIR/train_${BASE_LANG}.txt \
    --alignments $OUTPUT_DIR/${SOURCE_LANG_CODE}-${BASE_LANG_CODE}_align_silver.txt

echo "Alignment generation complete. Results saved to $OUTPUT_DIR" 