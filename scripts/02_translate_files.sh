#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --partition=rome
#SBATCH --job-name=translate
#SBATCH --mem=16G
#SBATCH --output=outputs/02_translate_files.out

# Load necessary modules
module load 2023
module load Miniconda3/23.5.2-0
source ~/.bashrc
conda activate code-switch

# Define languages - can be modified for other language pairs
BASE_LANG="hindi"
SOURCE_LANG="english"

# Define language codes for translation
BASE_LANG_CODE="hi"
SOURCE_LANG_CODE="en"

# Define paths
INPUT_DIR="data/extracted_prompts"
OUTPUT_DIR="data/translate_api_outputs"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Translating ${SOURCE_LANG} text to ${BASE_LANG}..."
python src/translate_file.py \
    --input $INPUT_DIR/train_${SOURCE_LANG}.txt \
    --target $BASE_LANG_CODE \
    --output $OUTPUT_DIR/train_${BASE_LANG}.txt

echo "Translating ${BASE_LANG} text to ${SOURCE_LANG}..."
python src/translate_file.py \
    --input $INPUT_DIR/train_${BASE_LANG}.txt \
    --target $SOURCE_LANG_CODE \
    --output $OUTPUT_DIR/train_${SOURCE_LANG}.txt

echo "Translation complete. Results saved to $OUTPUT_DIR" 