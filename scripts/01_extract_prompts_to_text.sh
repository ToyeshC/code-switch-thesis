#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --partition=rome
#SBATCH --job-name=extract_txt
#SBATCH --mem=16G
#SBATCH --output=outputs/01_extract_prompts_to_text.out

# Load necessary modules
module load 2023
module load Miniconda3/23.5.2-0
source ~/.bashrc
conda activate code-switch

# Define languages - can be modified for other language pairs
BASE_LANG="hindi"
SOURCE_LANG="english"

# Define paths
INPUT_DIR="data/output"
OUTPUT_DIR="data/extracted_prompts"
ID_MAP_DIR="data/id_mappings"

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $ID_MAP_DIR

echo "Extracting prompts to text files for translation and alignment..."
python src/extract_prompts_to_text.py \
    --input_csv $INPUT_DIR/${BASE_LANG}_prompts_with_id.csv \
    --output_txt $OUTPUT_DIR/train_${BASE_LANG}.txt \
    --id_map $ID_MAP_DIR/${BASE_LANG}_id_map.json

python src/extract_prompts_to_text.py \
    --input_csv $INPUT_DIR/${SOURCE_LANG}_prompts_with_id.csv \
    --output_txt $OUTPUT_DIR/train_${SOURCE_LANG}.txt \
    --id_map $ID_MAP_DIR/${SOURCE_LANG}_id_map.json

echo "Extraction complete. Text files saved to $OUTPUT_DIR"
echo "ID mappings saved to $ID_MAP_DIR" 