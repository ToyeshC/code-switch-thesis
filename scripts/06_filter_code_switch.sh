#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --partition=rome
#SBATCH --job-name=filter_cs
#SBATCH --mem=16G
#SBATCH --output=outputs/06_filter_code_switch.out

# Load necessary modules
module load 2023
module load Miniconda3/23.5.2-0
source ~/.bashrc
conda activate code-switch

# Define languages - can be modified for other language pairs
BASE_LANG="hindi"
SOURCE_LANG="english"

# Define paths
INPUT_DIR="data/output/language_detection"
OUTPUT_DIR="data/output/filtered"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Filtering Aya model outputs for code-switched content..."
python src/filter_language_mix_with_id.py \
    --input $INPUT_DIR/aya_${BASE_LANG}_detection.csv \
    --output $OUTPUT_DIR/aya_${BASE_LANG}_filtered.csv \
    --min_hindi_percent 20 \
    --min_english_percent 20

echo "Filtering LLaMA model outputs for code-switched content..."
python src/filter_language_mix_with_id.py \
    --input $INPUT_DIR/llama_${BASE_LANG}_detection.csv \
    --output $OUTPUT_DIR/llama_${BASE_LANG}_filtered.csv \
    --min_hindi_percent 20 \
    --min_english_percent 20

# If you used Indic LID, you can also filter those results
echo "Filtering Indic LID detected Aya outputs..."
python src/filter_language_mix_with_id.py \
    --input $INPUT_DIR/aya_${BASE_LANG}_indic_detection.csv \
    --output $OUTPUT_DIR/aya_${BASE_LANG}_indic_filtered.csv \
    --min_hindi_prob 0.2 \
    --min_english_prob 0.2 \
    --use_indic_format

echo "Filtering Indic LID detected LLaMA outputs..."
python src/filter_language_mix_with_id.py \
    --input $INPUT_DIR/llama_${BASE_LANG}_indic_detection.csv \
    --output $OUTPUT_DIR/llama_${BASE_LANG}_indic_filtered.csv \
    --min_hindi_prob 0.2 \
    --min_english_prob 0.2 \
    --use_indic_format

echo "Filtering complete. Results saved to $OUTPUT_DIR"
