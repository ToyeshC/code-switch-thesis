#!/bin/bash

#SBATCH --time=6:00:00
#SBATCH --partition=rome
#SBATCH --job-name=analyze_tox
#SBATCH --mem=16G
#SBATCH --output=outputs/08_analyze_toxicity.out

# Load necessary modules
module load 2023
module load Miniconda3/23.5.2-0
source ~/.bashrc
conda activate code-switch

# Define languages - can be modified for other language pairs
BASE_LANG="hindi"
SOURCE_LANG="english"

# Get Perspective API key from config
API_KEY=$(grep -oP "PERSPECTIVE_API_KEY = \"\K[^\"]+" src/config.py)
echo "Retrieved Perspective API key from src/config.py"

# Define paths
INPUT_DIR="data/output/filtered"
RESPONSES_DIR="data/output/model_responses"
OUTPUT_DIR="data/output/toxicity_analysis"

# Create output directory
mkdir -p $OUTPUT_DIR

# Analyze toxicity of base language prompts
echo "Analyzing toxicity of ${BASE_LANG^} prompts..."
python src/analyze_toxicity_with_id.py \
    --input $INPUT_DIR/${BASE_LANG}_filtered.csv \
    --output $OUTPUT_DIR/${BASE_LANG}_prompt_toxicity.csv \
    --progress_file $OUTPUT_DIR/${BASE_LANG}_prompt_toxicity_progress.csv \
    --api_key "$API_KEY" \
    --batch_size 5

# Analyze toxicity of source language prompts
echo "Analyzing toxicity of ${SOURCE_LANG^} prompts..."
python src/analyze_toxicity_with_id.py \
    --input $INPUT_DIR/${SOURCE_LANG}_filtered.csv \
    --output $OUTPUT_DIR/${SOURCE_LANG}_prompt_toxicity.csv \
    --progress_file $OUTPUT_DIR/${SOURCE_LANG}_prompt_toxicity_progress.csv \
    --api_key "$API_KEY" \
    --batch_size 5

# Analyze toxicity of model responses
echo "Analyzing toxicity of LLaMA responses for ${BASE_LANG^} prompts..."
python src/analyze_toxicity_with_id.py \
    --input $RESPONSES_DIR/llama3_8b_${BASE_LANG}_responses.csv \
    --output $OUTPUT_DIR/llama3_8b_${BASE_LANG}_toxicity.csv \
    --progress_file $OUTPUT_DIR/llama3_8b_${BASE_LANG}_toxicity_progress.csv \
    --api_key "$API_KEY" \
    --batch_size 5

echo "Analyzing toxicity of LLaMA responses for ${SOURCE_LANG^} prompts..."
python src/analyze_toxicity_with_id.py \
    --input $RESPONSES_DIR/llama3_8b_${SOURCE_LANG}_responses.csv \
    --output $OUTPUT_DIR/llama3_8b_${SOURCE_LANG}_toxicity.csv \
    --progress_file $OUTPUT_DIR/llama3_8b_${SOURCE_LANG}_toxicity_progress.csv \
    --api_key "$API_KEY" \
    --batch_size 5

echo "Analyzing toxicity of Aya responses for ${BASE_LANG^} prompts..."
python src/analyze_toxicity_with_id.py \
    --input $RESPONSES_DIR/aya_${BASE_LANG}_responses.csv \
    --output $OUTPUT_DIR/aya_${BASE_LANG}_toxicity.csv \
    --progress_file $OUTPUT_DIR/aya_${BASE_LANG}_toxicity_progress.csv \
    --api_key "$API_KEY" \
    --batch_size 5

echo "Analyzing toxicity of Aya responses for ${SOURCE_LANG^} prompts..."
python src/analyze_toxicity_with_id.py \
    --input $RESPONSES_DIR/aya_${SOURCE_LANG}_responses.csv \
    --output $OUTPUT_DIR/aya_${SOURCE_LANG}_toxicity.csv \
    --progress_file $OUTPUT_DIR/aya_${SOURCE_LANG}_toxicity_progress.csv \
    --api_key "$API_KEY" \
    --batch_size 5

echo "Toxicity analysis complete. Results saved to $OUTPUT_DIR"
