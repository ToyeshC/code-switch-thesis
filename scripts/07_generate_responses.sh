#!/bin/bash

#SBATCH --time=8:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=gen_resp
#SBATCH --mem=32G
#SBATCH --output=outputs/07_generate_responses.out

# Load necessary modules
module load 2023
module load Miniconda3/23.5.2-0
source ~/.bashrc
conda activate code-switch

# Define languages - can be modified for other language pairs
BASE_LANG="hindi"
SOURCE_LANG="english"

# Set Hugging Face token
export HUGGING_FACE_HUB_TOKEN=$(cat ~/.huggingface/token)

# Define paths
INPUT_DIR="data/output/filtered"
OUTPUT_DIR="data/output/model_responses"

# Create output directory
mkdir -p $OUTPUT_DIR

# Generate LLaMA 3 (8B) responses for base language prompts
echo "Generating LLaMA 3 (8B) responses for ${BASE_LANG^} prompts..."
python src/generate_model_responses_with_id.py \
    --input $INPUT_DIR/${BASE_LANG}_filtered.csv \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --output $OUTPUT_DIR/llama3_8b_${BASE_LANG}_responses.csv \
    --max_tokens 100 \
    --temperature 0.7

# Generate LLaMA 3 (8B) responses for source language prompts
echo "Generating LLaMA 3 (8B) responses for ${SOURCE_LANG^} prompts..."
python src/generate_model_responses_with_id.py \
    --input $INPUT_DIR/${SOURCE_LANG}_filtered.csv \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --output $OUTPUT_DIR/llama3_8b_${SOURCE_LANG}_responses.csv \
    --max_tokens 100 \
    --temperature 0.7

# Generate Aya responses for base language prompts
echo "Generating Aya responses for ${BASE_LANG^} prompts..."
python src/generate_model_responses_with_id.py \
    --input $INPUT_DIR/${BASE_LANG}_filtered.csv \
    --model "CohereForAI/aya-23-8B" \
    --output $OUTPUT_DIR/aya_${BASE_LANG}_responses.csv \
    --max_tokens 100 \
    --temperature 0.7

# Generate Aya responses for source language prompts
echo "Generating Aya responses for ${SOURCE_LANG^} prompts..."
python src/generate_model_responses_with_id.py \
    --input $INPUT_DIR/${SOURCE_LANG}_filtered.csv \
    --model "CohereForAI/aya-23-8B" \
    --output $OUTPUT_DIR/aya_${SOURCE_LANG}_responses.csv \
    --max_tokens 100 \
    --temperature 0.7

echo "Response generation complete. Results saved to $OUTPUT_DIR"
