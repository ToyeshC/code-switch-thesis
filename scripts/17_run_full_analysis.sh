#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=17_run_full_analysis
#SBATCH --mem=32G
#SBATCH --output=outputs/17_run_full_analysis.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# Set environment variables
export HUGGING_FACE_HUB_TOKEN=$(cat ~/.huggingface/token)

# Define paths
PROMPT_FILE="data/output/hindi/(yes) filtered_output_small.csv"
LLAMA_RESPONSE_FILE="data/output/full_text_llama.csv" # Path to your existing LLaMA output
AYA_RESPONSE_FILE="data/output/full_text_aya.csv"     # Path to your existing Aya output
OUTPUT_DIR="data/output/model_toxicity_analysis"
API_KEY="AIzaSyDf0c2MkAItSv7TBFps65WavRFLP-N275Y"  # The API key from config.py

echo "==== Step 1: Prepare Model Outputs ===="
python src/prepare_model_outputs.py \
    --input "$PROMPT_FILE" \
    --llama_output "$LLAMA_RESPONSE_FILE" \
    --aya_output "$AYA_RESPONSE_FILE" \
    --output_dir "$OUTPUT_DIR"

echo "==== Step 2: Run Toxicity Analysis ===="
python src/analyze_model_toxicity.py \
    --input "$PROMPT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --skip_generation \
    --api_key "$API_KEY"

echo "Analysis complete! Results saved to $OUTPUT_DIR/analysis" 