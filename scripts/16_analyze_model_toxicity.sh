#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=16_analyze_model_toxicity
#SBATCH --mem=32G
#SBATCH --output=outputs/16_analyze_model_toxicity.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# Set Hugging Face token
export HUGGING_FACE_HUB_TOKEN=$(cat ~/.huggingface/token)

# Define input and output paths
INPUT_FILE="data/output/hindi/(yes) filtered_output_small.csv"
OUTPUT_DIR="data/output/model_toxicity_analysis"

# Run the analysis
echo "Starting analysis of model responses and toxicity..."
python src/analyze_model_toxicity.py \
    --input "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --llama_model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --aya_model "CohereForAI/aya-23-8B" \
    --max_tokens 100 \
    --temperature 0.7 \
    --skip_generation

echo "Analysis complete! Results saved to $OUTPUT_DIR" 