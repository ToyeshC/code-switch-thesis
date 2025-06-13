#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=tweets_perspective
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=tweets_job_outputs/03_perspective_%j.out

# Activate conda environment
source /home/tchakravorty/.bashrc
conda activate code-switch

# Install required packages
pip install google-api-python-client
pip install pandas tqdm

# Create output directories if they don't exist
mkdir -p tweets_outputs/perspective
mkdir -p tweets_job_outputs

# Set Google API key from config
export GOOGLE_API_KEY=$(python -c "import sys; sys.path.append('.'); import config; print(config.GOOGLE_API_KEY)")

# Define base directories
CS_CONT_BASE_DIR="tweets_outputs/llama_local"
CS_PERSP_OUTPUT_DIR="tweets_outputs/perspective"

# Process each model's continuations
# LLAMA3_CONT_FILE="$CS_CONT_BASE_DIR/code_switched_llama3.csv"
# if [ -f "$LLAMA3_CONT_FILE" ]; then
#     echo "Running Perspective API on Llama 3 continuations..."
#     python new_python_scripts/run_perspective.py \
#         --input_file "$LLAMA3_CONT_FILE" \
#         --output_file "$CS_PERSP_OUTPUT_DIR/llama3_continuations_perspective.csv" \
#         --text_column continuation \
#         --batch_size 100
# else
#     echo "Warning: Llama 3 continuations file not found, skipping..."
# fi

# LLAMA31_CONT_FILE="$CS_CONT_BASE_DIR/code_switched_llama31.csv"
# if [ -f "$LLAMA31_CONT_FILE" ]; then
#     echo "Running Perspective API on Llama 3.1 continuations..."
#     python new_python_scripts/run_perspective.py \
#         --input_file "$LLAMA31_CONT_FILE" \
#         --output_file "$CS_PERSP_OUTPUT_DIR/llama31_continuations_perspective.csv" \
#         --text_column continuation \
#         --batch_size 100
# else
#     echo "Warning: Llama 3.1 continuations file not found, skipping..."
# fi

AYA_CONT_FILE="$CS_CONT_BASE_DIR/code_switched_aya.csv"
if [ -f "$AYA_CONT_FILE" ]; then
    echo "Running Perspective API on Aya continuations..."
    python new_python_scripts/run_perspective.py \
        --input_file "$AYA_CONT_FILE" \
        --output_file "$CS_PERSP_OUTPUT_DIR/aya_continuations_perspective.csv" \
        --text_column continuation \
        --batch_size 100
else
    echo "Warning: Aya continuations file not found, skipping..."
fi

echo "Perspective API analysis complete!" 