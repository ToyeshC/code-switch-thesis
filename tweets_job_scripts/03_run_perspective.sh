#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=tweets_perspective_small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=tweets_job_outputs/03_perspective_small_%j.out

# Activate conda environment
source /home/tchakravorty/.bashrc
conda activate code-switch

# Install required packages
pip install --quiet requests pandas tqdm

# Create output directory if it doesn't exist
mkdir -p tweets_outputs/perspective_small

# Get Perspective API key
PERSPECTIVE_API_KEY=$(python -c "import config; print(config.PERSPECTIVE_API_KEY)")

# Check if we have the Perspective API key
if [ -z "$PERSPECTIVE_API_KEY" ]; then
    echo "Error: PERSPECTIVE_API_KEY not found in config.py"
    exit 1
fi

# Define base paths for Perspective input/output
CS_CONT_BASE_DIR="tweets_outputs/llama_local_small"
CS_PERSP_OUTPUT_DIR="tweets_outputs/perspective_small"

# Run Perspective API on Llama 3 continuations
LLAMA3_CONT_FILE="$CS_CONT_BASE_DIR/code_switched_llama3_small.csv"
if [ -f "$LLAMA3_CONT_FILE" ]; then
    echo "Running Perspective API on Llama 3 continuations (small dataset)..."
    python new_python_scripts/run_perspective_api.py \
        --input_file "$LLAMA3_CONT_FILE" \
        --output_file "$CS_PERSP_OUTPUT_DIR/llama3_continuations_perspective_small.csv" \
        --api_key $PERSPECTIVE_API_KEY \
        --text_column llama3_continuation
else
    echo "Warning: Llama 3 continuation file not found ($LLAMA3_CONT_FILE), skipping Perspective API."
fi

# Run Perspective API on Llama 3.1 continuations
LLAMA31_CONT_FILE="$CS_CONT_BASE_DIR/code_switched_llama31_small.csv"
if [ -f "$LLAMA31_CONT_FILE" ]; then
    echo "Running Perspective API on Llama 3.1 continuations (small dataset)..."
    python new_python_scripts/run_perspective_api.py \
        --input_file "$LLAMA31_CONT_FILE" \
        --output_file "$CS_PERSP_OUTPUT_DIR/llama31_continuations_perspective_small.csv" \
        --api_key $PERSPECTIVE_API_KEY \
        --text_column llama31_continuation
else
    echo "Warning: Llama 3.1 continuation file not found ($LLAMA31_CONT_FILE), skipping Perspective API."
fi

# Run Perspective API on Aya continuations
AYA_CONT_FILE="$CS_CONT_BASE_DIR/code_switched_aya_small.csv"
if [ -f "$AYA_CONT_FILE" ]; then
    echo "Running Perspective API on Aya continuations (small dataset)..."
    python new_python_scripts/run_perspective_api.py \
        --input_file "$AYA_CONT_FILE" \
        --output_file "$CS_PERSP_OUTPUT_DIR/aya_continuations_perspective_small.csv" \
        --api_key $PERSPECTIVE_API_KEY \
        --text_column aya_continuation
else
    echo "Warning: Aya continuation file not found ($AYA_CONT_FILE), skipping Perspective API."
fi

echo "Perspective API analysis complete for small dataset!"
echo "Results can be found in: $CS_PERSP_OUTPUT_DIR/" 