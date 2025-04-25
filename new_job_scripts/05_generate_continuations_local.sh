#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=05_gen_cs_local_full # Updated job name
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8 # Increased CPUs for potentially heavy loading
#SBATCH --gpus-per-node=1 
#SBATCH --mem=64G # Request more RAM if needed
#SBATCH --time=48:00:00 # Increased time for full dataset
#SBATCH --output=job_outputs/05_gen_cs_local_full_%j.out # Updated output filename

# Activate conda environment
source /home/tchakravorty/.bashrc
conda activate code-switch

# Optional Cache Clearing
# echo "Clearing HuggingFace hub cache..."
# rm -rf ~/.cache/huggingface/hub/*
# echo "Cache cleared."

# Install required packages
pip install --quiet --upgrade transformers>=4.30.0 
pip install --quiet sentencepiece accelerate huggingface_hub protobuf cohere requests pandas tqdm matplotlib seaborn
# Ensure tokenizers version is suitable, adjust if needed
# pip install --quiet --force-reinstall "tokenizers==0.13.3" # Example version 

# Create output directories if they don't exist
mkdir -p new_outputs/llama_local_full # Updated dir name
mkdir -p new_outputs/perspective_full # Updated dir name
mkdir -p job_outputs

# Define input file (output from script 01)
FILTERED_INPUT_FILE="new_outputs/filtered_output_full.csv"

# --- REMOVED Step 1: Initial Perspective API run is now done in script 01 --- 

# --- Step 2: Generate Continuations Locally (using filtered data) --- 

# Check if input file exists
if [ ! -f "$FILTERED_INPUT_FILE" ]; then
    echo "Error: Filtered input file '$FILTERED_INPUT_FILE' not found! (Expected from script 01). Cannot run generation." 
    exit 1
fi

# --- Process Llama 3 --- 
echo "Running Llama 3 model locally on FULL filtered data..."
python new_python_scripts/generate_continuations_local.py \
    --input "$FILTERED_INPUT_FILE" \
    --output new_outputs/llama_local_full/code_switched_llama3_full.csv \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --text_column generated \
    --max_tokens 50 \
    --temperature 0.7
    # Removed --max_rows
    # --token_file path/to/your/token # Optional

# --- Process Llama 3.1 --- 
echo "Running Llama 3.1 model locally on FULL filtered data..."
python new_python_scripts/generate_continuations_local.py \
    --input "$FILTERED_INPUT_FILE" \
    --output new_outputs/llama_local_full/code_switched_llama31_full.csv \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --text_column generated \
    --max_tokens 50 \
    --temperature 0.7
    # Removed --max_rows
    # --token_file path/to/your/token # Optional

# --- Process Aya --- 
echo "Running Aya model locally on FULL filtered data..."
python new_python_scripts/generate_continuations_local.py \
    --input "$FILTERED_INPUT_FILE" \
    --output new_outputs/llama_local_full/code_switched_aya_full.csv \
    --model CohereForAI/aya-23-8B \
    --text_column generated \
    --max_tokens 50 \
    --temperature 0.7
    # Removed --max_rows
    # --token_file path/to/your/token # Optional

echo "Local CS continuation generation completed for full dataset."

# --- Step 3: Run Perspective API on the generated continuations --- 

# Get Perspective API key
PERSPECTIVE_API_KEY=$(python -c "import config; print(config.PERSPECTIVE_API_KEY)")

# Check if we have the Perspective API key
if [ -z "$PERSPECTIVE_API_KEY" ]; then
    echo "Error: PERSPECTIVE_API_KEY not found in config.py"
    exit 1
fi

# Define base paths for Perspective input/output
CS_CONT_BASE_DIR="new_outputs/llama_local_full"
CS_PERSP_OUTPUT_DIR="new_outputs/perspective_full"

# Run Perspective API on Llama 3 continuations
LLAMA3_CONT_FILE="$CS_CONT_BASE_DIR/code_switched_llama3_full.csv"
if [ -f "$LLAMA3_CONT_FILE" ]; then
    echo "Running Perspective API on Llama 3 continuations (local, full)..."
    python new_python_scripts/run_perspective_api.py \
        --input_file "$LLAMA3_CONT_FILE" \
        --output_file "$CS_PERSP_OUTPUT_DIR/llama3_continuations_perspective_local_full.csv" \
        --api_key $PERSPECTIVE_API_KEY \
        --text_column llama3_continuation # Specify the column to analyze
        # Removed --max_rows
else
    echo "Warning: Llama 3 continuation file not found ($LLAMA3_CONT_FILE), skipping Perspective API."
fi

# Run Perspective API on Llama 3.1 continuations
LLAMA31_CONT_FILE="$CS_CONT_BASE_DIR/code_switched_llama31_full.csv"
if [ -f "$LLAMA31_CONT_FILE" ]; then
    echo "Running Perspective API on Llama 3.1 continuations (local, full)..."
    python new_python_scripts/run_perspective_api.py \
        --input_file "$LLAMA31_CONT_FILE" \
        --output_file "$CS_PERSP_OUTPUT_DIR/llama31_continuations_perspective_local_full.csv" \
        --api_key $PERSPECTIVE_API_KEY \
        --text_column llama31_continuation # Specify the column to analyze
        # Removed --max_rows
else
    echo "Warning: Llama 3.1 continuation file not found ($LLAMA31_CONT_FILE), skipping Perspective API."
fi

# Run Perspective API on Aya continuations
AYA_CONT_FILE="$CS_CONT_BASE_DIR/code_switched_aya_full.csv"
if [ -f "$AYA_CONT_FILE" ]; then
    echo "Running Perspective API on Aya continuations (local, full)..."
    python new_python_scripts/run_perspective_api.py \
        --input_file "$AYA_CONT_FILE" \
        --output_file "$CS_PERSP_OUTPUT_DIR/aya_continuations_perspective_local_full.csv" \
        --api_key $PERSPECTIVE_API_KEY \
        --text_column aya_continuation # Specify the column to analyze
        # Removed --max_rows
else
    echo "Warning: Aya continuation file not found ($AYA_CONT_FILE), skipping Perspective API."
fi

echo "CS local generation and analysis complete for full dataset!"
echo "Results can be found in:"
echo "- Model continuations: $CS_CONT_BASE_DIR/"
echo "- Perspective API results: $CS_PERSP_OUTPUT_DIR/" 