#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=07_gen_srctgt_full
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1 
#SBATCH --mem=64G 
#SBATCH --time=48:00:00
#SBATCH --output=job_outputs/07_gen_srctgt_full_%j.out

# Activate conda environment
source /home/tchakravorty/.bashrc
conda activate code-switch

# Install required packages (ensure they are present)
pip install --quiet --upgrade transformers>=4.30.0 
pip install --quiet sentencepiece accelerate huggingface_hub protobuf cohere requests pandas tqdm
# Ensure tokenizers version is suitable
# pip install --quiet --force-reinstall "tokenizers==0.13.3" 

# Define Directories and Settings
INPUT_FILE="new_outputs/filtered_output_full.csv" # File containing src and tgt columns (output from script 01)
SRC_OUTPUT_DIR="new_outputs/src_results_full" # Updated dir name
TGT_OUTPUT_DIR="new_outputs/tgt_results_full" # Updated dir name
# MAX_ROWS removed
MAX_TOKENS=50
TEMPERATURE=0.7

# Create output directories
mkdir -p $SRC_OUTPUT_DIR
mkdir -p $TGT_OUTPUT_DIR
mkdir -p job_outputs

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found! (Expected from script 01)." 
    exit 1
fi

# --- Models to process --- 
MODELS=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "CohereForAI/aya-23-8B"
)

# --- Loop through models and columns --- 
for model_full_name in "${MODELS[@]}"; do
    # Extract a short name for filename
    if [[ "$model_full_name" == *"aya-23-8B"* ]]; then model_short_name="aya"; 
    elif [[ "$model_full_name" == *"Meta-Llama-3.1-8B-Instruct"* ]]; then model_short_name="llama31"; 
    elif [[ "$model_full_name" == *"Meta-Llama-3-8B-Instruct"* ]]; then model_short_name="llama3"; 
    else model_short_name=$(basename "$model_full_name"); fi

    echo "\n========================================================="
    echo "Processing Model: $model_full_name (Short: $model_short_name) for SRC/TGT on FULL data"
    echo "========================================================="

    # --- Process SRC column --- 
    echo "\n---> Generating continuations for SRC column..."
    python new_python_scripts/generate_continuations_local.py \
        --input "$INPUT_FILE" \
        --output "$SRC_OUTPUT_DIR/${model_short_name}_src_continuations_full.csv" \
        --model "$model_full_name" \
        --text_column src \
        --max_tokens $MAX_TOKENS \
        --temperature $TEMPERATURE
        # Removed --max_rows
        # Add --token_file if needed
        
    # --- Process TGT column --- 
    echo "\n---> Generating continuations for TGT column..."
    python new_python_scripts/generate_continuations_local.py \
        --input "$INPUT_FILE" \
        --output "$TGT_OUTPUT_DIR/${model_short_name}_tgt_continuations_full.csv" \
        --model "$model_full_name" \
        --text_column tgt \
        --max_tokens $MAX_TOKENS \
        --temperature $TEMPERATURE
        # Removed --max_rows
        # Add --token_file if needed
        
    echo "\nFinished processing model: $model_full_name"
done

echo "\nAll models processed for SRC and TGT columns for full dataset."
echo "SRC results are in: $SRC_OUTPUT_DIR"
echo "TGT results are in: $TGT_OUTPUT_DIR" 