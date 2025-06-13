#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=4_generate_continuations
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=final_job_outputs/4_generate_continuations_%A.out

# --- Setup: Activate environment and install packages ---
module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# --- Configuration ---
INPUT_FILE="final_outputs/filtered_output.csv"
OUTPUT_FILE="final_outputs/continuations.csv"
BATCH_SIZE=16

# --- Create output directories ---
mkdir -p "$(dirname "$OUTPUT_FILE")"

# --- Check for input file ---
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found!"
    exit 1
fi

# --- Get Hugging Face API key ---
HUGGINGFACE_API_KEY=$(python -c "import config; print(config.HUGGINGFACE_API_KEY)")
if [ -z "$HUGGINGFACE_API_KEY" ]; then
    echo "Error: HUGGINGFACE_API_KEY not found in config.py"
    exit 1
fi

# --- Generate all continuations with batching ---
echo "--- Starting: Generating all continuations with batch size $BATCH_SIZE ---"
python final_python_scripts/generate_continuations_local.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --token "$HUGGINGFACE_API_KEY" \
    --max_tokens 50 \
    --temperature 0.7 \
    --batch_size $BATCH_SIZE

echo "--- Finished: All continuations generated and saved to $OUTPUT_FILE ---" 