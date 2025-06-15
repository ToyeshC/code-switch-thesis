#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=102_run_perspective_api_tweets
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=final_job_outputs/102_run_perspective_api_tweets_%A.out

# --- Setup: Activate environment and install packages ---
module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# --- Configuration ---
INPUT_FILE="final_outputs/continuations_tweets.csv"
OUTPUT_FILE="final_outputs/perspective_analysis_tweets.csv"

# --- Check for input file ---
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found! (Run script 4 first)"
    exit 1
fi

# --- Get Perspective API key ---
PERSPECTIVE_API_KEY=$(python -c "import config; print(config.PERSPECTIVE_API_KEY)")
if [ -z "$PERSPECTIVE_API_KEY" ]; then
    echo "Error: PERSPECTIVE_API_KEY not found in config.py"
    exit 1
fi

# --- Run Perspective API analysis on all columns ---
echo "--- Starting: Running Perspective API analysis in parallel ---"
python final_python_scripts/run_perspective_api.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --api_key "$PERSPECTIVE_API_KEY" \
    --num_workers 16 \
    --columns "generated,llama3_generated_continuation,llama31_generated_continuation,aya_generated_continuation"

echo "--- Finished: Perspective API analysis complete. Results in $OUTPUT_FILE ---" 