#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=8_evaluate_cs_generation_methods
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=final_job_outputs/8_evaluate_cs_generation_methods_%A.out

# --- Setup: Activate environment and set cache ---
module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# --- Set Hugging Face cache directory ---
# This prevents re-downloading large models every time and avoids filling up the home directory.
# You might want to change this path to a shared project space.
export TRANSFORMERS_CACHE="/home/tchakravorty/.cache/huggingface/transformers"
echo "Transformers cache is set to: $TRANSFORMERS_CACHE"

# --- Configuration ---
INPUT_FILE="temp_outputs/perspective_analysis.csv"
OUTPUT_DIR="temp_outputs"

# --- Check for input file ---
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found!"
    exit 1
fi

# --- Run the evaluation script ---
echo "--- Starting: Evaluation of Code-Switching Generation Methods ---"
python final_python_scripts/evaluate_cs_generation_methods.py \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR"

echo "--- Finished: Evaluation complete. Results in $OUTPUT_DIR/experiment_c ---"

# --- Display summary of generated files ---
echo "--- Generated files in experiment_c ---"
ls -la "$OUTPUT_DIR"/experiment_c/ 2>/dev/null || echo "No files found in experiment_c" 