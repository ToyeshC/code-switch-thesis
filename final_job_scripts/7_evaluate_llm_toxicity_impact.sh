#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=7_evaluate_llm_toxicity_impact
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=final_job_outputs/7_evaluate_llm_toxicity_impact_%A.out

# --- Setup: Activate environment and install packages ---
module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# --- Configuration ---
INPUT_FILE="temp_outputs/perspective_analysis.csv"
OUTPUT_DIR="temp_outputs"

# --- Check for input file ---
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found! (Run script 6 first)"
    exit 1
fi

# --- Run LLM toxicity impact analysis ---
echo "--- Starting: LLM Toxicity Impact Analysis ---"
python final_python_scripts/evaluate_llm_toxicity_impact.py \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR"

echo "--- Finished: LLM toxicity impact analysis complete. Results in $OUTPUT_DIR/experiment_b ---"

# --- Display summary of generated files ---
echo "--- Generated files ---"
ls -la "$OUTPUT_DIR"/experiment_b/ 2>/dev/null || echo "No files found in experiment_b" 