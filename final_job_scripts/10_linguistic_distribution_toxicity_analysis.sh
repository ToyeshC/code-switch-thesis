#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=10_linguistic_distribution_toxicity_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=final_job_outputs/10_linguistic_distribution_toxicity_analysis_%A.out

# --- Setup: Activate environment and install packages ---
module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# --- Configuration ---
INPUT_FILE="final_outputs/perspective_analysis.csv"
OUTPUT_DIR="final_outputs"
FASTTEXT_MODEL="lid.176.bin"

# --- Check for input file ---
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found! (Run script 5 first)"
    exit 1
fi

# --- Check for FastText model ---
if [ ! -f "$FASTTEXT_MODEL" ]; then
    echo "Error: FastText model '$FASTTEXT_MODEL' not found!"
    exit 1
fi

# --- Run linguistic distribution of toxicity and feature attribution analysis ---
echo "--- Starting: Linguistic Distribution of Toxicity and Feature Attribution Analysis ---"
python final_python_scripts/linguistic_distribution_toxicity_analysis.py \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --fasttext_model "$FASTTEXT_MODEL"

echo "--- Finished: Linguistic distribution and feature attribution analysis complete. Results in $OUTPUT_DIR/experiment_e ---"

# --- Display summary of generated files ---
echo "--- Generated files ---"
ls -la "$OUTPUT_DIR"/experiment_e/ 2>/dev/null || echo "No files found in experiment_e" 