#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=6_evaluate_perspective_robustness
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=final_job_outputs/6_evaluate_perspective_robustness_%A.out

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
    echo "Error: Input file '$INPUT_FILE' not found! (Run script 5 first)"
    exit 1
fi

# --- Install additional required packages ---
echo "--- Installing required packages ---"
pip install scipy seaborn matplotlib pandas numpy

# --- Run Perspective API robustness evaluation ---
echo "--- Starting: Perspective API Robustness Evaluation ---"
python final_python_scripts/evaluate_perspective_robustness.py \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR"

echo "--- Finished: Perspective API robustness evaluation complete. Results in $OUTPUT_DIR ---"

# --- Display summary of generated files ---
echo "--- Generated files ---"
ls -la "$OUTPUT_DIR"/perspective_*_$(date +%Y%m%d)*.{csv,txt,png} 2>/dev/null || echo "No files found with today's date"
echo "--- All perspective analysis files ---"
ls -la "$OUTPUT_DIR"/perspective_*.{csv,txt,png} 2>/dev/null || echo "No perspective analysis files found" 