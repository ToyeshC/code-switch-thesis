#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=12_human_evaluation_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=final_job_outputs/12_human_evaluation_analysis_%A.out

# --- Setup: Activate environment and install packages ---
module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# --- Configuration ---
FORM_RESPONSES_FILE="final_data/Google Form Responses.csv"
PERSPECTIVE_FILE="temp_scripts/perspective_analysis_form.csv"
OUTPUT_DIR="final_outputs"

# --- Check for input files ---
if [ ! -f "$FORM_RESPONSES_FILE" ]; then
    echo "Error: Google Form responses file '$FORM_RESPONSES_FILE' not found!"
    exit 1
fi

if [ ! -f "$PERSPECTIVE_FILE" ]; then
    echo "Error: Perspective analysis file '$PERSPECTIVE_FILE' not found!"
    exit 1
fi

# --- Create output directory if it doesn't exist ---
mkdir -p "$OUTPUT_DIR"

# --- Run human evaluation analysis ---
echo "--- Starting: Human Evaluation Analysis ---"
echo "Input files:"
echo "  Google Form Responses: $FORM_RESPONSES_FILE"
echo "  Perspective Analysis: $PERSPECTIVE_FILE"
echo "Output directory: $OUTPUT_DIR/experiment_g"
echo

python final_python_scripts/human_evaluation_analysis.py \
    --form_responses_file "$FORM_RESPONSES_FILE" \
    --perspective_file "$PERSPECTIVE_FILE" \
    --output_dir "$OUTPUT_DIR"

echo "--- Finished: Human evaluation analysis complete. Results in $OUTPUT_DIR/experiment_g ---"

# --- Display summary of generated files ---
echo "--- Generated files ---"
if [ -d "$OUTPUT_DIR/experiment_g" ]; then
    ls -la "$OUTPUT_DIR"/experiment_g/
else
    echo "No experiment_g directory found"
fi 