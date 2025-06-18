#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=9_analyze_fluency_toxicity_correlation_multigpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=final_job_outputs/9_analyze_fluency_toxicity_correlation_multigpu_%A.out

# --- Setup: Activate environment and set cache ---
module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# --- Set Hugging Face cache directory ---
# This prevents re-downloading large models every time and avoids filling up the home directory.
export TRANSFORMERS_CACHE="/home/tchakravorty/.cache/huggingface/transformers"
echo "Transformers cache is set to: $TRANSFORMERS_CACHE"

# --- Set PyTorch settings for multi-GPU ---
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Available GPUs: $(nvidia-smi --list-gpus | wc -l)"

# --- Configuration ---
PERSPECTIVE_FILE="temp_outputs/perspective_analysis.csv"
HINGE_FILE="ezswitch/data/hinge/train.csv"
OUTPUT_DIR="temp_outputs"

# --- Check for input files ---
if [ ! -f "$PERSPECTIVE_FILE" ]; then
    echo "Error: Perspective analysis file '$PERSPECTIVE_FILE' not found!"
    echo "Checking alternative location..."
    PERSPECTIVE_FILE="temp_outputs/perspective_analysis.csv"
    if [ ! -f "$PERSPECTIVE_FILE" ]; then
        echo "Error: Perspective analysis file not found in temp_outputs either!"
        exit 1
    fi
    echo "Found perspective analysis file at: $PERSPECTIVE_FILE"
fi

if [ ! -f "$HINGE_FILE" ]; then
    echo "Error: HINGE dataset file '$HINGE_FILE' not found!"
    exit 1
fi

# --- GPU Memory and Performance Monitoring ---
echo "--- GPU Status Before Execution ---"
nvidia-smi

# --- Run the analysis script ---
echo "--- Starting: Multi-GPU Fluency and Toxicity Correlation Analysis (Experiment D) ---"
echo "Using 4 GPUs for accelerated processing"
python final_python_scripts/analyze_fluency_toxicity_correlation.py \
    --perspective_file "$PERSPECTIVE_FILE" \
    --hinge_file "$HINGE_FILE" \
    --output_dir "$OUTPUT_DIR"

# --- GPU Status After Execution ---
echo "--- GPU Status After Execution ---"
nvidia-smi

echo "--- Finished: Multi-GPU Analysis complete. Results in $OUTPUT_DIR/experiment_d ---"

# --- Display summary of generated files ---
echo "--- Generated files in experiment_d ---"
ls -la "$OUTPUT_DIR"/experiment_d/ 2>/dev/null || echo "No files found in experiment_d"

# --- Display first few lines of summary report if it exists ---
if [ -f "$OUTPUT_DIR/experiment_d/experiment_d_summary_report.txt" ]; then
    echo "--- Summary Report Preview ---"
    head -20 "$OUTPUT_DIR/experiment_d/experiment_d_summary_report.txt"
fi 