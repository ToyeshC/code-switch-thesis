#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=compare_scores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4 
#SBATCH --mem=16G 
#SBATCH --time=02:00:00 # Adjust time as needed
#SBATCH --output=job_outputs/06_compare_scores_%j.out

# Activate conda environment
source /home/tchakravorty/.bashrc
conda activate code-switch

# Install required Python packages for analysis
pip install --quiet pandas matplotlib seaborn requests tqdm

# Define directories and common variables
EN_HI_DIR="en_hi_results_vera"
CS_DIR="new_outputs/llama_local_20"
CS_PERSPECTIVE_DIR="new_outputs/perspective_20"
OUTPUT_DIR="new_outputs/comparison_results_20"
MAX_ROWS=200 # Keep max rows, comparison script handles the actual number of rows present

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p job_outputs

# Get Perspective API key
PERSPECTIVE_API_KEY=$(python -c "import config; print(config.PERSPECTIVE_API_KEY)")

# Check API key
if [ -z "$PERSPECTIVE_API_KEY" ]; then
    echo "Error: PERSPECTIVE_API_KEY not found in config.py"
    exit 1
fi

# --- Compare for Llama 3 --- 
MODEL_NAME="llama3"
MODEL_FULL_NAME="Meta-Llama-3-8B-Instruct"
echo "\n--- Comparing scores for $MODEL_FULL_NAME --- "
python new_python_scripts/compare_scores.py \
    --en_pkl "$EN_HI_DIR/${MODEL_FULL_NAME}_en_rtp-lx_completions.pkl" \
    --hi_pkl "$EN_HI_DIR/${MODEL_FULL_NAME}_hi_rtp-lx_completions.pkl" \
    --cs_csv "$CS_PERSPECTIVE_DIR/${MODEL_NAME}_continuations_perspective_local.csv" \
    --api_key "$PERSPECTIVE_API_KEY" \
    --output_dir "$OUTPUT_DIR" \
    --max_rows $MAX_ROWS \
    --model_name $MODEL_NAME

# --- Compare for Llama 3.1 --- 
MODEL_NAME="llama31"
MODEL_FULL_NAME="Meta-Llama-3.1-8B-Instruct"
echo "\n--- Comparing scores for $MODEL_FULL_NAME --- "
python new_python_scripts/compare_scores.py \
    --en_pkl "$EN_HI_DIR/${MODEL_FULL_NAME}_en_rtp-lx_completions.pkl" \
    --hi_pkl "$EN_HI_DIR/${MODEL_FULL_NAME}_hi_rtp-lx_completions.pkl" \
    --cs_csv "$CS_PERSPECTIVE_DIR/${MODEL_NAME}_continuations_perspective_local.csv" \
    --api_key "$PERSPECTIVE_API_KEY" \
    --output_dir "$OUTPUT_DIR" \
    --max_rows $MAX_ROWS \
    --model_name $MODEL_NAME
    
# --- Compare for Aya --- 
MODEL_NAME="aya"
MODEL_FULL_NAME="aya-23-8B"
echo "\n--- Comparing scores for $MODEL_FULL_NAME --- "
python new_python_scripts/compare_scores.py \
    --en_pkl "$EN_HI_DIR/${MODEL_FULL_NAME}_en_rtp-lx_completions.pkl" \
    --hi_pkl "$EN_HI_DIR/${MODEL_FULL_NAME}_hi_rtp-lx_completions.pkl" \
    --cs_csv "$CS_PERSPECTIVE_DIR/${MODEL_NAME}_continuations_perspective_local.csv" \
    --api_key "$PERSPECTIVE_API_KEY" \
    --output_dir "$OUTPUT_DIR" \
    --max_rows $MAX_ROWS \
    --model_name $MODEL_NAME

echo "\nComparison job finished. Results are in $OUTPUT_DIR" 