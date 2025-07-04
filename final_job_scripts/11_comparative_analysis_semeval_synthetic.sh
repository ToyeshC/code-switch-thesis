#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=11_comparative_analysis_semeval_synthetic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=final_job_outputs/11_comparative_analysis_semeval_synthetic_%A.out
#SBATCH --gres=gpu:1

# --- Setup: Activate environment and install packages ---
module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# --- Configuration ---
TWEETS_FILE="final_outputs/perspective_analysis_tweets.csv"
SYNTHETIC_FILE="temp_outputs/perspective_analysis.csv"
OUTPUT_DIR="temp_outputs"

# --- Check for input files ---
if [ ! -f "$TWEETS_FILE" ]; then
    echo "Error: Tweets file '$TWEETS_FILE' not found!"
    exit 1
fi

if [ ! -f "$SYNTHETIC_FILE" ]; then
    echo "Error: Synthetic data file '$SYNTHETIC_FILE' not found!"
    exit 1
fi

# --- Create output directory structure ---
mkdir -p "$OUTPUT_DIR"/experiment_f/visualizations

# --- Run comparative analysis ---
echo "--- Starting: Comparative Analysis between SemEval Tweets and Synthetic Data ---"
echo "Tweets file: $TWEETS_FILE ($(wc -l < $TWEETS_FILE) lines)"
echo "Synthetic file: $SYNTHETIC_FILE ($(wc -l < $SYNTHETIC_FILE) lines)"
echo "Output directory: $OUTPUT_DIR/experiment_f"

python final_python_scripts/comparative_analysis_semeval_synthetic.py \
    --tweets_file "$TWEETS_FILE" \
    --synthetic_file "$SYNTHETIC_FILE" \
    --output_dir "$OUTPUT_DIR"

echo "--- Finished: Comparative analysis complete. Results in $OUTPUT_DIR/experiment_f ---"

# --- Display summary of generated files ---
echo "--- Generated files ---"
find "$OUTPUT_DIR"/experiment_f/ -type f -name "*.csv" -o -name "*.json" -o -name "*.png" | sort
echo "--- Total files generated: $(find "$OUTPUT_DIR"/experiment_f/ -type f | wc -l) ---" 