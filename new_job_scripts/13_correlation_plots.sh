#!/bin/bash

#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=13_corr_plots_full
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=job_outputs/13_corr_plots_full_%j.out

# Job script to run correlation plots for toxicity scores

# Load the required modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Activate your virtual environment if needed
source /home/tchakravorty/.bashrc
conda activate code-switch

# Install required Python packages
pip install --quiet pandas numpy matplotlib seaborn scipy statsmodels

# Set the output directory
OUTPUT_DIR="new_outputs/correlation_plots"
mkdir -p $OUTPUT_DIR

# Set the input directories
SRC_DIR="new_outputs/src_results_full"
TGT_DIR="new_outputs/tgt_results_full"
CS_DIR="new_outputs/perspective_full"

# Define the models
MODELS="aya llama3 llama31"

# Run the correlation plots script with all analyses
echo "Running enhanced analysis scripts..."
python new_python_scripts/correlation_plots.py \
    --src_dir $SRC_DIR \
    --tgt_dir $TGT_DIR \
    --cs_dir $CS_DIR \
    --output_dir $OUTPUT_DIR \
    --models $MODELS \
    --analyses all \
    --src_pattern "{model}_src_continuations_full.csv" \
    --tgt_pattern "{model}_tgt_continuations_full.csv" \
    --cs_pattern "{model}_continuations_perspective_local_full.csv"

echo "Completed generating plots and statistical analyses. Results saved to $OUTPUT_DIR" 