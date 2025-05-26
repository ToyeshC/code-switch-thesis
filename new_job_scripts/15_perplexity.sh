#!/bin/bash

#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --job-name=15_perplexity
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=job_outputs/15_perplexity_%j.out

# Job script to run correlation plots for toxicity scores

# Load the required modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Activate your virtual environment if needed
source /home/tchakravorty/.bashrc
conda activate code-switch

# Install required dependencies
pip install sentencepiece

# Run the script
python new_python_scripts/analyze_perplexity_language_toxicity.py \
  --input_csv temp_scripts/perspective_analysis_outputs/perspective_analysis_results_small.csv \
  --save_plots