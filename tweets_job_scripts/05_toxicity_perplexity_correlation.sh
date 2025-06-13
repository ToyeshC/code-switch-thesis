#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=tweets_correlation
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=tweets_job_outputs/05_correlation_%j.out

# Activate conda environment
source /home/tchakravorty/.bashrc
conda activate code-switch

# Create output directory if it doesn't exist
mkdir -p tweets_outputs/analysis
mkdir -p tweets_job_outputs

# Run correlation analysis
python new_python_scripts/perplexity_toxicity_correlation.py \
    --llama3_csv tweets_outputs/analysis/llama3_perplexity_analysis.csv \
    --llama31_csv tweets_outputs/analysis/llama31_perplexity_analysis.csv \
    --aya_csv tweets_outputs/analysis/aya_perplexity_analysis.csv \
    --output_dir tweets_outputs/analysis

echo "Correlation analysis complete!" 