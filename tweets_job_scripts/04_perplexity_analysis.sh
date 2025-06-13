#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=tweets_perplexity
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=tweets_job_outputs/04_perplexity_%j.out

# Activate conda environment
source /home/tchakravorty/.bashrc
conda activate code-switch

# Create output directory if it doesn't exist
mkdir -p tweets_outputs/analysis
mkdir -p tweets_job_outputs

# Run the script for each model
# LLama 3
python new_python_scripts/analyze_perplexity_language_toxicity.py \
    --input_csv tweets_outputs/perspective/llama3_continuations_perspective.csv \
    --output_csv tweets_outputs/analysis/llama3_perplexity_analysis.csv \
    --model meta-llama/Meta-Llama-3-8B-Instruct

# LLama 3.1
python new_python_scripts/analyze_perplexity_language_toxicity.py \
    --input_csv tweets_outputs/perspective/llama31_continuations_perspective.csv \
    --output_csv tweets_outputs/analysis/llama31_perplexity_analysis.csv \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct

# Aya
python new_python_scripts/analyze_perplexity_language_toxicity.py \
    --input_csv tweets_outputs/perspective/aya_continuations_perspective.csv \
    --output_csv tweets_outputs/analysis/aya_perplexity_analysis.csv \
    --model CohereForAI/aya-23-8B

echo "Perplexity analysis complete!" 