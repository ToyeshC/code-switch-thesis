#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=tweets_eda
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=tweets_job_outputs/06_eda_%j.out

# Job script to perform exploratory data analysis on tweets dataset

# Load the required modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Activate your virtual environment if needed
source /home/tchakravorty/.bashrc
conda activate code-switch

# Install required dependencies for EDA
pip install matplotlib seaborn wordcloud scipy

# Create output directory
mkdir -p tweets_outputs/analysis
mkdir -p tweets_job_outputs

# Run the exploratory data analysis
python new_python_scripts/exploratory_data_analysis.py \
    --input_csv tweets_outputs/processed_hinglish.csv \
    --output_dir tweets_outputs/analysis

echo "Exploratory data analysis complete!" 