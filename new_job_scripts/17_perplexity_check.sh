#!/bin/bash

#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --job-name=17_perplexity_check
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=job_outputs/17_perplexity_check_%j.out

# Load the required modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Activate your virtual environment if needed
source /home/tchakravorty/.bashrc
conda activate code-switch

# pip install pandas matplotlib seaborn tqdm transformers torch

# Run the script
python python_scripts/perplexity_comparison.py