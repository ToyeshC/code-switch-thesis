#!/bin/bash

#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=18_tox_perplexity_in_out_compare
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=job_outputs/18_tox_perplexity_in_out_compare_%j.out

# Load the required modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Activate your virtual environment if needed
source /home/tchakravorty/.bashrc
conda activate code-switch

pip install pandas matplotlib seaborn tqdm transformers torch

# Run the script
python python_scripts/perplexity_toxicity_correlation.py