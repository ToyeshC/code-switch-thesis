#!/bin/bash

#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=17_perplexity_check
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=job_outputs/17_perplexity_check_%j.out

# Load the required modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Activate your virtual environment if needed
source /home/tchakravorty/.bashrc
conda activate code-switch

# Install required packages
pip --quiet uninstall torch torchvision torchaudio
pip --quiet install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip --quiet install pandas matplotlib seaborn tqdm transformers torch

# Run the script
python python_scripts/perplexity_comparison.py