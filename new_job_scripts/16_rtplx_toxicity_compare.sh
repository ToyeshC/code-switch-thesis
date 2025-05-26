#!/bin/bash

#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=16_rtplx_toxicity_compare
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=job_outputs/16_rtplx_toxicity_compare_%j.out

# Load the required modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Activate your virtual environment if needed
source /home/tchakravorty/.bashrc
conda activate code-switch

# Run the script
python python_scripts/toxicity_comparison.py