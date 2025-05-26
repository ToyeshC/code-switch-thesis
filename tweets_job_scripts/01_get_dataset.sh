#!/bin/bash

#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=01_get_dataset
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=tweets_job_outputs/01_get_dataset_%j.out

# Load the required modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Activate your virtual environment if needed
source /home/tchakravorty/.bashrc
conda activate code-switch

python tweets_python_scripts/process_hinglish.py