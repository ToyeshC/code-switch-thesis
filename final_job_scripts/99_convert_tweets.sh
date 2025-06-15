#!/bin/bash

#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=99_convert_tweets
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=final_job_outputs/99_convert_tweets_%j.out

# Load the required modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Activate your virtual environment if needed
source /home/tchakravorty/.bashrc
conda activate code-switch

python final_python_scripts/process_hinglish.py