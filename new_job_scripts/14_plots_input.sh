#!/bin/bash

#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=14_prompt_toxicity_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=job_outputs/14_prompt_toxicity_analysis_%j.out

# Job script to run correlation plots for toxicity scores

# Load the required modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Activate your virtual environment if needed
source /home/tchakravorty/.bashrc
conda activate code-switch

python new_python_scripts/input_prompt_analysis.py \
    --input_csv temp_scripts/perspective_analysis_outputs/perspective_analysis_results.csv \
    --output_dir output/input_prompt_analysis \
    --analyses all