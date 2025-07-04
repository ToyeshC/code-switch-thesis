#!/bin/bash

#SBATCH --partition=rome
#SBATCH --job-name=0_extract_prompts
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=final_job_outputs/0_extract_prompts_%A.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

python final_python_scripts/extract_prompts.py final_data/RTP-LX/RTP_LX_HI.json final_data/extracted_prompts/train_hi.txt
python final_python_scripts/extract_prompts.py final_data/RTP-LX/RTP_LX_EN.json final_data/extracted_prompts/train_en.txt