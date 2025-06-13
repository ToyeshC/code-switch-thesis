#!/bin/bash

#SBATCH --partition=rome
#SBATCH --job-name=1_translate_file
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=final_job_outputs/1_translate_file_%A.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

python final_python_scripts/translate_file.py \
    --input final_data/extracted_prompts/train_en.txt \
    --target hi \
    --output final_data/translate_api_outputs/train_hi.txt

python final_python_scripts/translate_file.py \
    --input final_data/extracted_prompts/train_hi.txt \
    --target en \
    --output final_data/translate_api_outputs/train_en.txt
