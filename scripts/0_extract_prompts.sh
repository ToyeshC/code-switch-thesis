#!/bin/bash

#SBATCH --partition=rome
#SBATCH --job-name=0_extract_prompts
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=outputs/0_extract_prompts.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

python src/extract_prompts.py data/RTP-LX/RTP_LX_HI.json data/extracted_prompts/train_hi.txt
python src/extract_prompts.py data/RTP-LX/RTP_LX_EN.json data/extracted_prompts/train_en.txt