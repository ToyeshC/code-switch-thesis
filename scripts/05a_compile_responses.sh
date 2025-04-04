#!/bin/bash

#SBATCH --partition=rome
#SBATCH --job-name=5a_compile_responses
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=10:00:00
#SBATCH --output=outputs/05a_compile_responses.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

python ezswitch/src/compile.py \
    --directory data/output/hindi \
    --output data/output/hindi/compile_hindi.csv