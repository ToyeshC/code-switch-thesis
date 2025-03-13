#!/bin/bash

#SBATCH --partition=rome
#SBATCH --job-name=4_compile_english
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=10:00:00
#SBATCH --output=/home/tchakravorty/tchakravorty/code-switch-thesis/outputs/4_compile_english.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

python ezswitch/src/compile.py \
    --directory data/output/english \
    --output data/output/compile_english.csv