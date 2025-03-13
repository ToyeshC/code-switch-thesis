#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=14_generate_continuations
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=10:00:00
#SBATCH --output=/home/tchakravorty/tchakravorty/code-switch-thesis/outputs/14_generate_continuations.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

python src/generate_continuations.py --input data/output/small_filtered_output.csv --output data/output/small_continued_output.csv --model meta-llama/Meta-Llama-3-8B-Instruct