#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=5_evaluate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=/home/tchakravorty/tchakravorty/code-switch-thesis/outputs/5_evaluate.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# Set Hugging Face token
export HUGGING_FACE_HUB_TOKEN=$(cat ~/.huggingface/token)

python ezswitch/src/evaluate_generation.py \
    --generation_file data/output/compile_hindi.csv \
    --lang hi \
    --output_file data/output/evaluation_hindi.csv \
    --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct"