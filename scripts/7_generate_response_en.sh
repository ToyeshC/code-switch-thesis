#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=7_generate_response_en_aya
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=10:00:00
#SBATCH --output=/home/tchakravorty/tchakravorty/code-switch-thesis/outputs/7_generate_response_en_aya.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# Set Hugging Face token
export HUGGING_FACE_HUB_TOKEN=$(cat ~/.huggingface/token)

# Login to Hugging Face
python -c "from huggingface_hub.cli.cli import login; login()"

python ezswitch/src/inference.py \
    --src data/extracted_prompts/train_hi.txt \
    --tgt data/extracted_prompts/train_en.txt \
    --src_translated data/translate_api_outputs/train_hi.txt \
    --tgt_translated data/translate_api_outputs/train_en.txt \
    --gold_align data/alignments/hi-en_align_gold.txt \
    --silver_src_align data/alignments/hi-en_align_silver.txt \
    --silver_tgt_align data/alignments/hi-en_align_silver.txt \
    --model_id "CohereForAI/aya-23-8B" \
    --output data/output/english/full_aya.csv


python ezswitch/src/inference.py \
    --src data/extracted_prompts/train_hi.txt \
    --tgt data/extracted_prompts/train_en.txt \
    --src_translated data/translate_api_outputs/train_hi.txt \
    --tgt_translated data/translate_api_outputs/train_en.txt \
    --gold_align data/alignments/hi-en_align_gold.txt \
    --silver_src_align data/alignments/hi-en_align_silver.txt \
    --silver_tgt_align data/alignments/hi-en_align_silver.txt \
    --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --output data/output/english/full_llama3_1.csv


python ezswitch/src/inference.py \
    --src data/extracted_prompts/train_hi.txt \
    --tgt data/extracted_prompts/train_en.txt \
    --src_translated data/translate_api_outputs/train_hi.txt \
    --tgt_translated data/translate_api_outputs/train_en.txt \
    --gold_align data/alignments/hi-en_align_gold.txt \
    --silver_src_align data/alignments/hi-en_align_silver.txt \
    --silver_tgt_align data/alignments/hi-en_align_silver.txt \
    --model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
    --output data/output/english/full_llama3.csv