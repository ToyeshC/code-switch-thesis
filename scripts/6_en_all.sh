#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=6_en_all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=12:00:00
#SBATCH --output=/home/tchakravorty/tchakravorty/code-switch-thesis/outputs/6_en_all.out

module purge

module load 2024
module load Boost/1.85.0-GCC-13.3.0

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# Generate Alignment Files from gold translations
python ezswitch/alignment/giza-py/giza.py \
 --bin ezswitch/alignment/mgiza/mgizapp/bin \
 --source data/extracted_prompts/train_hi.txt \
 --target data/extracted_prompts/train_en.txt \
 --alignments data/alignments/hi-en_align_gold.txt

# Generate Alignment Files from silver translations
 python ezswitch/alignment/giza-py/giza.py \
 --bin ezswitch/alignment/mgiza/mgizapp/bin \
 --source data/extracted_prompts/train_hi.txt \
 --target data/translate_api_outputs/train_en.txt \
 --alignments data/alignments/hi-en_align_silver.txt

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
    --output data/output/english/full_aya.csv


python ezswitch/src/inference.py \
    --src data/extracted_prompts/train_hi.txt \
    --tgt data/extracted_prompts/train_en.txt \
    --src_translated data/translate_api_outputs/train_hi.txt \
    --tgt_translated data/translate_api_outputs/train_en.txt \
    --gold_align data/alignments/hi-en_align_gold.txt \
    --silver_src_align data/alignments/hi-en_align_silver.txt \
    --silver_tgt_align data/alignments/hi-en_align_silver.txt \
    --model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
    --output data/output/english/full_aya.csv

python ezswitch/src/compile.py \
    --directory data/output/english \
    --output data/output/compile_hindi.csv