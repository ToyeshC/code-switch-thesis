#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=2_run_ezswitch_hi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=final_job_outputs/2_run_ezswitch_hi_%A.out

# Create necessary directories
mkdir -p final_data/alignments
mkdir -p final_data/output/hindi

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
 --source final_data/extracted_prompts/train_en.txt \
 --target final_data/extracted_prompts/train_hi.txt \
 --alignments final_data/alignments/en-hi_align_gold.txt

# Generate Alignment Files from silver translations
 python ezswitch/alignment/giza-py/giza.py \
 --bin ezswitch/alignment/mgiza/mgizapp/bin \
 --source final_data/extracted_prompts/train_en.txt \
 --target final_data/translate_api_outputs/train_hi.txt \
 --alignments final_data/alignments/en-hi_align_silver.txt

# Set Hugging Face token
export HUGGING_FACE_HUB_TOKEN=$(cat ~/.huggingface/token)

# Login to Hugging Face
python -c "from huggingface_hub.cli.cli import login; login()"

python ezswitch/src/inference.py \
    --src final_data/extracted_prompts/train_en.txt \
    --tgt final_data/extracted_prompts/train_hi.txt \
    --src_translated final_data/translate_api_outputs/train_en.txt \
    --tgt_translated final_data/translate_api_outputs/train_hi.txt \
    --gold_align final_data/alignments/en-hi_align_gold.txt \
    --silver_src_align final_data/alignments/en-hi_align_silver.txt \
    --silver_tgt_align final_data/alignments/en-hi_align_silver.txt \
    --model_id "CohereForAI/aya-23-8B" \
    --output final_data/output/hindi/aya_23_8B.csv


python ezswitch/src/inference.py \
    --src final_data/extracted_prompts/train_en.txt \
    --tgt final_data/extracted_prompts/train_hi.txt \
    --src_translated final_data/translate_api_outputs/train_en.txt \
    --tgt_translated final_data/translate_api_outputs/train_hi.txt \
    --gold_align final_data/alignments/en-hi_align_gold.txt \
    --silver_src_align final_data/alignments/en-hi_align_silver.txt \
    --silver_tgt_align final_data/alignments/en-hi_align_silver.txt \
    --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --output final_data/output/hindi/llama_3_1_8B.csv


python ezswitch/src/inference.py \
    --src final_data/extracted_prompts/train_en.txt \
    --tgt final_data/extracted_prompts/train_hi.txt \
    --src_translated final_data/translate_api_outputs/train_en.txt \
    --tgt_translated final_data/translate_api_outputs/train_hi.txt \
    --gold_align final_data/alignments/en-hi_align_gold.txt \
    --silver_src_align final_data/alignments/en-hi_align_silver.txt \
    --silver_tgt_align final_data/alignments/en-hi_align_silver.txt \
    --model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
    --output final_data/output/hindi/llama_3_8B.csv

python ezswitch/src/compile.py \
    --directory final_data/output/hindi \
    --output final_data/output/compile_hindi.csv