#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=tweets_gen_cs_small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=tweets_job_outputs/02_gen_cs_small_%j.out

# Activate conda environment
source /home/tchakravorty/.bashrc
conda activate code-switch

# First uninstall existing packages to avoid conflicts
pip uninstall -y torch torchvision torchaudio transformers

# Install PyTorch with CUDA support
pip install --quiet torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install transformers and other required packages
pip install --quiet transformers==4.36.0
pip install --quiet sentencepiece accelerate huggingface_hub protobuf cohere requests pandas tqdm matplotlib seaborn
pip install --quiet cohere-tokenizer  # Add Cohere tokenizer package

# Create output directories if they don't exist
mkdir -p tweets_outputs/llama_local_small
mkdir -p tweets_job_outputs

# Define input file (output from process_hinglish.py)
FILTERED_INPUT_FILE="tweets_outputs/processed_hinglish_small.csv"

# Check if input file exists
if [ ! -f "$FILTERED_INPUT_FILE" ]; then
    echo "Error: Input file '$FILTERED_INPUT_FILE' not found! Cannot run generation." 
    exit 1
fi

# --- Process Llama 3 --- 
echo "Running Llama 3 model locally on Hinglish small dataset..."
python new_python_scripts/generate_continuations_local.py \
    --input "$FILTERED_INPUT_FILE" \
    --output tweets_outputs/llama_local_small/code_switched_llama3_small.csv \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --text_column generated \
    --max_tokens 50 \
    --temperature 0.7

# --- Process Llama 3.1 --- 
echo "Running Llama 3.1 model locally on Hinglish small dataset..."
python new_python_scripts/generate_continuations_local.py \
    --input "$FILTERED_INPUT_FILE" \
    --output tweets_outputs/llama_local_small/code_switched_llama31_small.csv \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --text_column generated \
    --max_tokens 50 \
    --temperature 0.7 \
    --rope_scaling '{"type": "dynamic", "factor": 2.0}'

# --- Process Aya --- 
echo "Running Aya model locally on Hinglish small dataset..."
python new_python_scripts/generate_continuations_local.py \
    --input "$FILTERED_INPUT_FILE" \
    --output tweets_outputs/llama_local_small/code_switched_aya_small.csv \
    --model CohereForAI/aya-23-8B \
    --text_column generated \
    --max_tokens 50 \
    --temperature 0.7 \
    --use_cohere_tokenizer

echo "Local CS continuation generation completed for Hinglish small dataset."
echo "Results can be found in: tweets_outputs/llama_local_small/" 