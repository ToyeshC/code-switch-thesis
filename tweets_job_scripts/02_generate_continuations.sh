#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=tweets_gen_cs
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=tweets_job_outputs/02_gen_cs_%j.out

# Activate conda environment
source /home/tchakravorty/.bashrc
conda activate code-switch

# First uninstall existing packages to avoid conflicts
pip uninstall -y torch torchvision torchaudio transformers numpy tokenizers

# Install compatible NumPy version first
pip install numpy==1.26.4

# Install PyTorch with CUDA support
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install compatible transformers and tokenizers versions
pip install transformers==4.41.1 tokenizers>=0.15.0
pip install accelerate datasets sentencepiece protobuf
pip install cohere  # For Aya model tokenizer support
pip install huggingface_hub requests pandas tqdm

# Set Hugging Face token from config
export HUGGINGFACE_HUB_TOKEN=$(python -c "import sys; sys.path.append('.'); import config; print(config.HUGGINGFACE_API_KEY)")

# Create output directories if they don't exist
mkdir -p tweets_outputs/llama_local
mkdir -p tweets_job_outputs

# Define input file (output from process_hinglish.py)
FILTERED_INPUT_FILE="tweets_outputs/processed_hinglish.csv"

# Check if input file exists
if [ ! -f "$FILTERED_INPUT_FILE" ]; then
    echo "Error: Input file '$FILTERED_INPUT_FILE' not found! Cannot run generation." 
    exit 1
fi

# --- Process Llama 3 --- 
echo "Running Llama 3 model locally on Hinglish small dataset..."
python new_python_scripts/generate_continuations_local.py \
    --input "$FILTERED_INPUT_FILE" \
    --output tweets_outputs/llama_local/code_switched_llama3.csv \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --text_column generated \
    --max_tokens 50 \
    --temperature 0.7

# --- Process Llama 3.1 --- 
echo "Running Llama 3.1 model locally on Hinglish small dataset..."
python new_python_scripts/generate_continuations_local.py \
    --input "$FILTERED_INPUT_FILE" \
    --output tweets_outputs/llama_local/code_switched_llama31.csv \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --text_column generated \
    --max_tokens 50 \
    --temperature 0.7

# --- Process Aya --- 
echo "Running Aya model locally on Hinglish dataset..."
python new_python_scripts/generate_continuations_local.py \
    --input "$FILTERED_INPUT_FILE" \
    --output tweets_outputs/llama_local/code_switched_aya.csv \
    --model CohereForAI/aya-23-8B \
    --text_column generated \
    --max_tokens 50 \
    --temperature 0.7

echo "Local CS continuation generation completed for Hinglish dataset."
echo "Results can be found in: tweets_outputs/llama_local/" 