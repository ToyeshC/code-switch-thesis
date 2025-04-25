#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=10_finetune_clf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=job_outputs/10_finetune_clf_%j.out

# Activate conda environment
source /home/tchakravorty/.bashrc
conda activate code-switch

# Install required packages if needed
pip install --quiet transformers datasets evaluate scikit-learn torch

# Define models to process
declare -a MODELS=("aya" "llama3" "llama31")
TEXT_COLUMN="generated"
LABEL_COLUMN="perspective_generated_toxicity"
MODEL_NAME="google/muril-base-cased"  # Good for Hindi-English code-switched text
BATCH_SIZE=16
EPOCHS=5
MAX_LENGTH=128
TEST_SIZE=0.2
MAX_SAMPLES=2000  # Limit samples for faster experimentation

# Create output directories
mkdir -p job_outputs

# Process each LLM model
for MODEL in "${MODELS[@]}"
do
    echo "===================================================="
    echo "Starting classifier fine-tuning for ${MODEL} model..."
    echo "===================================================="
    
    # Set variables for current model
    INPUT_FILE="new_outputs/perspective/${MODEL}_continuations_perspective_local.csv"
    OUTPUT_DIR="new_outputs/models/${MODEL}_toxicity_classifier"
    
    # Create model output directory
    mkdir -p $OUTPUT_DIR
    
    # Check if input file exists
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Warning: Input file $INPUT_FILE not found. Skipping ${MODEL}..."
        continue
    fi
    
    # Run the fine-tuning script
    python new_python_scripts/finetune_classifier.py \
        --input_file $INPUT_FILE \
        --output_dir $OUTPUT_DIR \
        --text_column $TEXT_COLUMN \
        --label_column $LABEL_COLUMN \
        --model_name $MODEL_NAME \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --max_length $MAX_LENGTH \
        --test_size $TEST_SIZE \
        --max_samples $MAX_SAMPLES
    
    echo "Fine-tuning complete for ${MODEL}. Model saved to $OUTPUT_DIR"
    echo "----------------------------------------------------"
done

echo "All fine-tuning jobs completed." 