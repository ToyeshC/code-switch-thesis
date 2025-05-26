#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=10_finetune_clf_full
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=job_outputs/10_finetune_clf_full_%j.out

# Activate conda environment
source /home/tchakravorty/.bashrc
conda activate code-switch

# Install required packages if needed
pip install --quiet transformers datasets evaluate scikit-learn torch

# Define models to process
declare -a MODELS=("aya" "llama3" "llama31")
TEXT_COLUMN="generated"
LABEL_COLUMN="perspective_aya_continuation_toxicity"
MODEL_NAME="google/muril-base-cased"  # Good for Hindi-English code-switched text
BATCH_SIZE=16
EPOCHS=5
MAX_LENGTH=128
TEST_SIZE=0.2
# MAX_SAMPLES=2000  # Limit samples for faster experimentation - REMOVED for full dataset run

# Create output directories
mkdir -p job_outputs
mkdir -p new_outputs/classifier_metrics

# Process each LLM model
for MODEL in "${MODELS[@]}"
do
    echo "===================================================="
    echo "Starting classifier fine-tuning for ${MODEL} model..."
    echo "===================================================="
    
    # Set variables for current model
    INPUT_FILE="new_outputs/perspective_full/${MODEL}_continuations_perspective_local_full.csv"
    OUTPUT_DIR="new_outputs/models/${MODEL}_toxicity_classifier_full"
    METRICS_DIR="new_outputs/classifier_metrics/${MODEL}_full"
    
    # Create directories
    mkdir -p $OUTPUT_DIR
    mkdir -p $METRICS_DIR
    
    # Check if input file exists
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Warning: Input file $INPUT_FILE not found. Skipping ${MODEL}..."
        continue
    fi
    
    # Run the fine-tuning script with evaluation
    python new_python_scripts/finetune_classifier.py \
        --input_file $INPUT_FILE \
        --output_dir $OUTPUT_DIR \
        --metrics_dir $METRICS_DIR \
        --text_column $TEXT_COLUMN \
        --label_column $LABEL_COLUMN \
        --model_name $MODEL_NAME \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --max_length $MAX_LENGTH \
        --test_size $TEST_SIZE \
        --evaluate_accuracy true
    
    if [ $? -eq 0 ]; then
        echo "Fine-tuning complete for ${MODEL}. Model saved to $OUTPUT_DIR"
        echo "Classification metrics saved to $METRICS_DIR"
    else
        echo "ERROR: Fine-tuning failed for ${MODEL}."
        mkdir -p $METRICS_DIR
    fi
    echo "----------------------------------------------------"
done

# Generate combined metrics report
echo "Generating combined metrics report..."
python new_python_scripts/compare_classifier_metrics.py \
    --metrics_dir new_outputs/classifier_metrics \
    --output_file new_outputs/classifier_metrics/combined_report.html \
    --models "aya,llama3,llama31"

echo "All fine-tuning jobs completed."
echo "Combined metrics report saved to new_outputs/classifier_metrics/combined_report.html" 