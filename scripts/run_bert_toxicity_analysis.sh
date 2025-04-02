#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=bert_toxicity
#SBATCH --mem=32G
#SBATCH --output=outputs/bert_toxicity_analysis.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# Set Hugging Face token
export HUGGING_FACE_HUB_TOKEN=$(cat ~/.huggingface/token)

# Define paths
TOXICITY_DIR="data/output/toxicity_analysis"
FILTERED_DIR="data/output/filtered"
LANG_DIR="data/output/language_detection"
OUTPUT_DIR="data/output/bert_toxicity_analysis"

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/model
mkdir -p $OUTPUT_DIR/feature_attribution
mkdir -p $OUTPUT_DIR/feature_attribution/monolingual
mkdir -p $OUTPUT_DIR/feature_attribution/code_switched

echo "========================================"
echo "BERT-based Toxicity Analysis for Code-switched Hindi-English Text"
echo "========================================"

# Step 0: Prepare categorized dataset with language information
echo "Step 0: Preparing categorized dataset..."
python scripts/prepare_for_bert_analysis.py \
    --toxicity_file $TOXICITY_DIR/hindi_prompt_toxicity.csv \
    --language_file $LANG_DIR/hindi_language_detection.csv \
    --filtered_file $FILTERED_DIR/hindi_filtered.csv \
    --output_dir $OUTPUT_DIR

# Step 1: Fine-tune mBERT for toxicity classification
echo "Step 1: Fine-tuning BERT for toxicity classification..."
python scripts/finetune_bert_classifier.py \
    --input_file $OUTPUT_DIR/hindi_categorized_toxicity.csv \
    --output_dir $OUTPUT_DIR/model \
    --model_name "bert-base-multilingual-cased" \
    --max_length 128 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --epochs 4

# Step 2: Perform feature attribution analysis for monolingual text
echo "Step 2a: Performing feature attribution analysis for monolingual Hindi..."
python scripts/feature_attribution.py \
    --input_file $OUTPUT_DIR/monolingual_hindi_samples.csv \
    --model_path $OUTPUT_DIR/model/best_model \
    --output_dir $OUTPUT_DIR/feature_attribution/monolingual \
    --method lime \
    --num_samples 10

# Step 3: Perform feature attribution analysis for code-switched text
echo "Step 2b: Performing feature attribution analysis for code-switched text..."
python scripts/feature_attribution.py \
    --input_file $OUTPUT_DIR/code_switched_samples.csv \
    --model_path $OUTPUT_DIR/model/best_model \
    --output_dir $OUTPUT_DIR/feature_attribution/code_switched \
    --method lime \
    --num_samples 10

# Step 4: Compare toxicity patterns between monolingual and code-switched text
echo "Step 3: Comparing toxicity patterns between language categories..."
python scripts/compare_feature_importance.py \
    --monolingual_dir $OUTPUT_DIR/feature_attribution/monolingual \
    --code_switched_dir $OUTPUT_DIR/feature_attribution/code_switched \
    --output_dir $OUTPUT_DIR

echo "Analysis complete! Results saved to $OUTPUT_DIR"
 