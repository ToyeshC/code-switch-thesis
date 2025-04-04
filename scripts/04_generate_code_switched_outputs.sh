#!/bin/bash

#SBATCH --time=8:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=code_switch
#SBATCH --mem=32G
#SBATCH --output=outputs/04_generate_code_switched_outputs.out

# Load necessary modules
module load 2023
module load Miniconda3/23.5.2-0
source ~/.bashrc
conda activate code-switch

# Define languages - can be modified for other language pairs
BASE_LANG="hindi"
SOURCE_LANG="english"

# Define language codes for file naming
BASE_LANG_CODE="hi"
SOURCE_LANG_CODE="en"

# Set Hugging Face token
# Uncomment and replace with your token if needed
# export HUGGING_FACE_HUB_TOKEN="your_huggingface_token_here"

# Set the number of outputs to generate
MAX_OUTPUTS=100

# Define paths
INPUT_DIR="data/extracted_prompts"
TRANSLATE_DIR="data/translate_api_outputs"
ALIGN_DIR="data/alignments"
OUTPUT_DIR="data/output/${BASE_LANG}"
TEMP_DIR="data/temp_limited"

# Create output directory and temp directory for limited files
mkdir -p $OUTPUT_DIR
mkdir -p $TEMP_DIR

# Create limited versions of all input files (only first 200 lines)
echo "Creating limited versions of input files (${MAX_OUTPUTS} lines)..."
head -n $MAX_OUTPUTS $INPUT_DIR/train_${SOURCE_LANG}.txt > $TEMP_DIR/train_${SOURCE_LANG}_raw.txt
head -n $MAX_OUTPUTS $INPUT_DIR/train_${BASE_LANG}.txt > $TEMP_DIR/train_${BASE_LANG}_raw.txt
head -n $MAX_OUTPUTS $TRANSLATE_DIR/train_${SOURCE_LANG}.txt > $TEMP_DIR/train_${SOURCE_LANG}_translated_raw.txt
head -n $MAX_OUTPUTS $TRANSLATE_DIR/train_${BASE_LANG}.txt > $TEMP_DIR/train_${BASE_LANG}_translated_raw.txt

# Fix JSON formatting issues in translated files
echo "Fixing JSON format issues in translated files..."
python src/fix_json_inputs.py --input $TEMP_DIR/train_${SOURCE_LANG}_translated_raw.txt --output $TEMP_DIR/train_${SOURCE_LANG}_translated_raw.fixed.txt
python src/fix_json_inputs.py --input $TEMP_DIR/train_${BASE_LANG}_translated_raw.txt --output $TEMP_DIR/train_${BASE_LANG}_translated_raw.fixed.txt

# Use the fixed files for further processing
mv $TEMP_DIR/train_${SOURCE_LANG}_translated_raw.fixed.txt $TEMP_DIR/train_${SOURCE_LANG}_translated_raw.txt
mv $TEMP_DIR/train_${BASE_LANG}_translated_raw.fixed.txt $TEMP_DIR/train_${BASE_LANG}_translated_raw.txt

# Extract the "Prompt" field from each JSON line using the Python script
echo "Extracting prompts from JSON input..."
python src/extract_prompts_from_json.py --input $TEMP_DIR/train_${SOURCE_LANG}_raw.txt --output $TEMP_DIR/train_${SOURCE_LANG}.txt
python src/extract_prompts_from_json.py --input $TEMP_DIR/train_${BASE_LANG}_raw.txt --output $TEMP_DIR/train_${BASE_LANG}.txt
python src/extract_prompts_from_json.py --input $TEMP_DIR/train_${SOURCE_LANG}_translated_raw.txt --output $TEMP_DIR/train_${SOURCE_LANG}_translated.txt
python src/extract_prompts_from_json.py --input $TEMP_DIR/train_${BASE_LANG}_translated_raw.txt --output $TEMP_DIR/train_${BASE_LANG}_translated.txt

# Process alignment files to ensure consistent line counts
# Alignment files typically have a header line and then one alignment per line
echo "Processing alignment files..."

# For gold align file
if [ -f $ALIGN_DIR/${SOURCE_LANG_CODE}-${BASE_LANG_CODE}_align_gold.txt ]; then
    # Copy header line
    head -n 1 $ALIGN_DIR/${SOURCE_LANG_CODE}-${BASE_LANG_CODE}_align_gold.txt > $TEMP_DIR/${SOURCE_LANG_CODE}-${BASE_LANG_CODE}_align_gold.txt
    # Process remaining lines after header (up to MAX_OUTPUTS)
    tail -n +2 $ALIGN_DIR/${SOURCE_LANG_CODE}-${BASE_LANG_CODE}_align_gold.txt | head -n $MAX_OUTPUTS >> $TEMP_DIR/${SOURCE_LANG_CODE}-${BASE_LANG_CODE}_align_gold.txt
    # If we have fewer than MAX_OUTPUTS alignments, pad with empty alignments
    GOLD_LINES=$(wc -l < $TEMP_DIR/${SOURCE_LANG_CODE}-${BASE_LANG_CODE}_align_gold.txt)
    GOLD_NEEDED=$((MAX_OUTPUTS + 1 - GOLD_LINES))
    if [ $GOLD_NEEDED -gt 0 ]; then
        # Repeat the last line as padding (or create empty alignment if needed)
        LAST_LINE=$(tail -n 1 $TEMP_DIR/${SOURCE_LANG_CODE}-${BASE_LANG_CODE}_align_gold.txt)
        for i in $(seq 1 $GOLD_NEEDED); do
            echo "$LAST_LINE" >> $TEMP_DIR/${SOURCE_LANG_CODE}-${BASE_LANG_CODE}_align_gold.txt
        done
    fi
fi

# For silver align file
if [ -f $ALIGN_DIR/${SOURCE_LANG_CODE}-${BASE_LANG_CODE}_align_silver.txt ]; then
    # Copy header line
    head -n 1 $ALIGN_DIR/${SOURCE_LANG_CODE}-${BASE_LANG_CODE}_align_silver.txt > $TEMP_DIR/${SOURCE_LANG_CODE}-${BASE_LANG_CODE}_align_silver.txt
    # Process remaining lines after header (up to MAX_OUTPUTS)
    tail -n +2 $ALIGN_DIR/${SOURCE_LANG_CODE}-${BASE_LANG_CODE}_align_silver.txt | head -n $MAX_OUTPUTS >> $TEMP_DIR/${SOURCE_LANG_CODE}-${BASE_LANG_CODE}_align_silver.txt
    # If we have fewer than MAX_OUTPUTS alignments, pad with empty alignments
    SILVER_LINES=$(wc -l < $TEMP_DIR/${SOURCE_LANG_CODE}-${BASE_LANG_CODE}_align_silver.txt)
    SILVER_NEEDED=$((MAX_OUTPUTS + 1 - SILVER_LINES))
    if [ $SILVER_NEEDED -gt 0 ]; then
        # Repeat the last line as padding (or create empty alignment if needed)
        LAST_LINE=$(tail -n 1 $TEMP_DIR/${SOURCE_LANG_CODE}-${BASE_LANG_CODE}_align_silver.txt)
        for i in $(seq 1 $SILVER_NEEDED); do
            echo "$LAST_LINE" >> $TEMP_DIR/${SOURCE_LANG_CODE}-${BASE_LANG_CODE}_align_silver.txt
        done
    fi
fi

# Make sure all our extracted text files have the same number of lines
for file in $TEMP_DIR/train_${SOURCE_LANG}.txt $TEMP_DIR/train_${BASE_LANG}.txt $TEMP_DIR/train_${SOURCE_LANG}_translated.txt $TEMP_DIR/train_${BASE_LANG}_translated.txt; do
    LINES=$(wc -l < "$file")
    if [ $LINES -lt $MAX_OUTPUTS ]; then
        NEEDED=$((MAX_OUTPUTS - LINES))
        echo "Adding $NEEDED placeholder lines to $file"
        for i in $(seq 1 $NEEDED); do
            echo "[PLACEHOLDER_TEXT]" >> "$file"
        done
    fi
done

# Generate a fixed version of the inference.py script and backup the original
ORIGINAL_INFERENCE="ezswitch/src/inference.py"
BACKUP_INFERENCE="ezswitch/src/inference.py.backup"

# Only create a backup if one doesn't exist yet
if [ ! -f "$BACKUP_INFERENCE" ]; then
    echo "Backing up original inference.py script..."
    cp $ORIGINAL_INFERENCE $BACKUP_INFERENCE
fi

# Fix and replace the original inference.py script
echo "Fixing and updating inference.py script..."
python src/fix_inference.py --script_path $ORIGINAL_INFERENCE --output_path "${ORIGINAL_INFERENCE}.tmp"
mv "${ORIGINAL_INFERENCE}.tmp" $ORIGINAL_INFERENCE

# Run each model using the fixed inference script
echo "Generating ${MAX_OUTPUTS} code-switched outputs using Aya model..."
python $ORIGINAL_INFERENCE \
    --src $TEMP_DIR/train_${SOURCE_LANG}.txt \
    --tgt $TEMP_DIR/train_${BASE_LANG}.txt \
    --src_translated $TEMP_DIR/train_${SOURCE_LANG}_translated.txt \
    --tgt_translated $TEMP_DIR/train_${BASE_LANG}_translated.txt \
    --gold_align $TEMP_DIR/${SOURCE_LANG_CODE}-${BASE_LANG_CODE}_align_gold.txt \
    --silver_src_align $TEMP_DIR/${SOURCE_LANG_CODE}-${BASE_LANG_CODE}_align_silver.txt \
    --silver_tgt_align $TEMP_DIR/${SOURCE_LANG_CODE}-${BASE_LANG_CODE}_align_silver.txt \
    --model_id "CohereForAI/aya-23-8B" \
    --output $OUTPUT_DIR/full_aya.csv

echo "Generating ${MAX_OUTPUTS} code-switched outputs using LLaMA model..."
python $ORIGINAL_INFERENCE \
    --src $TEMP_DIR/train_${SOURCE_LANG}.txt \
    --tgt $TEMP_DIR/train_${BASE_LANG}.txt \
    --src_translated $TEMP_DIR/train_${SOURCE_LANG}_translated.txt \
    --tgt_translated $TEMP_DIR/train_${BASE_LANG}_translated.txt \
    --gold_align $TEMP_DIR/${SOURCE_LANG_CODE}-${BASE_LANG_CODE}_align_gold.txt \
    --silver_src_align $TEMP_DIR/${SOURCE_LANG_CODE}-${BASE_LANG_CODE}_align_silver.txt \
    --silver_tgt_align $TEMP_DIR/${SOURCE_LANG_CODE}-${BASE_LANG_CODE}_align_silver.txt \
    --model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
    --output $OUTPUT_DIR/full_llama.csv

# Clean up temporary files if needed
# echo "Cleaning up temporary files..."
# rm -rf $TEMP_DIR

echo "Code-switched output generation complete. Results saved to $OUTPUT_DIR" 