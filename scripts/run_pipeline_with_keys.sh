#!/bin/bash

#SBATCH --partition=rome
#SBATCH --job-name=run_pipeline_with_keys
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=3:00:00
#SBATCH --output=/home/tchakravorty/tchakravorty/code-switch-thesis/outputs/run_pipeline_with_keys.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# Create output directory if it doesn't exist
mkdir -p data/output

# Step 1: Add primary keys to original Hindi data
echo "Adding primary keys to Hindi data..."
python src/add_primary_keys.py \
    --input data/input/hindi_data.csv \
    --output data/output/hindi_data_with_keys.csv \
    --key_prefix "hi_"

# Step 2: Add primary keys to original English data
echo "Adding primary keys to English data..."
python src/add_primary_keys.py \
    --input data/input/english_data.csv \
    --output data/output/english_data_with_keys.csv \
    --key_prefix "en_"

# Step 3: Propagate keys to code-switched data
echo "Propagating keys to code-switched data..."
python src/add_primary_keys.py \
    --input data/output/generated_sentences.csv \
    --output data/output/generated_sentences_with_keys.csv \
    --propagate

# Step 4: Run language detection (preserves primary keys)
echo "Running language detection..."
python src/language_detection.py \
    --input_file data/output/generated_sentences_with_keys.csv \
    --output_file data/output/language_detection_full.csv

# Step 5: Filter language mix and track filtered sentences
echo "Filtering language mix and tracking filtered sentences..."
python src/filter_language_mix.py \
    --input data/output/language_detection_full.csv \
    --output data/output/filtered_output.csv

echo "Pipeline completed successfully!"
echo "Check data/output/filtered_keys.csv for tracking information about filtered sentences." 