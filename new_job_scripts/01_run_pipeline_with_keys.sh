#!/bin/bash

#SBATCH --partition=rome
#SBATCH --job-name=01_full_preprocess
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/tchakravorty/tchakravorty/code-switch-thesis/job_outputs/01_full_preprocess_%j.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# Create output directory if it doesn't exist
mkdir -p new_outputs
mkdir -p job_outputs

# Step 1: Add primary keys to the FULL code-switched data
echo "Adding primary keys to FULL code-switched data..."
python new_python_scripts/add_primary_keys.py \
    --input data/output/hindi/code_switched/compile_hindi.csv \
    --output new_outputs/code_switched_with_keys_full.csv \
    --key_prefix "cs_"

# Step 2: Run language detection (preserves primary keys)
echo "Running language detection on FULL data..."
python new_python_scripts/language_detection.py \
    --input_file new_outputs/code_switched_with_keys_full.csv \
    --output_file new_outputs/language_detection_full.csv

# Step 3: Filter language mix and track filtered sentences
echo "Filtering language mix on FULL data..."
python new_python_scripts/filter_language_mix.py \
    --input new_outputs/language_detection_full.csv \
    --output new_outputs/filtered_output_full.csv

echo "Pre-processing pipeline completed successfully!"
echo "Final filtered data with keys: new_outputs/filtered_output_full.csv"
# Note: Assuming the python scripts handle unique_id and full data correctly. 