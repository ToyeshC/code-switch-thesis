#!/bin/bash

#SBATCH --partition=rome
#SBATCH --job-name=3_processing_ezswitch_generation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=final_job_outputs/3_processing_ezswitch_generation_%A.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# Create necessary directories
mkdir -p final_outputs

# Define the single output file
OUTPUT_FILE="final_outputs/filtered_output.csv"

# Step 1: Add primary keys to the FULL code-switched data
echo "Adding primary keys to FULL code-switched data..."
python final_python_scripts/add_primary_keys.py \
    --input final_data/output/compile_hindi.csv \
    --output "$OUTPUT_FILE" \
    --key_prefix "cs_"

# Step 2: Run language detection (preserves primary keys and overwrites the file)
echo "Running language detection on FULL data..."
python final_python_scripts/language_detection.py \
    --input_file "$OUTPUT_FILE" \
    --output_file "$OUTPUT_FILE"

# Step 3: Filter language mix and track filtered sentences (overwrites the file again)
echo "Filtering language mix on FULL data..."
python final_python_scripts/filter_language_mix.py \
    --input "$OUTPUT_FILE" \
    --output "$OUTPUT_FILE"

echo "Pre-processing pipeline completed successfully!"
echo "Final filtered data is available at: $OUTPUT_FILE"