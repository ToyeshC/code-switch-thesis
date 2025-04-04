#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --partition=rome
#SBATCH --job-name=extract_data
#SBATCH --mem=16G
#SBATCH --output=outputs/00_extract_rtp_data.out

# Load necessary modules
module load 2023
module load Miniconda3/23.5.2-0
source ~/.bashrc
conda activate code-switch

# Define paths
OUTPUT_DIR="data/output"
RTP_DIR="data/RTP-LX"

# Define languages - can be modified for other language pairs
BASE_LANG="hindi"
SOURCE_LANG="english"

# Define language codes for filenames
BASE_LANG_CODE="HI"
SOURCE_LANG_CODE="EN"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Extracting prompts from RTP-LX dataset..."
python src/add_primary_key.py \
    --base_file $RTP_DIR/RTP_LX_${BASE_LANG_CODE}.json \
    --source_file $RTP_DIR/RTP_LX_${SOURCE_LANG_CODE}.json \
    --base_lang $BASE_LANG \
    --source_lang $SOURCE_LANG \
    --output_dir $OUTPUT_DIR

echo "Data extraction complete. Results saved to $OUTPUT_DIR"
