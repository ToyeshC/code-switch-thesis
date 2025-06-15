#!/bin/bash

#SBATCH --partition=rome
#SBATCH --job-name=100_filter_negative_tweets
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=final_job_outputs/100_filter_negative_tweets_%A.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# Define file paths
INPUT_FILE="final_outputs/processed_hinglish.csv"
OUTPUT_FILE="final_outputs/filtered_negative_tweets.csv"

# Create necessary directories
mkdir -p final_outputs

# Check for input file
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found!"
    exit 1
fi

# Filter the CSV to keep only rows with 'negative' sentiment
echo "Filtering for negative sentiment tweets..."
python final_python_scripts/filter_by_sentiment.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --column "sentiment" \
    --value "negative"

echo "Pipeline completed successfully!"
echo "Filtered negative tweets are available at: $OUTPUT_FILE"