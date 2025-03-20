#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=18_rate_limited_toxicity
#SBATCH --mem=32G
#SBATCH --output=outputs/18_rate_limited_toxicity.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# Define paths
PROMPT_FILE="data/output/hindi/(yes) filtered_output_small.csv"
OUTPUT_DIR="data/output/model_toxicity_analysis"
API_KEY="AIzaSyDf0c2MkAItSv7TBFps65WavRFLP-N275Y"  # The API key from config.py

# Create output directory (in case it doesn't exist yet)
mkdir -p "$OUTPUT_DIR"

# Use the existing response files in the output directory
LLAMA_RESPONSES_FILE="$OUTPUT_DIR/llama_responses.csv"
AYA_RESPONSES_FILE="$OUTPUT_DIR/aya_responses.csv"

echo "==== Step 1: Checking existing model outputs ===="
# Verify the files exist
if [ ! -f "$LLAMA_RESPONSES_FILE" ]; then
  echo "ERROR: LLaMA responses not found at $LLAMA_RESPONSES_FILE"
  exit 1
fi

if [ ! -f "$AYA_RESPONSES_FILE" ]; then
  echo "ERROR: Aya responses not found at $AYA_RESPONSES_FILE"
  exit 1
fi

echo "Found LLaMA responses at: $LLAMA_RESPONSES_FILE"
echo "Found Aya responses at: $AYA_RESPONSES_FILE"

# Skip the preparation step since we already have the files
echo "==== Step 2: Extract Responses for Analysis ===="

# Create files with just the comments for Perspective API
python -c "
import pandas as pd
# Process prompts
prompts_df = pd.read_csv('$PROMPT_FILE')
if 'sentence' in prompts_df.columns:
    pd.DataFrame({'comment': prompts_df['sentence']}).to_csv('$OUTPUT_DIR/prompts_for_analysis.csv', index=False)
else:
    pd.DataFrame({'comment': prompts_df.iloc[:,0]}).to_csv('$OUTPUT_DIR/prompts_for_analysis.csv', index=False)

# Process LLaMA responses
llama_df = pd.read_csv('$LLAMA_RESPONSES_FILE')
if 'response' in llama_df.columns:
    pd.DataFrame({'comment': llama_df['response']}).to_csv('$OUTPUT_DIR/llama_for_analysis.csv', index=False)
else:
    text_cols = [col for col in llama_df.columns if llama_df[col].dtype == 'object']
    if len(text_cols) > 1:
        pd.DataFrame({'comment': llama_df[text_cols[1]]}).to_csv('$OUTPUT_DIR/llama_for_analysis.csv', index=False)
    else:
        pd.DataFrame({'comment': llama_df[text_cols[0]]}).to_csv('$OUTPUT_DIR/llama_for_analysis.csv', index=False)

# Process Aya responses
aya_df = pd.read_csv('$AYA_RESPONSES_FILE')
if 'response' in aya_df.columns:
    pd.DataFrame({'comment': aya_df['response']}).to_csv('$OUTPUT_DIR/aya_for_analysis.csv', index=False)
else:
    text_cols = [col for col in aya_df.columns if aya_df[col].dtype == 'object']
    if len(text_cols) > 1:
        pd.DataFrame({'comment': aya_df[text_cols[1]]}).to_csv('$OUTPUT_DIR/aya_for_analysis.csv', index=False)
    else:
        pd.DataFrame({'comment': aya_df[text_cols[0]]}).to_csv('$OUTPUT_DIR/aya_for_analysis.csv', index=False)
"

# Now run the rate-limited analysis for each dataset
echo "==== Step 3: Analyze Prompts with Rate Limiting ===="
python src/rate_limited_toxicity.py \
    --input "$OUTPUT_DIR/prompts_for_analysis.csv" \
    --output "$OUTPUT_DIR/prompt_toxicity.csv" \
    --api_key "$API_KEY" \
    --rate_limit 1.2 \
    --batch_size 5

echo "==== Step 4: Analyze LLaMA Responses with Rate Limiting ===="
python src/rate_limited_toxicity.py \
    --input "$OUTPUT_DIR/llama_for_analysis.csv" \
    --output "$OUTPUT_DIR/llama_toxicity.csv" \
    --api_key "$API_KEY" \
    --rate_limit 1.2 \
    --batch_size 5

echo "==== Step 5: Analyze Aya Responses with Rate Limiting ===="
python src/rate_limited_toxicity.py \
    --input "$OUTPUT_DIR/aya_for_analysis.csv" \
    --output "$OUTPUT_DIR/aya_toxicity.csv" \
    --api_key "$API_KEY" \
    --rate_limit 1.2 \
    --batch_size 5

echo "==== Step 6: Run Statistical Analysis ===="
python src/analyze_model_toxicity.py \
    --input "$PROMPT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --skip_generation \
    --skip_perspective

echo "Analysis complete! Results saved to $OUTPUT_DIR/analysis" 