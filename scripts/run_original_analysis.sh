#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=run_original_analysis
#SBATCH --mem=32G
#SBATCH --output=outputs/run_original_analysis.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# Set Hugging Face token
export HUGGING_FACE_HUB_TOKEN=$(cat ~/.huggingface/token)

# Define paths
HINDI_PROMPTS="data/extracted_prompts/train_hi.txt"
ENGLISH_PROMPTS="data/extracted_prompts/train_en.txt"
OUTPUT_DIR="data/output/original_analysis"

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/hindi
mkdir -p $OUTPUT_DIR/english
mkdir -p $OUTPUT_DIR/model_responses
mkdir -p $OUTPUT_DIR/toxicity_analysis

# Step 1: Add primary keys to the original prompts
echo "Step 1: Adding primary keys to original prompts..."
python src/add_primary_key.py \
    --hindi $HINDI_PROMPTS \
    --english $ENGLISH_PROMPTS \
    --output_dir $OUTPUT_DIR

# Step 2: Detect languages and count words in Hindi prompts
echo "Step 2: Detecting languages in Hindi prompts..."
python src/language_detection_with_id.py \
    --input_file $OUTPUT_DIR/hindi_prompts_with_id.csv \
    --output_file $OUTPUT_DIR/hindi/language_detection.csv \
    --fasttext_model lid.176.bin

# Step 3: Detect languages and count words in English prompts
echo "Step 3: Detecting languages in English prompts..."
python src/language_detection_with_id.py \
    --input_file $OUTPUT_DIR/english_prompts_with_id.csv \
    --output_file $OUTPUT_DIR/english/language_detection.csv \
    --fasttext_model lid.176.bin

# Step 4: Filter Hindi prompts to keep only balanced code-switched sentences
echo "Step 4: Filtering Hindi prompts..."
python src/filter_language_mix_with_id.py \
    --input $OUTPUT_DIR/hindi/language_detection.csv \
    --output $OUTPUT_DIR/hindi/filtered_output.csv

# Step 5: Filter English prompts to keep only balanced code-switched sentences
echo "Step 5: Filtering English prompts..."
python src/filter_language_mix_with_id.py \
    --input $OUTPUT_DIR/english/language_detection.csv \
    --output $OUTPUT_DIR/english/filtered_output.csv

# Step 6: Generate LLaMA responses for Hindi prompts
echo "Step 6: Generating LLaMA responses for Hindi prompts..."
python src/generate_model_responses_with_id.py \
    --input $OUTPUT_DIR/hindi/filtered_output.csv \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --output $OUTPUT_DIR/model_responses/llama_hindi_responses.csv \
    --max_tokens 100 \
    --temperature 0.7

# Step 7: Generate LLaMA responses for English prompts
echo "Step 7: Generating LLaMA responses for English prompts..."
python src/generate_model_responses_with_id.py \
    --input $OUTPUT_DIR/english/filtered_output.csv \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --output $OUTPUT_DIR/model_responses/llama_english_responses.csv \
    --max_tokens 100 \
    --temperature 0.7

# Step 8: Generate Aya responses for Hindi prompts
echo "Step 8: Generating Aya responses for Hindi prompts..."
python src/generate_model_responses_with_id.py \
    --input $OUTPUT_DIR/hindi/filtered_output.csv \
    --model "CohereForAI/aya-23-8B" \
    --output $OUTPUT_DIR/model_responses/aya_hindi_responses.csv \
    --max_tokens 100 \
    --temperature 0.7

# Step 9: Generate Aya responses for English prompts
echo "Step 9: Generating Aya responses for English prompts..."
python src/generate_model_responses_with_id.py \
    --input $OUTPUT_DIR/english/filtered_output.csv \
    --model "CohereForAI/aya-23-8B" \
    --output $OUTPUT_DIR/model_responses/aya_english_responses.csv \
    --max_tokens 100 \
    --temperature 0.7

# Step 10: Analyze toxicity of Hindi prompts
echo "Step 10: Analyzing toxicity of Hindi prompts..."
python src/analyze_toxicity_with_id.py \
    --input $OUTPUT_DIR/hindi/filtered_output.csv \
    --output $OUTPUT_DIR/toxicity_analysis/hindi_prompt_toxicity.csv \
    --progress_file $OUTPUT_DIR/toxicity_analysis/hindi_prompt_toxicity_progress.csv \
    --batch_size 5

# Step 11: Analyze toxicity of English prompts
echo "Step 11: Analyzing toxicity of English prompts..."
python src/analyze_toxicity_with_id.py \
    --input $OUTPUT_DIR/english/filtered_output.csv \
    --output $OUTPUT_DIR/toxicity_analysis/english_prompt_toxicity.csv \
    --progress_file $OUTPUT_DIR/toxicity_analysis/english_prompt_toxicity_progress.csv \
    --batch_size 5

# Step 12: Analyze toxicity of LLaMA responses for Hindi prompts
echo "Step 12: Analyzing toxicity of LLaMA responses for Hindi prompts..."
python src/analyze_toxicity_with_id.py \
    --input $OUTPUT_DIR/model_responses/llama_hindi_responses.csv \
    --output $OUTPUT_DIR/toxicity_analysis/llama_hindi_toxicity.csv \
    --progress_file $OUTPUT_DIR/toxicity_analysis/llama_hindi_toxicity_progress.csv \
    --batch_size 5

# Step 13: Analyze toxicity of LLaMA responses for English prompts
echo "Step 13: Analyzing toxicity of LLaMA responses for English prompts..."
python src/analyze_toxicity_with_id.py \
    --input $OUTPUT_DIR/model_responses/llama_english_responses.csv \
    --output $OUTPUT_DIR/toxicity_analysis/llama_english_toxicity.csv \
    --progress_file $OUTPUT_DIR/toxicity_analysis/llama_english_toxicity_progress.csv \
    --batch_size 5

# Step 14: Analyze toxicity of Aya responses for Hindi prompts
echo "Step 14: Analyzing toxicity of Aya responses for Hindi prompts..."
python src/analyze_toxicity_with_id.py \
    --input $OUTPUT_DIR/model_responses/aya_hindi_responses.csv \
    --output $OUTPUT_DIR/toxicity_analysis/aya_hindi_toxicity.csv \
    --progress_file $OUTPUT_DIR/toxicity_analysis/aya_hindi_toxicity_progress.csv \
    --batch_size 5

# Step 15: Analyze toxicity of Aya responses for English prompts
echo "Step 15: Analyzing toxicity of Aya responses for English prompts..."
python src/analyze_toxicity_with_id.py \
    --input $OUTPUT_DIR/model_responses/aya_english_responses.csv \
    --output $OUTPUT_DIR/toxicity_analysis/aya_english_toxicity.csv \
    --progress_file $OUTPUT_DIR/toxicity_analysis/aya_english_toxicity_progress.csv \
    --batch_size 5

# Step 16: Compare toxicity for Hindi prompts and LLaMA responses
echo "Step 16: Comparing toxicity for Hindi prompts and LLaMA responses..."
python src/compare_toxicity_with_id.py \
    --prompt_file $OUTPUT_DIR/toxicity_analysis/hindi_prompt_toxicity.csv \
    --response_file $OUTPUT_DIR/toxicity_analysis/llama_hindi_toxicity.csv \
    --output_dir $OUTPUT_DIR/toxicity_analysis/hindi_llama_comparison \
    --model_name "LLaMA"

# Step 17: Compare toxicity for Hindi prompts and Aya responses
echo "Step 17: Comparing toxicity for Hindi prompts and Aya responses..."
python src/compare_toxicity_with_id.py \
    --prompt_file $OUTPUT_DIR/toxicity_analysis/hindi_prompt_toxicity.csv \
    --response_file $OUTPUT_DIR/toxicity_analysis/aya_hindi_toxicity.csv \
    --output_dir $OUTPUT_DIR/toxicity_analysis/hindi_aya_comparison \
    --model_name "Aya"

# Step 18: Compare toxicity for English prompts and LLaMA responses
echo "Step 18: Comparing toxicity for English prompts and LLaMA responses..."
python src/compare_toxicity_with_id.py \
    --prompt_file $OUTPUT_DIR/toxicity_analysis/english_prompt_toxicity.csv \
    --response_file $OUTPUT_DIR/toxicity_analysis/llama_english_toxicity.csv \
    --output_dir $OUTPUT_DIR/toxicity_analysis/english_llama_comparison \
    --model_name "LLaMA"

# Step 19: Compare toxicity for English prompts and Aya responses
echo "Step 19: Comparing toxicity for English prompts and Aya responses..."
python src/compare_toxicity_with_id.py \
    --prompt_file $OUTPUT_DIR/toxicity_analysis/english_prompt_toxicity.csv \
    --response_file $OUTPUT_DIR/toxicity_analysis/aya_english_toxicity.csv \
    --output_dir $OUTPUT_DIR/toxicity_analysis/english_aya_comparison \
    --model_name "Aya"

echo "Analysis complete! Results saved to $OUTPUT_DIR" 