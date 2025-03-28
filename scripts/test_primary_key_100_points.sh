#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=test_primary_key
#SBATCH --mem=32G
#SBATCH --output=outputs/test_primary_key_100_points.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# Set Hugging Face token
export HUGGING_FACE_HUB_TOKEN=$(cat ~/.huggingface/token)

# Define paths
HINDI_PROMPTS="data/extracted_prompts/train_hi.txt"
ENGLISH_PROMPTS="data/extracted_prompts/train_en.txt"
OUTPUT_DIR="data/output/test_primary_key_100"

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/hindi
mkdir -p $OUTPUT_DIR/english
mkdir -p $OUTPUT_DIR/model_responses
mkdir -p $OUTPUT_DIR/toxicity_analysis

# Extract first 100 lines from original prompts
echo "Extracting first 100 lines from Hindi prompts..."
head -n 100 $HINDI_PROMPTS > $OUTPUT_DIR/hindi_100.txt

echo "Extracting first 100 lines from English prompts..."
head -n 100 $ENGLISH_PROMPTS > $OUTPUT_DIR/english_100.txt

# Step 1: Add primary keys to the original prompts
echo "Step 1: Adding primary keys to original prompts..."
python src/add_primary_key.py \
    --hindi $OUTPUT_DIR/hindi_100.txt \
    --english $OUTPUT_DIR/english_100.txt \
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

# Step 6: Generate LLaMA responses for Hindi prompts (sample 10 for speed)
echo "Step 6: Generating LLaMA responses for Hindi prompts..."
# First, extract 10 random filtered prompts for quick testing
python -c "
import pandas as pd
df = pd.read_csv('$OUTPUT_DIR/hindi/filtered_output.csv')
if len(df) > 10:
    df = df.sample(10, random_state=42)
df.to_csv('$OUTPUT_DIR/hindi/sample_10_filtered.csv', index=False)
print(f'Sampled {len(df)} prompts for quick testing')
"

python src/generate_model_responses_with_id.py \
    --input $OUTPUT_DIR/hindi/sample_10_filtered.csv \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --output $OUTPUT_DIR/model_responses/llama_hindi_responses.csv \
    --max_tokens 100 \
    --temperature 0.7

# Step 7: Generate LLaMA responses for English prompts (sample 10 for speed)
echo "Step 7: Generating LLaMA responses for English prompts..."
# First, extract 10 random filtered prompts for quick testing
python -c "
import pandas as pd
df = pd.read_csv('$OUTPUT_DIR/english/filtered_output.csv')
if len(df) > 10:
    df = df.sample(10, random_state=42)
df.to_csv('$OUTPUT_DIR/english/sample_10_filtered.csv', index=False)
print(f'Sampled {len(df)} prompts for quick testing')
"

python src/generate_model_responses_with_id.py \
    --input $OUTPUT_DIR/english/sample_10_filtered.csv \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --output $OUTPUT_DIR/model_responses/llama_english_responses.csv \
    --max_tokens 100 \
    --temperature 0.7

# Step 8: Generate Aya responses for Hindi prompts (using same 10 samples)
echo "Step 8: Generating Aya responses for Hindi prompts..."
python src/generate_model_responses_with_id.py \
    --input $OUTPUT_DIR/hindi/sample_10_filtered.csv \
    --model "CohereForAI/aya-23-8B" \
    --output $OUTPUT_DIR/model_responses/aya_hindi_responses.csv \
    --max_tokens 100 \
    --temperature 0.7

# Step 9: Generate Aya responses for English prompts (using same 10 samples)
echo "Step 9: Generating Aya responses for English prompts..."
python src/generate_model_responses_with_id.py \
    --input $OUTPUT_DIR/english/sample_10_filtered.csv \
    --model "CohereForAI/aya-23-8B" \
    --output $OUTPUT_DIR/model_responses/aya_english_responses.csv \
    --max_tokens 100 \
    --temperature 0.7

# Step 10: Analyze toxicity of Hindi prompts (using 10 samples)
echo "Step 10: Analyzing toxicity of Hindi prompts..."
python src/analyze_toxicity_with_id.py \
    --input $OUTPUT_DIR/hindi/sample_10_filtered.csv \
    --output $OUTPUT_DIR/toxicity_analysis/hindi_prompt_toxicity.csv \
    --progress_file $OUTPUT_DIR/toxicity_analysis/hindi_prompt_toxicity_progress.csv \
    --batch_size 2

# Step 11: Analyze toxicity of English prompts (using 10 samples)
echo "Step 11: Analyzing toxicity of English prompts..."
python src/analyze_toxicity_with_id.py \
    --input $OUTPUT_DIR/english/sample_10_filtered.csv \
    --output $OUTPUT_DIR/toxicity_analysis/english_prompt_toxicity.csv \
    --progress_file $OUTPUT_DIR/toxicity_analysis/english_prompt_toxicity_progress.csv \
    --batch_size 2

# Step 12: Analyze toxicity of LLaMA responses for Hindi prompts
echo "Step 12: Analyzing toxicity of LLaMA responses for Hindi prompts..."
python src/analyze_toxicity_with_id.py \
    --input $OUTPUT_DIR/model_responses/llama_hindi_responses.csv \
    --output $OUTPUT_DIR/toxicity_analysis/llama_hindi_toxicity.csv \
    --progress_file $OUTPUT_DIR/toxicity_analysis/llama_hindi_toxicity_progress.csv \
    --batch_size 2

# Step 13: Analyze toxicity of LLaMA responses for English prompts
echo "Step 13: Analyzing toxicity of LLaMA responses for English prompts..."
python src/analyze_toxicity_with_id.py \
    --input $OUTPUT_DIR/model_responses/llama_english_responses.csv \
    --output $OUTPUT_DIR/toxicity_analysis/llama_english_toxicity.csv \
    --progress_file $OUTPUT_DIR/toxicity_analysis/llama_english_toxicity_progress.csv \
    --batch_size 2

# Step 14: Analyze toxicity of Aya responses for Hindi prompts
echo "Step 14: Analyzing toxicity of Aya responses for Hindi prompts..."
python src/analyze_toxicity_with_id.py \
    --input $OUTPUT_DIR/model_responses/aya_hindi_responses.csv \
    --output $OUTPUT_DIR/toxicity_analysis/aya_hindi_toxicity.csv \
    --progress_file $OUTPUT_DIR/toxicity_analysis/aya_hindi_toxicity_progress.csv \
    --batch_size 2

# Step 15: Analyze toxicity of Aya responses for English prompts
echo "Step 15: Analyzing toxicity of Aya responses for English prompts..."
python src/analyze_toxicity_with_id.py \
    --input $OUTPUT_DIR/model_responses/aya_english_responses.csv \
    --output $OUTPUT_DIR/toxicity_analysis/aya_english_toxicity.csv \
    --progress_file $OUTPUT_DIR/toxicity_analysis/aya_english_toxicity_progress.csv \
    --batch_size 2

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

# Step 20: Create verification report to check primary key consistency
echo "Step 20: Creating verification report for primary key tracking..."
python -c "
import pandas as pd
import os
import glob
import json

# Function to check primary keys
def check_primary_keys(file_path, file_desc):
    if not os.path.exists(file_path):
        return {'file': file_path, 'description': file_desc, 'status': 'File not found', 'prompt_ids': []}
    
    try:
        df = pd.read_csv(file_path)
        if 'prompt_id' not in df.columns:
            return {'file': file_path, 'description': file_desc, 'status': 'No prompt_id column', 'prompt_ids': []}
        
        prompt_ids = sorted(df['prompt_id'].unique().tolist())
        return {
            'file': file_path, 
            'description': file_desc, 
            'status': 'OK', 
            'count': len(prompt_ids),
            'prompt_ids': prompt_ids
        }
    except Exception as e:
        return {'file': file_path, 'description': file_desc, 'status': f'Error: {str(e)}', 'prompt_ids': []}

# Base directory
base_dir = '$OUTPUT_DIR'

# Files to check
files_to_check = [
    # Original data with IDs
    (os.path.join(base_dir, 'hindi_prompts_with_id.csv'), 'Hindi prompts with ID'),
    (os.path.join(base_dir, 'english_prompts_with_id.csv'), 'English prompts with ID'),
    
    # Language detection
    (os.path.join(base_dir, 'hindi', 'language_detection.csv'), 'Hindi language detection'),
    (os.path.join(base_dir, 'english', 'language_detection.csv'), 'English language detection'),
    
    # Filtered outputs
    (os.path.join(base_dir, 'hindi', 'filtered_output.csv'), 'Hindi filtered output'),
    (os.path.join(base_dir, 'english', 'filtered_output.csv'), 'English filtered output'),
    
    # Sample 10 filtered
    (os.path.join(base_dir, 'hindi', 'sample_10_filtered.csv'), 'Hindi sample 10 filtered'),
    (os.path.join(base_dir, 'english', 'sample_10_filtered.csv'), 'English sample 10 filtered'),
    
    # Model responses
    (os.path.join(base_dir, 'model_responses', 'llama_hindi_responses.csv'), 'LLaMA Hindi responses'),
    (os.path.join(base_dir, 'model_responses', 'llama_english_responses.csv'), 'LLaMA English responses'),
    (os.path.join(base_dir, 'model_responses', 'aya_hindi_responses.csv'), 'Aya Hindi responses'),
    (os.path.join(base_dir, 'model_responses', 'aya_english_responses.csv'), 'Aya English responses'),
    
    # Toxicity analysis
    (os.path.join(base_dir, 'toxicity_analysis', 'hindi_prompt_toxicity.csv'), 'Hindi prompt toxicity'),
    (os.path.join(base_dir, 'toxicity_analysis', 'english_prompt_toxicity.csv'), 'English prompt toxicity'),
    (os.path.join(base_dir, 'toxicity_analysis', 'llama_hindi_toxicity.csv'), 'LLaMA Hindi toxicity'),
    (os.path.join(base_dir, 'toxicity_analysis', 'llama_english_toxicity.csv'), 'LLaMA English toxicity'),
    (os.path.join(base_dir, 'toxicity_analysis', 'aya_hindi_toxicity.csv'), 'Aya Hindi toxicity'),
    (os.path.join(base_dir, 'toxicity_analysis', 'aya_english_toxicity.csv'), 'Aya English toxicity'),
]

# Check each file
results = [check_primary_keys(file_path, file_desc) for file_path, file_desc in files_to_check]

# Save results
report_file = os.path.join(base_dir, 'primary_key_verification.json')
with open(report_file, 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print(f'Primary key verification report saved to {report_file}')
print('\\nSummary:')
for result in results:
    print(f\"{result['description']}: {result['status']} - {'N/A' if 'count' not in result else result['count']} IDs\")

# Verify tracking consistency across paired files
print('\\nVerifying primary key tracking consistency between paired files:')
# Check Hindi model responses against Hindi filtered
if os.path.exists(os.path.join(base_dir, 'hindi', 'sample_10_filtered.csv')) and os.path.exists(os.path.join(base_dir, 'model_responses', 'llama_hindi_responses.csv')):
    hindi_filtered = pd.read_csv(os.path.join(base_dir, 'hindi', 'sample_10_filtered.csv'))
    llama_hindi = pd.read_csv(os.path.join(base_dir, 'model_responses', 'llama_hindi_responses.csv'))
    hindi_ids_filtered = set(hindi_filtered['prompt_id'])
    hindi_ids_llama = set(llama_hindi['prompt_id'])
    print(f'Hindi filtered -> LLaMA Hindi: {hindi_ids_filtered == hindi_ids_llama} ({len(hindi_ids_filtered)} IDs in filtered, {len(hindi_ids_llama)} IDs in LLaMA)')

# Check English model responses against English filtered
if os.path.exists(os.path.join(base_dir, 'english', 'sample_10_filtered.csv')) and os.path.exists(os.path.join(base_dir, 'model_responses', 'llama_english_responses.csv')):
    english_filtered = pd.read_csv(os.path.join(base_dir, 'english', 'sample_10_filtered.csv'))
    llama_english = pd.read_csv(os.path.join(base_dir, 'model_responses', 'llama_english_responses.csv'))
    english_ids_filtered = set(english_filtered['prompt_id'])
    english_ids_llama = set(llama_english['prompt_id'])
    print(f'English filtered -> LLaMA English: {english_ids_filtered == english_ids_llama} ({len(english_ids_filtered)} IDs in filtered, {len(english_ids_llama)} IDs in LLaMA)')

# Check toxicity analysis
# Hindi prompt vs LLaMA Hindi
if os.path.exists(os.path.join(base_dir, 'toxicity_analysis', 'hindi_prompt_toxicity.csv')) and os.path.exists(os.path.join(base_dir, 'toxicity_analysis', 'llama_hindi_toxicity.csv')):
    hindi_prompt_tox = pd.read_csv(os.path.join(base_dir, 'toxicity_analysis', 'hindi_prompt_toxicity.csv'))
    llama_hindi_tox = pd.read_csv(os.path.join(base_dir, 'toxicity_analysis', 'llama_hindi_toxicity.csv'))
    hindi_prompt_ids = set(hindi_prompt_tox['prompt_id'])
    llama_hindi_ids = set(llama_hindi_tox['prompt_id'])
    common_ids = hindi_prompt_ids.intersection(llama_hindi_ids)
    print(f'Hindi prompt toxicity ∩ LLaMA Hindi toxicity: {len(common_ids)} common IDs of {len(hindi_prompt_ids)} Hindi and {len(llama_hindi_ids)} LLaMA')
"

echo "Test analysis complete! Results saved to $OUTPUT_DIR"
echo "Check $OUTPUT_DIR/primary_key_verification.json for primary key tracking information" 