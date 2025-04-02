#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=test_primary_key_hindi
#SBATCH --mem=32G
#SBATCH --output=outputs/test_primary_key_hindi.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# Set Hugging Face token
export HUGGING_FACE_HUB_TOKEN=$(cat ~/.huggingface/token)

# Define paths
HINDI_PROMPTS="data/extracted_prompts/train_hi.txt"
ENGLISH_PROMPTS="data/extracted_prompts/train_en.txt" # Needed for compatibility
OUTPUT_DIR="data/output/test_primary_key_hindi"

# Get the Perspective API key directly from config.py
API_KEY=$(grep -oP "PERSPECTIVE_API_KEY = \"\K[^\"]+" src/config.py)
echo "Retrieved Perspective API key from src/config.py"

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/hindi
mkdir -p $OUTPUT_DIR/model_responses
mkdir -p $OUTPUT_DIR/toxicity_analysis

# Extract first 200 lines from original prompts
echo "Extracting first 200 lines from Hindi prompts..."
head -n 200 $HINDI_PROMPTS > $OUTPUT_DIR/hindi_200.txt

# Create dummy English file with same number of lines as Hindi file
echo "Creating dummy English file with matching line count..."
# Count lines in Hindi file
HINDI_LINES=$(wc -l < $OUTPUT_DIR/hindi_200.txt)
# Create English dummy file with same number of lines (filled with "Dummy English text")
for ((i=1; i<=$HINDI_LINES; i++)); do
    echo "Dummy English text $i" >> $OUTPUT_DIR/english_dummy.txt
done
echo "Created dummy English file with $HINDI_LINES lines to match Hindi file"

# Step 1: Add primary keys to the original prompts
echo "Step 1: Adding primary keys to original prompts..."
python src/add_primary_key.py \
    --hindi $OUTPUT_DIR/hindi_200.txt \
    --english $OUTPUT_DIR/english_dummy.txt \
    --output_dir $OUTPUT_DIR

# Step 2: Detect languages and count words in Hindi prompts
echo "Step 2: Detecting languages in Hindi prompts..."
python src/language_detection_with_id.py \
    --input_file $OUTPUT_DIR/hindi_prompts_with_id.csv \
    --output_file $OUTPUT_DIR/hindi/language_detection.csv \
    --fasttext_model lid.176.bin

# Step 3: Filter Hindi prompts to keep only balanced code-switched sentences
echo "Step 3: Filtering Hindi prompts..."
python src/filter_language_mix_with_id.py \
    --input $OUTPUT_DIR/hindi/language_detection.csv \
    --output $OUTPUT_DIR/hindi/filtered_output.csv

# Step 4: Generate LLaMA 3 (8B) responses for Hindi prompts (sample 5 for speed)
echo "Step 4: Generating LLaMA 3 (8B) responses for Hindi prompts..."
# Previously we sampled 5 prompts, now use all filtered prompts
# Copy filtered output for use in generation
cp $OUTPUT_DIR/hindi/filtered_output.csv $OUTPUT_DIR/hindi/all_filtered.csv
echo "Using all filtered prompts for generation instead of sampling"

python src/generate_model_responses_with_id.py \
    --input $OUTPUT_DIR/hindi/all_filtered.csv \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --output $OUTPUT_DIR/model_responses/llama3_8b_hindi_responses.csv \
    --max_tokens 100 \
    --temperature 0.7

# Step 5: Generate LLaMA 3.1 (8B) responses for Hindi prompts
echo "Step 5: Generating LLaMA 3.1 (8B) responses for Hindi prompts..."
python src/generate_model_responses_with_id.py \
    --input $OUTPUT_DIR/hindi/all_filtered.csv \
    --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --output $OUTPUT_DIR/model_responses/llama3_1_8b_hindi_responses.csv \
    --max_tokens 100 \
    --temperature 0.7

# Step 6: Generate Aya responses for Hindi prompts
echo "Step 6: Generating Aya responses for Hindi prompts..."
python src/generate_model_responses_with_id.py \
    --input $OUTPUT_DIR/hindi/all_filtered.csv \
    --model "CohereForAI/aya-23-8B" \
    --output $OUTPUT_DIR/model_responses/aya_hindi_responses.csv \
    --max_tokens 100 \
    --temperature 0.7

# Step 7: Analyze toxicity of Hindi prompts
echo "Step 7: Analyzing toxicity of Hindi prompts..."
python src/analyze_toxicity_with_id.py \
    --input $OUTPUT_DIR/hindi/all_filtered.csv \
    --output $OUTPUT_DIR/toxicity_analysis/hindi_prompt_toxicity.csv \
    --progress_file $OUTPUT_DIR/toxicity_analysis/hindi_prompt_toxicity_progress.csv \
    --api_key "$API_KEY" \
    --batch_size 5

# Step 8: Analyze toxicity of LLaMA 3 (8B) responses for Hindi prompts
echo "Step 8: Analyzing toxicity of LLaMA 3 (8B) responses for Hindi prompts..."
python src/analyze_toxicity_with_id.py \
    --input $OUTPUT_DIR/model_responses/llama3_8b_hindi_responses.csv \
    --output $OUTPUT_DIR/toxicity_analysis/llama3_8b_hindi_toxicity.csv \
    --progress_file $OUTPUT_DIR/toxicity_analysis/llama3_8b_hindi_toxicity_progress.csv \
    --api_key "$API_KEY" \
    --batch_size 5

# Step 9: Analyze toxicity of LLaMA 3.1 (8B) responses for Hindi prompts
echo "Step 9: Analyzing toxicity of LLaMA 3.1 (8B) responses for Hindi prompts..."
python src/analyze_toxicity_with_id.py \
    --input $OUTPUT_DIR/model_responses/llama3_1_8b_hindi_responses.csv \
    --output $OUTPUT_DIR/toxicity_analysis/llama3_1_8b_hindi_toxicity.csv \
    --progress_file $OUTPUT_DIR/toxicity_analysis/llama3_1_8b_hindi_toxicity_progress.csv \
    --api_key "$API_KEY" \
    --batch_size 5

# Step 10: Analyze toxicity of Aya responses for Hindi prompts
echo "Step 10: Analyzing toxicity of Aya responses for Hindi prompts..."
python src/analyze_toxicity_with_id.py \
    --input $OUTPUT_DIR/model_responses/aya_hindi_responses.csv \
    --output $OUTPUT_DIR/toxicity_analysis/aya_hindi_toxicity.csv \
    --progress_file $OUTPUT_DIR/toxicity_analysis/aya_hindi_toxicity_progress.csv \
    --api_key "$API_KEY" \
    --batch_size 5

# Step 11: Compare toxicity for Hindi prompts and LLaMA 3 (8B) responses
echo "Step 11: Comparing toxicity for Hindi prompts and LLaMA 3 (8B) responses..."
python src/compare_toxicity_with_id.py \
    --prompt_file $OUTPUT_DIR/toxicity_analysis/hindi_prompt_toxicity.csv \
    --response_file $OUTPUT_DIR/toxicity_analysis/llama3_8b_hindi_toxicity.csv \
    --output_dir $OUTPUT_DIR/toxicity_analysis/hindi_llama3_8b_comparison \
    --model_name "LLaMA 3 (8B)"

# Step 12: Compare toxicity for Hindi prompts and LLaMA 3.1 (8B) responses
echo "Step 12: Comparing toxicity for Hindi prompts and LLaMA 3.1 (8B) responses..."
python src/compare_toxicity_with_id.py \
    --prompt_file $OUTPUT_DIR/toxicity_analysis/hindi_prompt_toxicity.csv \
    --response_file $OUTPUT_DIR/toxicity_analysis/llama3_1_8b_hindi_toxicity.csv \
    --output_dir $OUTPUT_DIR/toxicity_analysis/hindi_llama3_1_8b_comparison \
    --model_name "LLaMA 3.1 (8B)"

# Step 13: Create verification report to check primary key consistency
echo "Step 13: Creating verification report for primary key tracking..."
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
    
    # Language detection
    (os.path.join(base_dir, 'hindi', 'language_detection.csv'), 'Hindi language detection'),
    
    # Filtered outputs
    (os.path.join(base_dir, 'hindi', 'filtered_output.csv'), 'Hindi filtered output'),
    
    # All filtered
    (os.path.join(base_dir, 'hindi', 'all_filtered.csv'), 'Hindi all filtered for generation'),
    
    # Model responses
    (os.path.join(base_dir, 'model_responses', 'llama3_8b_hindi_responses.csv'), 'LLaMA 3 (8B) Hindi responses'),
    (os.path.join(base_dir, 'model_responses', 'llama3_1_8b_hindi_responses.csv'), 'LLaMA 3.1 (8B) Hindi responses'),
    (os.path.join(base_dir, 'model_responses', 'aya_hindi_responses.csv'), 'Aya Hindi responses'),
    
    # Toxicity analysis
    (os.path.join(base_dir, 'toxicity_analysis', 'hindi_prompt_toxicity.csv'), 'Hindi prompt toxicity'),
    (os.path.join(base_dir, 'toxicity_analysis', 'llama3_8b_hindi_toxicity.csv'), 'LLaMA 3 (8B) Hindi toxicity'),
    (os.path.join(base_dir, 'toxicity_analysis', 'llama3_1_8b_hindi_toxicity.csv'), 'LLaMA 3.1 (8B) Hindi toxicity'),
    (os.path.join(base_dir, 'toxicity_analysis', 'aya_hindi_toxicity.csv'), 'Aya Hindi toxicity'),
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
if os.path.exists(os.path.join(base_dir, 'hindi', 'all_filtered.csv')) and os.path.exists(os.path.join(base_dir, 'model_responses', 'llama3_8b_hindi_responses.csv')):
    hindi_filtered = pd.read_csv(os.path.join(base_dir, 'hindi', 'all_filtered.csv'))
    llama_hindi = pd.read_csv(os.path.join(base_dir, 'model_responses', 'llama3_8b_hindi_responses.csv'))
    hindi_ids_filtered = set(hindi_filtered['prompt_id'])
    hindi_ids_llama = set(llama_hindi['prompt_id'])
    print(f'Hindi filtered -> LLaMA 3 (8B) Hindi: {hindi_ids_filtered == hindi_ids_llama} ({len(hindi_ids_filtered)} IDs in filtered, {len(hindi_ids_llama)} IDs in LLaMA)')
"

# Step 14: Generate histograms and charts for toxicity comparison
echo "Step 14: Generating histograms and charts for toxicity analysis (PNG format only)..."

# Create visualization directories
mkdir -p $OUTPUT_DIR/visualizations
mkdir -p $OUTPUT_DIR/visualizations/histograms
mkdir -p $OUTPUT_DIR/visualizations/boxplots
mkdir -p $OUTPUT_DIR/visualizations/monolingual_vs_codeswitched

# Generate histograms for prompt toxicity
echo "Generating histograms for Hindi prompt toxicity..."
python -c "
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set styling
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Load toxicity data
hindi_prompt_toxicity = pd.read_csv('$OUTPUT_DIR/toxicity_analysis/hindi_prompt_toxicity.csv')
llama3_toxicity = pd.read_csv('$OUTPUT_DIR/toxicity_analysis/llama3_8b_hindi_toxicity.csv')
llama3_1_toxicity = pd.read_csv('$OUTPUT_DIR/toxicity_analysis/llama3_1_8b_hindi_toxicity.csv')
aya_toxicity = pd.read_csv('$OUTPUT_DIR/toxicity_analysis/aya_hindi_toxicity.csv')

# Load language detection data to identify code-switched vs monolingual
language_data = pd.read_csv('$OUTPUT_DIR/hindi/language_detection.csv')

# Function to generate histogram for a specific toxicity type
def generate_histogram(data, col_name, title, output_path):
    plt.figure(figsize=(12, 8))
    sns.histplot(data=data, x=col_name, bins=20, kde=True)
    plt.title(title)
    plt.xlabel(f'{col_name.title()} Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# Generate histograms for prompts
for col in ['toxicity', 'severe_toxicity', 'identity_attack', 'insult', 'profanity', 'threat']:
    # Skip if column doesn't exist
    if col not in hindi_prompt_toxicity.columns:
        continue
    output_path = f'$OUTPUT_DIR/visualizations/histograms/hindi_prompt_{col}_histogram.png'
    generate_histogram(hindi_prompt_toxicity, col, f'Hindi Prompts {col.title()} Distribution', output_path)

# Generate histograms for model responses
for model_name, toxicity_df in [
    ('LLaMA 3 (8B)', llama3_toxicity),
    ('LLaMA 3.1 (8B)', llama3_1_toxicity),
    ('Aya', aya_toxicity)
]:
    for col in ['toxicity', 'severe_toxicity', 'identity_attack', 'insult', 'profanity', 'threat']:
        # Skip if column doesn't exist
        if col not in toxicity_df.columns:
            continue
        output_path = f'$OUTPUT_DIR/visualizations/histograms/{model_name.lower().replace(\" \", \"_\")}_{col}_histogram.png'
        generate_histogram(toxicity_df, col, f'{model_name} Responses {col.title()} Distribution', output_path)

print('Generated histograms for prompt and response toxicity metrics')
"

# Generate boxplots comparing prompts vs responses
echo "Generating boxplots comparing prompt and response toxicity..."
python -c "
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set styling
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)

# Load toxicity data
hindi_prompt_toxicity = pd.read_csv('$OUTPUT_DIR/toxicity_analysis/hindi_prompt_toxicity.csv')
llama3_toxicity = pd.read_csv('$OUTPUT_DIR/toxicity_analysis/llama3_8b_hindi_toxicity.csv')
llama3_1_toxicity = pd.read_csv('$OUTPUT_DIR/toxicity_analysis/llama3_1_8b_hindi_toxicity.csv')
aya_toxicity = pd.read_csv('$OUTPUT_DIR/toxicity_analysis/aya_hindi_toxicity.csv')

# Function to prepare data for boxplot comparison
def prepare_boxplot_data(prompt_df, response_df, toxicity_type):
    if toxicity_type not in prompt_df.columns or toxicity_type not in response_df.columns:
        return None
    
    prompt_data = prompt_df[['prompt_id', toxicity_type]].rename(columns={toxicity_type: 'value'})
    prompt_data['source'] = 'Prompt'
    
    response_data = response_df[['prompt_id', toxicity_type]].rename(columns={toxicity_type: 'value'})
    response_data['source'] = 'Response'
    
    combined = pd.concat([prompt_data, response_data])
    combined['toxicity_type'] = toxicity_type.title()
    
    return combined

# Generate boxplots for each model and toxicity type
for model_name, toxicity_df in [
    ('LLaMA 3 (8B)', llama3_toxicity),
    ('LLaMA 3.1 (8B)', llama3_1_toxicity),
    ('Aya', aya_toxicity)
]:
    combined_data = pd.DataFrame()
    
    for toxicity_type in ['toxicity', 'severe_toxicity', 'identity_attack', 'insult', 'profanity', 'threat']:
        data = prepare_boxplot_data(hindi_prompt_toxicity, toxicity_df, toxicity_type)
        if data is not None:
            combined_data = pd.concat([combined_data, data])
    
    if not combined_data.empty:
        plt.figure(figsize=(16, 10))
        sns.boxplot(data=combined_data, x='toxicity_type', y='value', hue='source')
        plt.title(f'Hindi Prompts vs {model_name} Responses - Toxicity Comparison')
        plt.xlabel('Toxicity Type')
        plt.ylabel('Score')
        plt.legend(title='')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'$OUTPUT_DIR/visualizations/boxplots/hindi_vs_{model_name.lower().replace(\" \", \"_\")}_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()

print('Generated boxplots comparing prompt and response toxicity for all models')
"

# Generate comparison between monolingual and code-switched prompts
echo "Generating comparison between monolingual and code-switched prompts..."
python -c "
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set styling
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)

# Load toxicity data
hindi_prompt_toxicity = pd.read_csv('$OUTPUT_DIR/toxicity_analysis/hindi_prompt_toxicity.csv')
llama3_toxicity = pd.read_csv('$OUTPUT_DIR/toxicity_analysis/llama3_8b_hindi_toxicity.csv')
llama3_1_toxicity = pd.read_csv('$OUTPUT_DIR/toxicity_analysis/llama3_1_8b_hindi_toxicity.csv')
aya_toxicity = pd.read_csv('$OUTPUT_DIR/toxicity_analysis/aya_hindi_toxicity.csv')

# Load language detection data to identify code-switched vs monolingual
language_data = pd.read_csv('$OUTPUT_DIR/hindi/language_detection.csv')

# Determine code-switched vs monolingual based on english_percent
language_data['is_code_switched'] = language_data['english_percent'] > 0

# Join toxicity data with language classification
prompt_with_lang = pd.merge(hindi_prompt_toxicity, 
                           language_data[['prompt_id', 'is_code_switched', 'english_percent', 'total_hindi_percent']], 
                           on='prompt_id', how='left')

# Generate comparison plots for each model's response toxicity
for model_name, model_df in [
    ('LLaMA 3 (8B)', llama3_toxicity),
    ('LLaMA 3.1 (8B)', llama3_1_toxicity),
    ('Aya', aya_toxicity)
]:
    # Merge model toxicity with prompt language info
    model_with_lang = pd.merge(model_df, language_data[['prompt_id', 'is_code_switched']], on='prompt_id', how='left')
    
    # Plot toxicity distributions by prompt type (code-switched vs monolingual)
    for toxicity_type in ['toxicity', 'severe_toxicity', 'identity_attack', 'insult', 'profanity', 'threat']:
        if toxicity_type not in model_with_lang.columns:
            continue
            
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=model_with_lang, x='is_code_switched', y=toxicity_type)
        plt.title(f'{model_name} Response {toxicity_type.title()} by Prompt Type')
        plt.xlabel('Is Code-Switched Prompt')
        plt.ylabel(f'{toxicity_type.title()} Score')
        plt.xticks([0, 1], ['Monolingual', 'Code-Switched'])
        plt.grid(True, alpha=0.3)
        plt.savefig(f'$OUTPUT_DIR/visualizations/monolingual_vs_codeswitched/{model_name.lower().replace(\" \", \"_\")}_{toxicity_type}_by_prompt_type.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create scatterplot of toxicity vs english_percent
    for toxicity_type in ['toxicity', 'severe_toxicity', 'identity_attack', 'insult', 'profanity', 'threat']:
        if toxicity_type not in model_with_lang.columns:
            continue
            
        # Merge with language data to get english_percent
        scatter_data = pd.merge(model_with_lang, language_data[['prompt_id', 'english_percent']], on='prompt_id', how='left')
        
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=scatter_data, x='english_percent', y=toxicity_type)
        plt.title(f'{model_name} Response {toxicity_type.title()} vs English Percentage in Prompt')
        plt.xlabel('English Percentage in Prompt')
        plt.ylabel(f'{toxicity_type.title()} Score')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'$OUTPUT_DIR/visualizations/monolingual_vs_codeswitched/{model_name.lower().replace(\" \", \"_\")}_{toxicity_type}_vs_english_percent.png', dpi=300, bbox_inches='tight')
        plt.close()

print('Generated comparisons between monolingual and code-switched prompts')
"

# Calculate API usage
echo "Perspective API usage summary:"
echo "-----------------------------"
echo "Hindi prompts: $(wc -l < $OUTPUT_DIR/hindi/all_filtered.csv) requests (minus header)"
echo "LLaMA 3 (8B) Hindi responses: $(wc -l < $OUTPUT_DIR/model_responses/llama3_8b_hindi_responses.csv) requests (minus header)"
echo "LLaMA 3.1 (8B) Hindi responses: $(wc -l < $OUTPUT_DIR/model_responses/llama3_1_8b_hindi_responses.csv) requests (minus header)"
echo "Aya Hindi responses: $(wc -l < $OUTPUT_DIR/model_responses/aya_hindi_responses.csv) requests (minus header)"
echo "-----------------------------"
echo "Test analysis complete! Results saved to $OUTPUT_DIR"
echo "Check $OUTPUT_DIR/primary_key_verification.json for primary key tracking information"
echo "Visualizations saved to $OUTPUT_DIR/visualizations/" 