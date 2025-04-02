#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=fix_viz
#SBATCH --mem=16G
#SBATCH --output=outputs/fix_visualization.out

# Load necessary modules
module load 2023
module load Miniconda3/23.5.2-0
source ~/.bashrc
conda activate code-switch

# Define paths - Using the test_primary_key_hindi directory
TEST_DIR="data/output/test_primary_key_hindi"
TOXICITY_DIR="${TEST_DIR}/toxicity_analysis"
OUTPUT_DIR="data/output/visualizations"

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/histograms
mkdir -p $OUTPUT_DIR/boxplots
mkdir -p $OUTPUT_DIR/model_comparisons

echo "========================================"
echo "Fixed Visualization Script for Hindi Analysis"
echo "Using data from: ${TEST_DIR}"
echo "========================================"

# First, let's check that the files exist
echo "Checking for required files..."
if [ -f "${TOXICITY_DIR}/hindi_prompt_toxicity.csv" ]; then
    echo "Found hindi_prompt_toxicity.csv"
else
    echo "ERROR: hindi_prompt_toxicity.csv not found at ${TOXICITY_DIR}"
    exit 1
fi

if [ -f "${TOXICITY_DIR}/llama3_8b_hindi_toxicity.csv" ]; then
    echo "Found llama3_8b_hindi_toxicity.csv"
else
    echo "ERROR: llama3_8b_hindi_toxicity.csv not found at ${TOXICITY_DIR}"
    exit 1
fi

if [ -f "${TOXICITY_DIR}/aya_hindi_toxicity.csv" ]; then
    echo "Found aya_hindi_toxicity.csv"
else
    echo "ERROR: aya_hindi_toxicity.csv not found at ${TOXICITY_DIR}"
    exit 1
fi

# Compare toxicity for Hindi prompts and model responses
echo "Comparing toxicity for Hindi prompts and LLaMA responses..."
python src/compare_toxicity_with_id.py \
    --prompt_file ${TOXICITY_DIR}/hindi_prompt_toxicity.csv \
    --response_file ${TOXICITY_DIR}/llama3_8b_hindi_toxicity.csv \
    --output_dir ${OUTPUT_DIR}/hindi_llama_comparison \
    --model_name "LLaMA 3 (8B)"

echo "Comparing toxicity for Hindi prompts and Aya responses..."
python src/compare_toxicity_with_id.py \
    --prompt_file ${TOXICITY_DIR}/hindi_prompt_toxicity.csv \
    --response_file ${TOXICITY_DIR}/aya_hindi_toxicity.csv \
    --output_dir ${OUTPUT_DIR}/hindi_aya_comparison \
    --model_name "Aya"

# Create a Python script for enhanced visualizations
echo "Creating enhanced visualization script..."
cat > enhanced_visualizations.py << 'EOF'
#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import numpy as np

# Set up paths
TEST_DIR = "data/output/test_primary_key_hindi"
TOXICITY_DIR = f"{TEST_DIR}/toxicity_analysis"
OUTPUT_DIR = "data/output/visualizations"

# Ensure output directories exist
for dir_path in [OUTPUT_DIR, f"{OUTPUT_DIR}/histograms", f"{OUTPUT_DIR}/boxplots", f"{OUTPUT_DIR}/model_comparisons"]:
    os.makedirs(dir_path, exist_ok=True)

# Define toxicity columns to analyze
TOXICITY_COLUMNS = ['toxicity', 'severe_toxicity', 'identity_attack', 'insult', 'threat']

try:
    print(f"Loading data from {TOXICITY_DIR}...")
    
    # Load toxicity data
    hindi_prompt_toxicity = pd.read_csv(f"{TOXICITY_DIR}/hindi_prompt_toxicity.csv")
    llama_hindi_toxicity = pd.read_csv(f"{TOXICITY_DIR}/llama3_8b_hindi_toxicity.csv")
    aya_hindi_toxicity = pd.read_csv(f"{TOXICITY_DIR}/aya_hindi_toxicity.csv")
    
    # Check if llama 3.1 file exists and load it if available
    llama31_file = f"{TOXICITY_DIR}/llama3_1_8b_hindi_toxicity.csv"
    has_llama31 = os.path.exists(llama31_file)
    if has_llama31:
        llama31_hindi_toxicity = pd.read_csv(llama31_file)
        print("Found LLaMA 3.1 toxicity data")
    
    print(f"Loaded toxicity data: {len(hindi_prompt_toxicity)} prompt entries, {len(llama_hindi_toxicity)} LLaMA responses, {len(aya_hindi_toxicity)} Aya responses")
    
    # Try to find language detection data
    language_detection_file = None
    possible_paths = [
        f"{TEST_DIR}/language_detection/hindi_language_detection.csv", 
        f"{TEST_DIR}/hindi/language_detection.csv",
        f"{TEST_DIR}/hindi_language_detection.csv"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            language_detection_file = path
            break
    
    # If we found the language file, load it
    has_language_data = False
    if language_detection_file:
        print(f"Found language detection data at {language_detection_file}")
        hindi_lang = pd.read_csv(language_detection_file)
        has_language_data = True
    else:
        print("Warning: Could not find language detection data. Some visualizations will be skipped.")
    
    # Generate model comparison visualizations
    print("Generating model comparison visualizations...")
    
    # Calculate mean toxicity for each metric and model
    prompt_means = hindi_prompt_toxicity[TOXICITY_COLUMNS].mean()
    llama_means = llama_hindi_toxicity[TOXICITY_COLUMNS].mean()
    aya_means = aya_hindi_toxicity[TOXICITY_COLUMNS].mean()
    
    # Create comparison DataFrame
    comparison_data = {
        'Hindi Prompts': prompt_means,
        'LLaMA 3 (8B)': llama_means,
        'Aya': aya_means
    }
    
    # Add LLaMA 3.1 if available
    if has_llama31:
        llama31_means = llama31_hindi_toxicity[TOXICITY_COLUMNS].mean()
        comparison_data['LLaMA 3.1 (8B)'] = llama31_means
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Generate bar chart
    plt.figure(figsize=(14, 8))
    comparison_df.plot(kind='bar')
    plt.title('Average Toxicity: Hindi Prompts vs. Model Responses', fontsize=16)
    plt.xlabel('Toxicity Metric', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.xticks(rotation=30)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparisons/hindi_model_toxicity_comparison.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save comparison data to JSON
    with open(f"{OUTPUT_DIR}/model_comparisons/hindi_model_toxicity_stats.json", 'w') as f:
        json.dump({
            'prompt_toxicity': prompt_means.to_dict(),
            'llama_toxicity': llama_means.to_dict(),
            'aya_toxicity': aya_means.to_dict(),
            'llama31_toxicity': llama31_means.to_dict() if has_llama31 else None,
            'sample_counts': {
                'prompts': len(hindi_prompt_toxicity),
                'llama': len(llama_hindi_toxicity),
                'aya': len(aya_hindi_toxicity),
                'llama31': len(llama31_hindi_toxicity) if has_llama31 else 0
            }
        }, f, indent=4)
    
    # Generate boxplot comparison
    plt.figure(figsize=(15, 10))
    
    # Prepare data for boxplot - reshape to long format
    boxplot_data = []
    
    # For each toxicity column, gather the values from each source
    for column in TOXICITY_COLUMNS:
        for idx, row in hindi_prompt_toxicity.iterrows():
            boxplot_data.append({
                'Metric': column.replace('_', ' ').title(),
                'Value': row[column],
                'Source': 'Hindi Prompts'
            })
        
        for idx, row in llama_hindi_toxicity.iterrows():
            boxplot_data.append({
                'Metric': column.replace('_', ' ').title(),
                'Value': row[column],
                'Source': 'LLaMA 3 (8B)'
            })
        
        for idx, row in aya_hindi_toxicity.iterrows():
            boxplot_data.append({
                'Metric': column.replace('_', ' ').title(),
                'Value': row[column],
                'Source': 'Aya'
            })
        
        if has_llama31:
            for idx, row in llama31_hindi_toxicity.iterrows():
                boxplot_data.append({
                    'Metric': column.replace('_', ' ').title(),
                    'Value': row[column],
                    'Source': 'LLaMA 3.1 (8B)'
                })
    
    # Create the boxplot DataFrame
    boxplot_df = pd.DataFrame(boxplot_data)
    
    # Generate the boxplot
    plt.figure(figsize=(15, 10))
    sns.boxplot(x='Metric', y='Value', hue='Source', data=boxplot_df)
    plt.title('Toxicity Distribution: Hindi Prompts vs. Model Responses', fontsize=16)
    plt.xlabel('Toxicity Metric', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.legend(title='Source')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/boxplots/hindi_model_toxicity_boxplot.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Model comparison visualizations complete.")
    
    # If we have language data, generate language-based visualizations
    if has_language_data:
        print("Generating language-based visualizations...")
        
        # Define function to categorize language mix
        def categorize_language_mix(row):
            total_words = row['total_words']
            if total_words == 0:
                return 'unknown'
            
            hindi_percent = (row['hindi_words'] / total_words) * 100
            english_percent = (row['english_words'] / total_words) * 100
            
            if hindi_percent > 90:
                return 'monolingual_hindi'
            elif english_percent > 90:
                return 'monolingual_english'
            else:
                # Code-switched with meaningful mix
                if hindi_percent >= 30 and english_percent >= 30:
                    return 'code_switched_balanced'
                elif hindi_percent > english_percent:
                    return 'code_switched_hindi_dominant'
                else:
                    return 'code_switched_english_dominant'
        
        # Check if language data has required columns
        required_columns = ['id', 'hindi_words', 'english_words', 'total_words']
        if all(col in hindi_lang.columns for col in required_columns):
            # Add language category
            hindi_lang['language_category'] = hindi_lang.apply(categorize_language_mix, axis=1)
            
            # Merge with toxicity data
            merged_df = pd.merge(hindi_prompt_toxicity, 
                                 hindi_lang[['id', 'language_category']], 
                                 on='id', how='inner')
            
            # Generate summary statistics
            category_counts = merged_df['language_category'].value_counts().to_dict()
            print("Language categories found:")
            for category, count in category_counts.items():
                print(f"  - {category}: {count} samples")
            
            # Split into monolingual and code-switched
            mono_hindi = merged_df[merged_df['language_category'] == 'monolingual_hindi']
            code_switched = merged_df[merged_df['language_category'].str.contains('code_switched')]
            
            # Create comparison bar chart
            if not mono_hindi.empty and not code_switched.empty:
                mono_means = mono_hindi[TOXICITY_COLUMNS].mean()
                cs_means = code_switched[TOXICITY_COLUMNS].mean()
                
                comparison_data = pd.DataFrame({
                    'Monolingual Hindi': mono_means,
                    'Code-switched': cs_means
                })
                
                plt.figure(figsize=(12, 8))
                comparison_data.plot(kind='bar')
                plt.title('Average Toxicity: Monolingual Hindi vs. Code-switched Prompts', fontsize=16)
                plt.xlabel('Toxicity Metric', fontsize=14)
                plt.ylabel('Average Score', fontsize=14)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.xticks(rotation=30)
                plt.tight_layout()
                plt.savefig(f"{OUTPUT_DIR}/hindi_mono_vs_cs_toxicity.png", bbox_inches='tight', dpi=300)
                plt.close()
                
                # Save language comparison data
                with open(f"{OUTPUT_DIR}/hindi_language_analysis.json", 'w') as f:
                    json.dump({
                        'language_categories': category_counts,
                        'monolingual_toxicity': mono_means.to_dict(),
                        'code_switched_toxicity': cs_means.to_dict(),
                        'monolingual_samples': len(mono_hindi),
                        'code_switched_samples': len(code_switched)
                    }, f, indent=4)
                
                print("Language comparison visualizations complete.")
            else:
                print("Warning: Not enough data in monolingual or code-switched categories for comparison.")
        else:
            print(f"Warning: Language data missing required columns: {required_columns}")
    
    print("All visualizations completed successfully!")

except Exception as e:
    import traceback
    print(f"Error in visualization generation: {str(e)}")
    traceback.print_exc()
EOF

# Run the enhanced visualization script
echo "Running enhanced visualizations..."
python enhanced_visualizations.py

echo "==========================================="
echo "Visualization complete. Results saved to:"
echo "${OUTPUT_DIR}"
echo "===========================================" 