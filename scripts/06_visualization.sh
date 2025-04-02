#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=visualize
#SBATCH --mem=16G
#SBATCH --output=outputs/06_visualization.out

# Load necessary modules
module load 2023
module load Miniconda3/23.5.2-0
source ~/.bashrc
conda activate code-switch

# Define paths - UPDATED to use test_primary_key_hindi directory
# The toxicity files are in this location based on directory listing
TOXICITY_DIR="data/output/test_primary_key_hindi/toxicity_analysis"
LANG_DIR="data/output/test_primary_key_hindi/language_detection"
FILTERED_DIR="data/output/test_primary_key_hindi/filtered"
OUTPUT_DIR="data/output/visualizations"

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/histograms
mkdir -p $OUTPUT_DIR/boxplots
mkdir -p $OUTPUT_DIR/monolingual_vs_codeswitched
mkdir -p $OUTPUT_DIR/heatmaps
mkdir -p $OUTPUT_DIR/scatter_plots

# Compare toxicity for Hindi prompts and model responses
echo "Comparing toxicity for Hindi prompts and LLaMA responses..."
python src/compare_toxicity_with_id.py \
    --prompt_file $TOXICITY_DIR/hindi_prompt_toxicity.csv \
    --response_file $TOXICITY_DIR/llama3_8b_hindi_toxicity.csv \
    --output_dir $OUTPUT_DIR/hindi_llama_comparison \
    --model_name "LLaMA 3 (8B)"

echo "Comparing toxicity for Hindi prompts and LLaMA 3.1 responses..."
python src/compare_toxicity_with_id.py \
    --prompt_file $TOXICITY_DIR/hindi_prompt_toxicity.csv \
    --response_file $TOXICITY_DIR/llama3_1_8b_hindi_toxicity.csv \
    --output_dir $OUTPUT_DIR/hindi_llama31_comparison \
    --model_name "LLaMA 3.1 (8B)"

echo "Comparing toxicity for Hindi prompts and Aya responses..."
python src/compare_toxicity_with_id.py \
    --prompt_file $TOXICITY_DIR/hindi_prompt_toxicity.csv \
    --response_file $TOXICITY_DIR/aya_hindi_toxicity.csv \
    --output_dir $OUTPUT_DIR/hindi_aya_comparison \
    --model_name "Aya"

# Note: We're focusing only on Hindi analysis since English files aren't available
# in the test_primary_key_hindi directory

# Generate enhanced visualizations with our Python script
echo "Generating enhanced visualizations..."
cat > scripts/06_visualization_temp.py << 'EOF'
#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Set styling
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Output directories
OUTPUT_DIR = "data/output/visualizations"
HISTOGRAMS_DIR = f"{OUTPUT_DIR}/histograms"
BOXPLOTS_DIR = f"{OUTPUT_DIR}/boxplots"
MONO_VS_CS_DIR = f"{OUTPUT_DIR}/monolingual_vs_codeswitched"
HEATMAPS_DIR = f"{OUTPUT_DIR}/heatmaps"
SCATTER_DIR = f"{OUTPUT_DIR}/scatter_plots"

# Create directories if they don't exist
for dir_path in [OUTPUT_DIR, HISTOGRAMS_DIR, BOXPLOTS_DIR, MONO_VS_CS_DIR, HEATMAPS_DIR, SCATTER_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Load datasets - UPDATED PATHS to use test_primary_key_hindi directory
try:
    print("Loading datasets...")
    
    # Load toxicity data with updated paths
    hindi_prompt_toxicity = pd.read_csv('data/output/test_primary_key_hindi/toxicity_analysis/hindi_prompt_toxicity.csv')
    llama_hindi_toxicity = pd.read_csv('data/output/test_primary_key_hindi/toxicity_analysis/llama3_8b_hindi_toxicity.csv')
    llama31_hindi_toxicity = pd.read_csv('data/output/test_primary_key_hindi/toxicity_analysis/llama3_1_8b_hindi_toxicity.csv')
    aya_hindi_toxicity = pd.read_csv('data/output/test_primary_key_hindi/toxicity_analysis/aya_hindi_toxicity.csv')
    
    # English data is not available in test_primary_key_hindi directory, 
    # so we'll focus only on Hindi analysis
    
    # Look for language detection data - we might need to create a custom path for this
    try:
        hindi_lang = pd.read_csv('data/output/test_primary_key_hindi/language_detection/hindi_language_detection.csv')
    except FileNotFoundError:
        # Try alternative locations
        try:
            hindi_lang = pd.read_csv('data/output/test_primary_key_hindi/hindi/language_detection.csv')
        except FileNotFoundError:
            try:
                # Try another possible location
                hindi_lang = pd.read_csv('data/output/test_primary_key_hindi/hindi_language_detection.csv')
            except FileNotFoundError:
                # If we can't find it, we'll need to check the directory structure
                print("Warning: Could not find language detection file. Will look for it in various locations.")
                # Search for possible language detection files
                for root, dirs, files in os.walk('data/output/test_primary_key_hindi'):
                    for file in files:
                        if 'language' in file.lower() and 'detection' in file.lower() and file.endswith('.csv'):
                            print(f"Found potential language detection file: {os.path.join(root, file)}")
                            hindi_lang = pd.read_csv(os.path.join(root, file))
                            break
    
    # Look for filtered data
    try:
        hindi_filtered = pd.read_csv('data/output/test_primary_key_hindi/filtered/hindi_filtered.csv')
    except FileNotFoundError:
        try:
            hindi_filtered = pd.read_csv('data/output/test_primary_key_hindi/hindi/filtered.csv')
        except FileNotFoundError:
            try:
                hindi_filtered = pd.read_csv('data/output/test_primary_key_hindi/hindi_filtered.csv')
            except FileNotFoundError:
                # If we can't find it, we'll use all toxicity IDs
                print("Warning: Could not find filtered IDs file. Using all toxicity IDs.")
                hindi_filtered = pd.DataFrame({'id': hindi_prompt_toxicity['id'].unique()})
    
    print("All datasets loaded successfully.")
    
    # Ensure we're only using filtered IDs for fair comparison
    hindi_ids = set(hindi_filtered['id'])
    print(f"Using {len(hindi_ids)} filtered Hindi IDs for analysis")
    
    # Filter toxicity data to only include filtered IDs
    hindi_prompt_toxicity = hindi_prompt_toxicity[hindi_prompt_toxicity['id'].isin(hindi_ids)]
    llama_hindi_toxicity = llama_hindi_toxicity[llama_hindi_toxicity['id'].isin(hindi_ids)]
    llama31_hindi_toxicity = llama31_hindi_toxicity[llama31_hindi_toxicity['id'].isin(hindi_ids)]
    aya_hindi_toxicity = aya_hindi_toxicity[aya_hindi_toxicity['id'].isin(hindi_ids)]
    
    # Filter language data to only include filtered IDs
    if 'hindi_lang' in locals():
        hindi_lang = hindi_lang[hindi_lang['id'].isin(hindi_ids)]
    
    # Categorize prompts as monolingual or code-switched
    # Monolingual: >90% of words are from one language
    # Code-switched: Contains significant mix of languages
    
    print("Categorizing prompts as monolingual or code-switched...")
    
    # Function to categorize based on language percentages
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
    
    # Check if we have language data with the necessary columns
    if 'hindi_lang' in locals() and all(col in hindi_lang.columns for col in ['hindi_words', 'english_words', 'total_words']):
        # Add language category to datasets
        hindi_lang['language_category'] = hindi_lang.apply(categorize_language_mix, axis=1)
        
        # Merge language categories with toxicity data
        hindi_prompt_with_cat = pd.merge(hindi_prompt_toxicity, 
                                        hindi_lang[['id', 'language_category']], 
                                        on='id', how='left')
        
        # Generate summary statistics and visualizations based on language categories
        print("Generating summary statistics...")
        
        # Count prompts in each category
        hindi_categories = hindi_lang['language_category'].value_counts().to_dict()
        
        summary_data = {
            'hindi_prompt_categories': hindi_categories,
            'hindi_prompts_count': len(hindi_prompt_toxicity),
            'filtered_hindi_ids_count': len(hindi_ids)
        }
        
        # Save summary to JSON
        with open(f"{OUTPUT_DIR}/language_summary.json", 'w') as f:
            json.dump(summary_data, f, indent=4)
        
        print("Summary statistics saved to language_summary.json")
        
        # Define toxicity columns for analysis
        toxicity_columns = ['toxicity', 'severe_toxicity', 'identity_attack', 'insult', 'threat']
        
        # Create comparison visualizations with model responses
        print("Creating model response comparison visualizations...")
        
        # Create bar chart comparing average toxicity across models
        plt.figure(figsize=(14, 10))
        
        # Calculate mean toxicity for each metric and model
        prompt_means = hindi_prompt_toxicity[toxicity_columns].mean()
        llama_means = llama_hindi_toxicity[toxicity_columns].mean()
        llama31_means = llama31_hindi_toxicity[toxicity_columns].mean()
        aya_means = aya_hindi_toxicity[toxicity_columns].mean()
        
        # Combine into DataFrame
        comparison_df = pd.DataFrame({
            'Hindi Prompts': prompt_means,
            'LLaMA 3 (8B)': llama_means,
            'LLaMA 3.1 (8B)': llama31_means,
            'Aya': aya_means
        })
        
        # Plot as bar chart
        comparison_df.plot(kind='bar', figsize=(14, 8))
        plt.title('Average Toxicity Metrics: Hindi Prompts vs. Model Responses', fontsize=16)
        plt.xlabel('Toxicity Metric', fontsize=14)
        plt.ylabel('Average Score', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Source', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/hindi_models_toxicity_comparison.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        # Create detailed report
        report = {
            'prompt_toxicity': prompt_means.to_dict(),
            'llama3_toxicity': llama_means.to_dict(),
            'llama31_toxicity': llama31_means.to_dict(), 
            'aya_toxicity': aya_means.to_dict(),
            'prompt_samples': len(hindi_prompt_toxicity),
            'model_samples': {
                'llama3': len(llama_hindi_toxicity),
                'llama31': len(llama31_hindi_toxicity),
                'aya': len(aya_hindi_toxicity)
            }
        }
        
        with open(f"{OUTPUT_DIR}/model_toxicity_report.json", 'w') as f:
            json.dump(report, f, indent=4)
        
        print("Model comparison analysis complete.")
        
        # If we have language categories, generate those visualizations too
        if 'hindi_prompt_with_cat' in locals():
            # Continue with the language category visualizations
            
            # 1. Create histograms comparing toxicity distributions
            print("Generating histograms for toxicity distributions...")
            
            # Hindi prompts: Monolingual vs Code-switched
            plt.figure(figsize=(14, 8))
            
            for i, column in enumerate(toxicity_columns):
                plt.subplot(2, 3, i+1)
                
                # Filter for monolingual and code-switched
                mono = hindi_prompt_with_cat[hindi_prompt_with_cat['language_category'] == 'monolingual_hindi']
                code_switched = hindi_prompt_with_cat[hindi_prompt_with_cat['language_category'].str.contains('code_switched')]
                
                if not mono.empty:
                    sns.histplot(mono[column], kde=True, label='Monolingual Hindi', alpha=0.6, color='blue')
                if not code_switched.empty:
                    sns.histplot(code_switched[column], kde=True, label='Code-switched', alpha=0.6, color='red')
                    
                plt.xlabel(column.replace('_', ' ').title())
                plt.ylabel('Frequency')
                plt.legend()
                plt.tight_layout()
            
            plt.suptitle('Toxicity Distribution: Monolingual Hindi vs. Code-switched Prompts', y=1.02, fontsize=16)
            plt.tight_layout()
            plt.savefig(f"{HISTOGRAMS_DIR}/hindi_mono_vs_cs_toxicity_distribution.png", bbox_inches='tight', dpi=300)
            plt.close()
            
            # 2. Create boxplots comparing toxicity metrics - FIXED VERSION
            print("Generating boxplots for toxicity metrics comparison...")
            
            # Create a long-format dataframe for Hindi boxplots
            hindi_long_df = pd.DataFrame()
            
            # Process Hindi data - using a different approach to avoid duplicate indices
            for category in ['monolingual_hindi', 'code_switched_balanced', 
                            'code_switched_hindi_dominant', 'code_switched_english_dominant']:
                category_data = hindi_prompt_with_cat[hindi_prompt_with_cat['language_category'] == category]
                if not category_data.empty:
                    # For each toxicity metric, create a separate dataframe and concatenate
                    for metric in toxicity_columns:
                        temp_df = pd.DataFrame({
                            'Category': category.replace('_', ' ').title(),
                            'Metric': metric.replace('_', ' ').title(),
                            'Value': category_data[metric].values
                        })
                        hindi_long_df = pd.concat([hindi_long_df, temp_df], ignore_index=True)
            
            # Hindi boxplots
            if not hindi_long_df.empty:
                plt.figure(figsize=(15, 10))
                # Explicitly reset index to avoid duplicate label error
                sns.boxplot(x='Metric', y='Value', hue='Category', data=hindi_long_df.reset_index(drop=True), palette='Set2')
                plt.title('Toxicity Metrics by Language Category (Hindi Prompts)', fontsize=16)
                plt.xlabel('Toxicity Metric', fontsize=14)
                plt.ylabel('Score', fontsize=14)
                plt.legend(title='Language Category', title_fontsize=12)
                plt.xticks(rotation=20)
                plt.tight_layout()
                plt.savefig(f"{BOXPLOTS_DIR}/hindi_language_categories_toxicity_boxplot.png", bbox_inches='tight', dpi=300)
                plt.close()
            
            # 3. Direct comparison of monolingual vs code-switched (bar charts)
            print("Creating direct comparison bar charts...")
            
            # For Hindi: compare average toxicity between monolingual and code-switched
            mono_hindi = hindi_prompt_with_cat[hindi_prompt_with_cat['language_category'] == 'monolingual_hindi']
            cs_hindi = hindi_prompt_with_cat[hindi_prompt_with_cat['language_category'].str.contains('code_switched')]
            
            if not mono_hindi.empty and not cs_hindi.empty:
                mono_means = mono_hindi[toxicity_columns].mean()
                cs_means = cs_hindi[toxicity_columns].mean()
                
                comparison_data = pd.DataFrame({
                    'Monolingual Hindi': mono_means,
                    'Code-switched': cs_means
                })
                
                plt.figure(figsize=(12, 8))
                comparison_data.plot(kind='bar', figsize=(12, 8))
                plt.title('Average Toxicity: Monolingual Hindi vs. Code-switched Prompts', fontsize=16)
                plt.xlabel('Toxicity Metric', fontsize=14)
                plt.ylabel('Average Score', fontsize=14)
                plt.xticks(rotation=20)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(f"{MONO_VS_CS_DIR}/hindi_mono_vs_cs_average_toxicity.png", bbox_inches='tight', dpi=300)
                plt.close()
            
            print("Completed visualizations based on language categories.")
    else:
        print("Warning: Language detection data missing required columns. Skipping language category visualizations.")
        
        # Still create model response comparisons even without language data
        print("Creating model response comparison visualizations...")
        
        # Calculate mean toxicity for each metric and model
        prompt_means = hindi_prompt_toxicity[toxicity_columns].mean()
        llama_means = llama_hindi_toxicity[toxicity_columns].mean() 
        llama31_means = llama31_hindi_toxicity[toxicity_columns].mean()
        aya_means = aya_hindi_toxicity[toxicity_columns].mean()
        
        # Combine into DataFrame
        comparison_df = pd.DataFrame({
            'Hindi Prompts': prompt_means,
            'LLaMA 3 (8B)': llama_means,
            'LLaMA 3.1 (8B)': llama31_means,
            'Aya': aya_means
        })
        
        # Plot as bar chart
        comparison_df.plot(kind='bar', figsize=(14, 8))
        plt.title('Average Toxicity Metrics: Hindi Prompts vs. Model Responses', fontsize=16)
        plt.xlabel('Toxicity Metric', fontsize=14)
        plt.ylabel('Average Score', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Source', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/hindi_models_toxicity_comparison.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    print("All visualizations generated successfully!")

except Exception as e:
    import traceback
    print(f"Error in visualization generation: {str(e)}")
    traceback.print_exc()
EOF

# Make the Python script executable and run it
chmod +x scripts/06_visualization_temp.py
python scripts/06_visualization_temp.py

echo "Enhanced visualization complete. Results saved to $OUTPUT_DIR"
echo "Created additional visualizations comparing monolingual and code-switched content" 