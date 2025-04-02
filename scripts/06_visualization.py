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

# Load datasets
try:
    print("Loading datasets...")
    
    # Load toxicity data
    hindi_prompt_toxicity = pd.read_csv('data/output/toxicity_analysis/hindi_prompt_toxicity.csv')
    english_prompt_toxicity = pd.read_csv('data/output/toxicity_analysis/english_prompt_toxicity.csv')
    llama_hindi_toxicity = pd.read_csv('data/output/toxicity_analysis/llama3_8b_hindi_toxicity.csv')
    llama_english_toxicity = pd.read_csv('data/output/toxicity_analysis/llama3_8b_english_toxicity.csv')
    aya_hindi_toxicity = pd.read_csv('data/output/toxicity_analysis/aya_hindi_toxicity.csv')
    aya_english_toxicity = pd.read_csv('data/output/toxicity_analysis/aya_english_toxicity.csv')
    
    # Load language detection data
    hindi_lang = pd.read_csv('data/output/language_detection/hindi_language_detection.csv')
    english_lang = pd.read_csv('data/output/language_detection/english_language_detection.csv')
    
    # Load filtered data (containing the IDs we want to use)
    hindi_filtered = pd.read_csv('data/output/filtered/hindi_filtered.csv')
    english_filtered = pd.read_csv('data/output/filtered/english_filtered.csv')
    
    print("All datasets loaded successfully.")
    
    # Ensure we're only using filtered IDs for fair comparison
    hindi_ids = set(hindi_filtered['id'])
    english_ids = set(english_filtered['id'])
    
    # Filter toxicity data to only include filtered IDs
    hindi_prompt_toxicity = hindi_prompt_toxicity[hindi_prompt_toxicity['id'].isin(hindi_ids)]
    english_prompt_toxicity = english_prompt_toxicity[english_prompt_toxicity['id'].isin(english_ids)]
    llama_hindi_toxicity = llama_hindi_toxicity[llama_hindi_toxicity['id'].isin(hindi_ids)]
    llama_english_toxicity = llama_english_toxicity[llama_english_toxicity['id'].isin(english_ids)]
    aya_hindi_toxicity = aya_hindi_toxicity[aya_hindi_toxicity['id'].isin(hindi_ids)]
    aya_english_toxicity = aya_english_toxicity[aya_english_toxicity['id'].isin(english_ids)]
    
    # Filter language data to only include filtered IDs
    hindi_lang = hindi_lang[hindi_lang['id'].isin(hindi_ids)]
    english_lang = english_lang[english_lang['id'].isin(english_ids)]
    
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
    
    # Add language category to datasets
    hindi_lang['language_category'] = hindi_lang.apply(categorize_language_mix, axis=1)
    english_lang['language_category'] = english_lang.apply(categorize_language_mix, axis=1)
    
    # Merge language categories with toxicity data
    hindi_prompt_with_cat = pd.merge(hindi_prompt_toxicity, 
                                    hindi_lang[['id', 'language_category']], 
                                    on='id', how='left')
    english_prompt_with_cat = pd.merge(english_prompt_toxicity, 
                                     english_lang[['id', 'language_category']], 
                                     on='id', how='left')
    
    # Generate summary statistics and comparison report
    print("Generating summary statistics...")
    
    # Count prompts in each category
    hindi_categories = hindi_lang['language_category'].value_counts().to_dict()
    english_categories = english_lang['language_category'].value_counts().to_dict()
    
    summary_data = {
        'hindi_prompt_categories': hindi_categories,
        'english_prompt_categories': english_categories,
        'hindi_prompts_count': len(hindi_prompt_toxicity),
        'english_prompts_count': len(english_prompt_toxicity),
        'filtered_hindi_ids_count': len(hindi_ids),
        'filtered_english_ids_count': len(english_ids)
    }
    
    # Save summary to JSON
    with open(f"{OUTPUT_DIR}/language_summary.json", 'w') as f:
        json.dump(summary_data, f, indent=4)
    
    print("Summary statistics saved to language_summary.json")
    
    # 1. Create histograms comparing toxicity distributions
    print("Generating histograms for toxicity distributions...")
    
    # Define toxicity columns for analysis
    toxicity_columns = ['toxicity', 'severe_toxicity', 'identity_attack', 'insult', 'threat']
    
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
    
    # English prompts: Monolingual vs Code-switched
    plt.figure(figsize=(14, 8))
    
    for i, column in enumerate(toxicity_columns):
        plt.subplot(2, 3, i+1)
        
        # Filter for monolingual and code-switched
        mono = english_prompt_with_cat[english_prompt_with_cat['language_category'] == 'monolingual_english']
        code_switched = english_prompt_with_cat[english_prompt_with_cat['language_category'].str.contains('code_switched')]
        
        if not mono.empty:
            sns.histplot(mono[column], kde=True, label='Monolingual English', alpha=0.6, color='green')
        if not code_switched.empty:
            sns.histplot(code_switched[column], kde=True, label='Code-switched', alpha=0.6, color='red')
            
        plt.xlabel(column.replace('_', ' ').title())
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
    
    plt.suptitle('Toxicity Distribution: Monolingual English vs. Code-switched Prompts', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{HISTOGRAMS_DIR}/english_mono_vs_cs_toxicity_distribution.png", bbox_inches='tight', dpi=300)
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
    
    # Create a long-format dataframe for English boxplots
    english_long_df = pd.DataFrame()
    
    # Process English data
    for category in ['monolingual_english', 'code_switched_balanced', 
                    'code_switched_hindi_dominant', 'code_switched_english_dominant']:
        category_data = english_prompt_with_cat[english_prompt_with_cat['language_category'] == category]
        if not category_data.empty:
            # For each toxicity metric, create a separate dataframe and concatenate
            for metric in toxicity_columns:
                temp_df = pd.DataFrame({
                    'Category': category.replace('_', ' ').title(),
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': category_data[metric].values
                })
                english_long_df = pd.concat([english_long_df, temp_df], ignore_index=True)
    
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
    
    # English boxplots
    if not english_long_df.empty:
        plt.figure(figsize=(15, 10))
        # Explicitly reset index to avoid duplicate label error
        sns.boxplot(x='Metric', y='Value', hue='Category', data=english_long_df.reset_index(drop=True), palette='Set2')
        plt.title('Toxicity Metrics by Language Category (English Prompts)', fontsize=16)
        plt.xlabel('Toxicity Metric', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.legend(title='Language Category', title_fontsize=12)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(f"{BOXPLOTS_DIR}/english_language_categories_toxicity_boxplot.png", bbox_inches='tight', dpi=300)
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
    
    # For English: compare average toxicity between monolingual and code-switched
    mono_english = english_prompt_with_cat[english_prompt_with_cat['language_category'] == 'monolingual_english']
    cs_english = english_prompt_with_cat[english_prompt_with_cat['language_category'].str.contains('code_switched')]
    
    if not mono_english.empty and not cs_english.empty:
        mono_means = mono_english[toxicity_columns].mean()
        cs_means = cs_english[toxicity_columns].mean()
        
        comparison_data = pd.DataFrame({
            'Monolingual English': mono_means,
            'Code-switched': cs_means
        })
        
        plt.figure(figsize=(12, 8))
        comparison_data.plot(kind='bar', figsize=(12, 8))
        plt.title('Average Toxicity: Monolingual English vs. Code-switched Prompts', fontsize=16)
        plt.xlabel('Toxicity Metric', fontsize=14)
        plt.ylabel('Average Score', fontsize=14)
        plt.xticks(rotation=20)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{MONO_VS_CS_DIR}/english_mono_vs_cs_average_toxicity.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    # 4. Heatmaps for correlations between metrics for different language categories
    print("Generating heatmaps for metric correlations...")
    
    # Hindi monolingual correlations
    if not mono_hindi.empty and len(mono_hindi) > 5:
        plt.figure(figsize=(10, 8))
        corr = mono_hindi[toxicity_columns].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, annot=True, fmt=".2f")
        plt.title('Correlation Between Toxicity Metrics (Monolingual Hindi)', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{HEATMAPS_DIR}/mono_hindi_toxicity_correlation.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    # Hindi code-switched correlations
    if not cs_hindi.empty and len(cs_hindi) > 5:
        plt.figure(figsize=(10, 8))
        corr = cs_hindi[toxicity_columns].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, annot=True, fmt=".2f")
        plt.title('Correlation Between Toxicity Metrics (Code-switched Hindi)', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{HEATMAPS_DIR}/cs_hindi_toxicity_correlation.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    # English monolingual correlations
    if not mono_english.empty and len(mono_english) > 5:
        plt.figure(figsize=(10, 8))
        corr = mono_english[toxicity_columns].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, annot=True, fmt=".2f")
        plt.title('Correlation Between Toxicity Metrics (Monolingual English)', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{HEATMAPS_DIR}/mono_english_toxicity_correlation.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    # English code-switched correlations
    if not cs_english.empty and len(cs_english) > 5:
        plt.figure(figsize=(10, 8))
        corr = cs_english[toxicity_columns].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, annot=True, fmt=".2f")
        plt.title('Correlation Between Toxicity Metrics (Code-switched English)', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{HEATMAPS_DIR}/cs_english_toxicity_correlation.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    # 5. Scatter plots to visualize relationships between different metrics
    print("Creating scatter plots for metric relationships...")
    
    # Hindi: Toxicity vs Severe Toxicity by language category
    if not hindi_prompt_with_cat.empty:
        plt.figure(figsize=(12, 8))
        colors = {'monolingual_hindi': 'blue', 'code_switched_balanced': 'red', 
                'code_switched_hindi_dominant': 'orange', 'code_switched_english_dominant': 'purple'}
        
        for category, color in colors.items():
            df_subset = hindi_prompt_with_cat[hindi_prompt_with_cat['language_category'] == category]
            if not df_subset.empty:
                plt.scatter(df_subset['toxicity'], df_subset['severe_toxicity'], 
                        color=color, alpha=0.7, label=category.replace('_', ' ').title())
        
        plt.title('Toxicity vs Severe Toxicity (Hindi Prompts)', fontsize=16)
        plt.xlabel('Toxicity Score', fontsize=14)
        plt.ylabel('Severe Toxicity Score', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Language Category')
        plt.tight_layout()
        plt.savefig(f"{SCATTER_DIR}/hindi_toxicity_vs_severe_toxicity.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    # English: Toxicity vs Severe Toxicity by language category
    if not english_prompt_with_cat.empty:
        plt.figure(figsize=(12, 8))
        colors = {'monolingual_english': 'green', 'code_switched_balanced': 'red', 
                'code_switched_hindi_dominant': 'orange', 'code_switched_english_dominant': 'purple'}
        
        for category, color in colors.items():
            df_subset = english_prompt_with_cat[english_prompt_with_cat['language_category'] == category]
            if not df_subset.empty:
                plt.scatter(df_subset['toxicity'], df_subset['severe_toxicity'], 
                        color=color, alpha=0.7, label=category.replace('_', ' ').title())
        
        plt.title('Toxicity vs Severe Toxicity (English Prompts)', fontsize=16)
        plt.xlabel('Toxicity Score', fontsize=14)
        plt.ylabel('Severe Toxicity Score', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Language Category')
        plt.tight_layout()
        plt.savefig(f"{SCATTER_DIR}/english_toxicity_vs_severe_toxicity.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    # 6. Additional comparison: model responses by language category
    print("Creating analysis of model responses by language category...")
    
    # Merge language categories with model response toxicity data
    llama_hindi_with_cat = pd.merge(llama_hindi_toxicity, 
                                   hindi_lang[['id', 'language_category']], 
                                   on='id', how='left')
    aya_hindi_with_cat = pd.merge(aya_hindi_toxicity, 
                                 hindi_lang[['id', 'language_category']], 
                                 on='id', how='left')
    
    # Compare model response toxicity by prompt language category (Hindi)
    # LLaMA responses
    llama_hindi_by_cat = {}
    for category in ['monolingual_hindi', 'code_switched_balanced', 
                    'code_switched_hindi_dominant', 'code_switched_english_dominant']:
        cat_data = llama_hindi_with_cat[llama_hindi_with_cat['language_category'] == category]
        if not cat_data.empty:
            llama_hindi_by_cat[category] = cat_data[toxicity_columns].mean()
    
    if llama_hindi_by_cat:
        llama_hindi_comparison = pd.DataFrame(llama_hindi_by_cat)
        plt.figure(figsize=(12, 8))
        llama_hindi_comparison.plot(kind='bar', figsize=(12, 8))
        plt.title('LLaMA 3 Response Toxicity by Hindi Prompt Language Category', fontsize=16)
        plt.xlabel('Toxicity Metric', fontsize=14)
        plt.ylabel('Average Score', fontsize=14)
        plt.xticks(rotation=20)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/llama_hindi_response_by_category.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    # Aya responses
    aya_hindi_by_cat = {}
    for category in ['monolingual_hindi', 'code_switched_balanced', 
                    'code_switched_hindi_dominant', 'code_switched_english_dominant']:
        cat_data = aya_hindi_with_cat[aya_hindi_with_cat['language_category'] == category]
        if not cat_data.empty:
            aya_hindi_by_cat[category] = cat_data[toxicity_columns].mean()
    
    if aya_hindi_by_cat:
        aya_hindi_comparison = pd.DataFrame(aya_hindi_by_cat)
        plt.figure(figsize=(12, 8))
        aya_hindi_comparison.plot(kind='bar', figsize=(12, 8))
        plt.title('Aya Response Toxicity by Hindi Prompt Language Category', fontsize=16)
        plt.xlabel('Toxicity Metric', fontsize=14)
        plt.ylabel('Average Score', fontsize=14)
        plt.xticks(rotation=20)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/aya_hindi_response_by_category.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    # Create a comparison report
    print("Creating detailed comparison report...")
    
    # Collect statistics for all categories
    category_stats = {}
    
    # Process Hindi dataset
    for category in hindi_prompt_with_cat['language_category'].unique():
        if pd.notna(category):
            cat_data = hindi_prompt_with_cat[hindi_prompt_with_cat['language_category'] == category]
            if not cat_data.empty:
                category_stats[f"hindi_{category}"] = {
                    'count': len(cat_data),
                    'metrics': {col: {'mean': cat_data[col].mean(), 'std': cat_data[col].std()} 
                              for col in toxicity_columns}
                }
    
    # Process English dataset
    for category in english_prompt_with_cat['language_category'].unique():
        if pd.notna(category):
            cat_data = english_prompt_with_cat[english_prompt_with_cat['language_category'] == category]
            if not cat_data.empty:
                category_stats[f"english_{category}"] = {
                    'count': len(cat_data),
                    'metrics': {col: {'mean': cat_data[col].mean(), 'std': cat_data[col].std()} 
                              for col in toxicity_columns}
                }
    
    # Add model response statistics by prompt category
    for category in llama_hindi_with_cat['language_category'].unique():
        if pd.notna(category):
            cat_data = llama_hindi_with_cat[llama_hindi_with_cat['language_category'] == category]
            if not cat_data.empty:
                category_stats[f"llama_response_to_hindi_{category}"] = {
                    'count': len(cat_data),
                    'metrics': {col: {'mean': cat_data[col].mean(), 'std': cat_data[col].std()} 
                              for col in toxicity_columns}
                }
    
    for category in aya_hindi_with_cat['language_category'].unique():
        if pd.notna(category):
            cat_data = aya_hindi_with_cat[aya_hindi_with_cat['language_category'] == category]
            if not cat_data.empty:
                category_stats[f"aya_response_to_hindi_{category}"] = {
                    'count': len(cat_data),
                    'metrics': {col: {'mean': cat_data[col].mean(), 'std': cat_data[col].std()} 
                              for col in toxicity_columns}
                }
    
    # Save detailed statistics to JSON
    with open(f"{OUTPUT_DIR}/detailed_toxicity_by_language_category.json", 'w') as f:
        json.dump(category_stats, f, indent=4)
    
    print("Detailed comparison report saved to detailed_toxicity_by_language_category.json")
    
    print("All visualizations generated successfully!")

except Exception as e:
    import traceback
    print(f"Error in visualization generation: {str(e)}")
    traceback.print_exc() 