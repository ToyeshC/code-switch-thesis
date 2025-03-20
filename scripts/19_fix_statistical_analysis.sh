#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=19_fix_statistical_analysis
#SBATCH --mem=32G
#SBATCH --output=outputs/19_fix_statistical_analysis.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# Define paths
PROMPT_FILE="data/output/hindi/(yes) filtered_output_small.csv"
OUTPUT_DIR="data/output/model_toxicity_analysis"
API_KEY="AIzaSyDf0c2MkAItSv7TBFps65WavRFLP-N275Y"  # The API key from config.py

# Create script to fix and run the statistical analysis
cat > src/fix_statistical_analysis.py << 'EOF'
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse

def ensure_columns(df, required_columns):
    """Ensures that dataframe has all required columns, creating them with NaN if missing"""
    for col in required_columns:
        if col not in df.columns:
            print(f"WARNING: Column '{col}' missing from dataframe, adding it with NaN values")
            df[col] = np.nan
    return df

def analyze_toxicity_comparison(prompts_df, llama_df, aya_df, output_dir):
    """Analyze and compare toxicity between prompts and model responses"""
    
    print("\n===== Performing Statistical Analysis =====\n")
    
    # Ensure all dataframes have the same columns
    all_columns = ['comment', 'toxicity', 'severe_toxicity', 'identity_attack', 'insult', 'profanity', 'threat']
    
    print(f"Prompts DataFrame columns: {prompts_df.columns.tolist()}")
    print(f"LLaMA DataFrame columns: {llama_df.columns.tolist()}")
    print(f"Aya DataFrame columns: {aya_df.columns.tolist()}")
    
    # Ensure all dataframes have the required columns
    prompts_df = ensure_columns(prompts_df, all_columns)
    llama_df = ensure_columns(llama_df, all_columns)
    aya_df = ensure_columns(aya_df, all_columns)
    
    # Create copies to avoid modifying the original dataframes
    prompts_df = prompts_df.copy()
    llama_df = llama_df.copy()
    aya_df = aya_df.copy()
    
    # Rename comment column to prompt for clarity
    prompts_df = prompts_df.rename(columns={'comment': 'prompt'})
    llama_df = llama_df.rename(columns={'comment': 'prompt'})
    aya_df = aya_df.rename(columns={'comment': 'prompt'})
    
    print("\nAfter transformation:")
    print(f"Prompts DataFrame columns: {prompts_df.columns.tolist()}")
    print(f"LLaMA DataFrame columns: {llama_df.columns.tolist()}")
    print(f"Aya DataFrame columns: {aya_df.columns.tolist()}")
    
    # Merge the dataframes
    analysis_dir = os.path.join(output_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Define column sets
    prompt_cols = ['prompt']
    toxicity_cols = ['toxicity', 'severe_toxicity', 'identity_attack', 'insult', 'profanity', 'threat']
    
    print("Merging LLaMA data with prompts...")
    # Merge LLaMA with prompts
    llama_merged = prompts_df[prompt_cols + toxicity_cols].merge(
        llama_df[prompt_cols + toxicity_cols], 
        on='prompt', 
        suffixes=('_prompt', '_llama')
    )
    
    # Save the comparison
    llama_merged.to_csv(os.path.join(output_dir, 'llama_prompt_comparison.csv'), index=False)
    
    print("Merging Aya data with prompts...")
    # Merge Aya with prompts 
    aya_merged = prompts_df[prompt_cols + toxicity_cols].merge(
        aya_df[prompt_cols + toxicity_cols],
        on='prompt',
        suffixes=('_prompt', '_aya')
    )
    
    # Save the comparison
    aya_merged.to_csv(os.path.join(output_dir, 'aya_prompt_comparison.csv'), index=False)
    
    print("Merging all datasets...")
    # Create a new dataframe with all toxicity measures
    print(f"llama_merged columns: {llama_merged.columns.tolist()}")
    print(f"aya_merged columns: {aya_merged.columns.tolist()}")
    
    # Make sure we have just the prompt column without suffixes for merging
    llama_prompt_cols = [col for col in llama_merged.columns if col == 'prompt']
    
    try:
        all_merged = pd.merge(
            llama_merged, 
            aya_merged.drop([col for col in aya_merged.columns if col.endswith('_prompt')], axis=1), 
            on='prompt'
        )
        print(f"Successfully merged all data. Shape: {all_merged.shape}")
    except Exception as e:
        print(f"Error merging all data: {e}")
        print("Creating minimal merged dataset...")
        
        all_merged = pd.DataFrame({'prompt': prompts_df['prompt']})
        
        # Add LLaMA data if available
        for col in toxicity_cols:
            if f"{col}_llama" in llama_merged.columns:
                all_merged[f"{col}_llama"] = llama_merged[f"{col}_llama"]
            if f"{col}_prompt" in llama_merged.columns:
                all_merged[f"{col}_prompt"] = llama_merged[f"{col}_prompt"]
        
        # Add Aya data if available
        for col in toxicity_cols:
            col_name = f"{col}_aya"
            if col_name in aya_merged.columns:
                all_merged[col_name] = aya_merged[col_name]
    
    # Save the merged data
    all_merged.to_csv(os.path.join(analysis_dir, 'all_toxicity_metrics.csv'), index=False)
    
    # Calculate statistics for each toxicity metric
    stats_results = {}
    paired_ttest_results = {}
    
    for metric in toxicity_cols:
        # Get the metrics for prompts, LLaMA, and Aya
        try:
            prompt_metric = all_merged[f"{metric}_prompt"].dropna()
            llama_metric = all_merged[f"{metric}_llama"].dropna()
            aya_metric = all_merged[f"{metric}_aya"].dropna()
            
            # Calculate summary statistics
            stats_results[metric] = {
                'prompt': {
                    'mean': prompt_metric.mean(),
                    'median': prompt_metric.median(),
                    'std': prompt_metric.std(),
                    'min': prompt_metric.min(),
                    'max': prompt_metric.max()
                },
                'llama': {
                    'mean': llama_metric.mean(),
                    'median': llama_metric.median(),
                    'std': llama_metric.std(),
                    'min': llama_metric.min(),
                    'max': llama_metric.max()
                },
                'aya': {
                    'mean': aya_metric.mean(),
                    'median': aya_metric.median(),
                    'std': aya_metric.std(),
                    'min': aya_metric.min(),
                    'max': aya_metric.max()
                }
            }
            
            # Calculate paired t-test between prompt and each model
            # Only calculate if we have a sufficient number of paired observations
            common_llama = pd.DataFrame({
                'prompt': prompt_metric,
                'model': llama_metric
            }).dropna()
            
            common_aya = pd.DataFrame({
                'prompt': prompt_metric,
                'model': aya_metric
            }).dropna()
            
            if len(common_llama) >= 5:  # Arbitrary threshold for paired t-test
                tstat, pval = stats.ttest_rel(common_llama['prompt'], common_llama['model'])
                paired_ttest_results[f"{metric}_llama"] = {
                    'tstat': tstat,
                    'pval': pval,
                    'n': len(common_llama)
                }
            
            if len(common_aya) >= 5:
                tstat, pval = stats.ttest_rel(common_aya['prompt'], common_aya['model'])
                paired_ttest_results[f"{metric}_aya"] = {
                    'tstat': tstat,
                    'pval': pval,
                    'n': len(common_aya)
                }
            
        except Exception as e:
            print(f"Error calculating statistics for {metric}: {e}")
            continue
    
    # Create a summary table
    summary_df = pd.DataFrame(columns=[
        'Metric', 'Prompt Mean', 'LLaMA Mean', 'Aya Mean', 'LLaMA p-value', 'Aya p-value'
    ])
    
    for i, metric in enumerate(toxicity_cols):
        try:
            row = {
                'Metric': metric,
                'Prompt Mean': stats_results[metric]['prompt']['mean'],
                'LLaMA Mean': stats_results[metric]['llama']['mean'],
                'Aya Mean': stats_results[metric]['aya']['mean'],
                'LLaMA p-value': paired_ttest_results.get(f"{metric}_llama", {}).get('pval', np.nan),
                'Aya p-value': paired_ttest_results.get(f"{metric}_aya", {}).get('pval', np.nan)
            }
            summary_df.loc[i] = row
        except Exception as e:
            print(f"Error creating summary row for {metric}: {e}")
            # Add a partial row if possible
            row = {'Metric': metric}
            if metric in stats_results:
                if 'prompt' in stats_results[metric]:
                    row['Prompt Mean'] = stats_results[metric]['prompt'].get('mean', np.nan)
                if 'llama' in stats_results[metric]:
                    row['LLaMA Mean'] = stats_results[metric]['llama'].get('mean', np.nan)
                if 'aya' in stats_results[metric]:
                    row['Aya Mean'] = stats_results[metric]['aya'].get('mean', np.nan)
            summary_df.loc[i] = row
    
    # Save the summary
    summary_df.to_csv(os.path.join(analysis_dir, 'toxicity_summary.csv'), index=False)
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    
    try:
        # Prepare data for plotting
        plot_data = []
        
        for metric in toxicity_cols:
            if metric in stats_results:
                for source in ['prompt', 'llama', 'aya']:
                    if source in stats_results[metric]:
                        mean_val = stats_results[metric][source].get('mean', np.nan)
                        if not np.isnan(mean_val):
                            plot_data.append({
                                'Metric': metric.replace('_', ' ').title(),
                                'Source': source.title(),
                                'Mean': mean_val
                            })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create a grouped bar chart
        if not plot_df.empty:
            plt.figure(figsize=(12, 8))
            bar_plot = sns.barplot(x='Metric', y='Mean', hue='Source', data=plot_df)
            plt.title('Toxicity Comparison', fontsize=16)
            plt.xlabel('Toxicity Metric', fontsize=14)
            plt.ylabel('Mean Score', fontsize=14)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, 'toxicity_comparison.png'))
            plt.close()
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    # Return the statistics for further analysis
    return {
        'summary': summary_df,
        'detailed_stats': stats_results,
        'ttest_results': paired_ttest_results
    }

def main():
    parser = argparse.ArgumentParser(description="Fix and run the statistical analysis for toxicity comparison")
    parser.add_argument("--prompt_toxicity", required=True, help="Path to prompt toxicity CSV file")
    parser.add_argument("--llama_toxicity", required=True, help="Path to LLaMA toxicity CSV file")
    parser.add_argument("--aya_toxicity", required=True, help="Path to Aya toxicity CSV file")
    parser.add_argument("--output_dir", required=True, help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the toxicity data
    print(f"Loading toxicity data from {args.prompt_toxicity}, {args.llama_toxicity}, and {args.aya_toxicity}")
    
    try:
        prompts_df = pd.read_csv(args.prompt_toxicity)
        print(f"Loaded prompt toxicity data with shape {prompts_df.shape}")
    except Exception as e:
        print(f"Error loading prompt toxicity data: {e}")
        prompts_df = pd.DataFrame()
    
    try:
        llama_df = pd.read_csv(args.llama_toxicity)
        print(f"Loaded LLaMA toxicity data with shape {llama_df.shape}")
    except Exception as e:
        print(f"Error loading LLaMA toxicity data: {e}")
        llama_df = pd.DataFrame()
    
    try:
        aya_df = pd.read_csv(args.aya_toxicity)
        print(f"Loaded Aya toxicity data with shape {aya_df.shape}")
    except Exception as e:
        print(f"Error loading Aya toxicity data: {e}")
        aya_df = pd.DataFrame()
    
    # Run the analysis
    results = analyze_toxicity_comparison(prompts_df, llama_df, aya_df, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to {os.path.join(args.output_dir, 'analysis')}")

if __name__ == "__main__":
    main()
EOF

# Run the analysis
echo "==== Running Fixed Statistical Analysis ===="
python src/fix_statistical_analysis.py \
    --prompt_toxicity "$OUTPUT_DIR/prompt_toxicity.csv" \
    --llama_toxicity "$OUTPUT_DIR/llama_toxicity.csv" \
    --aya_toxicity "$OUTPUT_DIR/aya_toxicity.csv" \
    --output_dir "$OUTPUT_DIR"

echo "Fixed analysis complete! Results saved to $OUTPUT_DIR/analysis" 