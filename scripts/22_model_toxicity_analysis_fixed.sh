#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=22_model_toxicity
#SBATCH --mem=32G
#SBATCH --output=outputs/22_model_toxicity.out

module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# Define paths - using ONLY files from the model_toxicity_analysis folder
OUTPUT_DIR="data/output/model_toxicity_analysis"

# Create a Python script for comprehensive statistical analysis
cat > src/model_toxicity_analysis_fixed.py << 'EOF'
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
from scipy.stats import wilcoxon
import math

def cohens_d(x, y):
    """Calculate Cohen's d for two independent samples."""
    # Remove NaN values
    x = x.dropna().values
    y = y.dropna().values
    
    if len(x) < 2 or len(y) < 2:
        return np.nan
    
    # Calculate means
    mean1, mean2 = np.mean(x), np.mean(y)
    
    # Calculate pooled standard deviation
    n1, n2 = len(x), len(y)
    var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)  # ddof=1 for sample variance
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (mean1 - mean2) / pooled_std
    
    return d

def load_toxic_data(prompt_file, llama_file, aya_file, output_dir):
    """Load the toxicity data from all sources."""
    # Create output directory
    analysis_dir = os.path.join(output_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Load data
    print(f"Loading toxicity data...")
    prompt_df = pd.read_csv(prompt_file)
    llama_df = pd.read_csv(llama_file)
    aya_df = pd.read_csv(aya_file)
    
    print(f"Prompt data shape: {prompt_df.shape}")
    print(f"LLaMA data shape: {llama_df.shape}")
    print(f"Aya data shape: {aya_df.shape}")
    
    return prompt_df, llama_df, aya_df, analysis_dir

def compare_toxicity_stats(prompt_df, llama_df, aya_df, analysis_dir):
    """Perform comprehensive statistical comparisons between datasets."""
    # Define toxicity metrics
    metrics = ['toxicity', 'severe_toxicity', 'identity_attack', 'insult', 'profanity', 'threat']
    
    # Results dictionary to store all statistical results
    results = {}
    
    # 1. Paired comparisons between prompts and each model
    print("\n=== Performing Paired Comparisons Between Prompts and Models ===")
    
    results_rows = []
    
    for metric in metrics:
        print(f"\nAnalyzing metric: {metric}")
        result_row = {'Metric': metric}
        
        # Get values, limiting by the shortest dataset
        try:
            prompt_values = prompt_df[metric].dropna().reset_index(drop=True)
            llama_values = llama_df[metric].dropna().reset_index(drop=True)
            aya_values = aya_df[metric].dropna().reset_index(drop=True)
            
            min_len_llama = min(len(prompt_values), len(llama_values))
            min_len_aya = min(len(prompt_values), len(aya_values))
            
            # Calculate means
            result_row['Prompt Mean'] = prompt_values.mean()
            result_row['LLaMA Mean'] = llama_values.mean()
            result_row['Aya Mean'] = aya_values.mean()
            
            # Paired values for tests
            paired_prompt_llama = prompt_values.iloc[:min_len_llama]
            paired_llama = llama_values.iloc[:min_len_llama]
            
            paired_prompt_aya = prompt_values.iloc[:min_len_aya]
            paired_aya = aya_values.iloc[:min_len_aya]
            
            print(f"  Paired samples - Prompt vs LLaMA: {min_len_llama}, Prompt vs Aya: {min_len_aya}")
            
            # T-Tests
            if min_len_llama >= 2:
                t_stat, p_val = stats.ttest_rel(paired_prompt_llama, paired_llama)
                result_row['LLaMA T-test p-value'] = p_val
                print(f"  Prompt vs LLaMA t-test: t={t_stat:.4f}, p={p_val:.4f}")
            else:
                result_row['LLaMA T-test p-value'] = np.nan
                print("  Insufficient paired data for Prompt vs LLaMA t-test")
            
            if min_len_aya >= 2:
                t_stat, p_val = stats.ttest_rel(paired_prompt_aya, paired_aya)
                result_row['Aya T-test p-value'] = p_val
                print(f"  Prompt vs Aya t-test: t={t_stat:.4f}, p={p_val:.4f}")
            else:
                result_row['Aya T-test p-value'] = np.nan
                print("  Insufficient paired data for Prompt vs Aya t-test")
            
            # Wilcoxon Tests
            if min_len_llama >= 2:
                try:
                    w_stat, p_val = wilcoxon(paired_prompt_llama, paired_llama)
                    result_row['LLaMA Wilcoxon p-value'] = p_val
                    print(f"  Prompt vs LLaMA Wilcoxon: W={w_stat:.4f}, p={p_val:.4f}")
                except Exception as e:
                    print(f"  Error in Wilcoxon test for Prompt vs LLaMA: {e}")
                    result_row['LLaMA Wilcoxon p-value'] = np.nan
            else:
                result_row['LLaMA Wilcoxon p-value'] = np.nan
            
            if min_len_aya >= 2:
                try:
                    w_stat, p_val = wilcoxon(paired_prompt_aya, paired_aya)
                    result_row['Aya Wilcoxon p-value'] = p_val
                    print(f"  Prompt vs Aya Wilcoxon: W={w_stat:.4f}, p={p_val:.4f}")
                except Exception as e:
                    print(f"  Error in Wilcoxon test for Prompt vs Aya: {e}")
                    result_row['Aya Wilcoxon p-value'] = np.nan
            else:
                result_row['Aya Wilcoxon p-value'] = np.nan
            
            # Effect Sizes (Cohen's d)
            # Fixed: Using double quotes for keys with apostrophes
            result_row["LLaMA Effect Size (Cohen's d)"] = cohens_d(paired_prompt_llama, paired_llama)
            result_row["Aya Effect Size (Cohen's d)"] = cohens_d(paired_prompt_aya, paired_aya)
            
            # Fixed: Using double quotes for dictionary lookup in f-string
            print(f"  Effect sizes - LLaMA: {result_row['LLaMA Effect Size (Cohen\\'s d)']:.4f}, Aya: {result_row['Aya Effect Size (Cohen\\'s d)']:.4f}")
            
            # Store results for this metric
            results[metric] = {
                'prompt_values': prompt_values.tolist(),
                'llama_values': llama_values.tolist(),
                'aya_values': aya_values.tolist(),
                'prompt_mean': prompt_values.mean(),
                'llama_mean': llama_values.mean(),
                'aya_mean': aya_values.mean(),
                'prompt_vs_llama_ttest_pval': result_row.get('LLaMA T-test p-value', np.nan),
                'prompt_vs_aya_ttest_pval': result_row.get('Aya T-test p-value', np.nan),
                'prompt_vs_llama_wilcoxon_pval': result_row.get('LLaMA Wilcoxon p-value', np.nan),
                'prompt_vs_aya_wilcoxon_pval': result_row.get('Aya Wilcoxon p-value', np.nan),
                'prompt_vs_llama_cohens_d': result_row.get("LLaMA Effect Size (Cohen's d)", np.nan),  # Fixed
                'prompt_vs_aya_cohens_d': result_row.get("Aya Effect Size (Cohen's d)", np.nan)  # Fixed
            }
            
        except Exception as e:
            print(f"  Error processing metric {metric}: {e}")
            # Add partial results if available
            results[metric] = {}
        
        results_rows.append(result_row)
    
    # Create and save the results table
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(os.path.join(analysis_dir, 'prompt_vs_models_stats.csv'), index=False)
    print(f"Saved prompt vs models comparison to {os.path.join(analysis_dir, 'prompt_vs_models_stats.csv')}")
    
    # 2. LLaMA vs Aya comparison
    print("\n=== Comparing LLaMA vs Aya ===")
    llama_aya_rows = []
    
    for metric in metrics:
        print(f"\nComparing models for metric: {metric}")
        result_row = {'Metric': metric}
        
        try:
            llama_values = llama_df[metric].dropna().reset_index(drop=True)
            aya_values = aya_df[metric].dropna().reset_index(drop=True)
            
            min_len = min(len(llama_values), len(aya_values))
            
            if min_len < 2:
                print(f"  Insufficient data for {metric} comparison")
                continue
            
            paired_llama = llama_values.iloc[:min_len]
            paired_aya = aya_values.iloc[:min_len]
            
            # Means
            result_row['LLaMA Mean'] = paired_llama.mean()
            result_row['Aya Mean'] = paired_aya.mean()
            result_row['Mean Difference'] = paired_llama.mean() - paired_aya.mean()
            
            # T-test
            t_stat, p_val = stats.ttest_rel(paired_llama, paired_aya)
            result_row['T-test p-value'] = p_val
            
            # Wilcoxon
            try:
                w_stat, p_val = wilcoxon(paired_llama, paired_aya)
                result_row['Wilcoxon p-value'] = p_val
            except Exception as e:
                print(f"  Error in Wilcoxon test for LLaMA vs Aya: {e}")
                result_row['Wilcoxon p-value'] = np.nan
            
            # Effect size - Fixed
            result_row["Effect Size (Cohen's d)"] = cohens_d(paired_llama, paired_aya)
            
            print(f"  LLaMA mean: {result_row['LLaMA Mean']:.4f}, Aya mean: {result_row['Aya Mean']:.4f}")
            print(f"  T-test p-value: {result_row['T-test p-value']:.4f}")
            print(f"  Wilcoxon p-value: {result_row.get('Wilcoxon p-value', 'N/A')}")
            print(f"  Effect size: {result_row['Effect Size (Cohen\\'s d)']:.4f}")  # Fixed
            
            # Store in overall results
            if metric in results:
                results[metric].update({
                    'llama_vs_aya_ttest_pval': result_row.get('T-test p-value', np.nan),
                    'llama_vs_aya_wilcoxon_pval': result_row.get('Wilcoxon p-value', np.nan),
                    'llama_vs_aya_cohens_d': result_row.get("Effect Size (Cohen's d)", np.nan)  # Fixed
                })
            
            llama_aya_rows.append(result_row)
            
        except Exception as e:
            print(f"  Error comparing models for {metric}: {e}")
    
    # Save LLaMA vs Aya comparison
    llama_aya_df = pd.DataFrame(llama_aya_rows)
    llama_aya_df.to_csv(os.path.join(analysis_dir, 'llama_vs_aya_stats.csv'), index=False)
    print(f"Saved LLaMA vs Aya comparison to {os.path.join(analysis_dir, 'llama_vs_aya_stats.csv')}")
    
    return results

def create_visualizations(prompt_df, llama_df, aya_df, results, analysis_dir):
    """Create visualizations for toxicity comparisons."""
    metrics = ['toxicity', 'severe_toxicity', 'identity_attack', 'insult', 'profanity', 'threat']
    
    # 1. Create box plots comparing distributions
    print("\n=== Creating Box Plots ===")
    
    # Prepare data for box plots
    for metric in metrics:
        try:
            plt.figure(figsize=(10, 6))
            
            # Get data
            prompt_values = prompt_df[metric].dropna()
            llama_values = llama_df[metric].dropna()
            aya_values = aya_df[metric].dropna()
            
            # Create DataFrame for box plot
            box_data = pd.DataFrame({
                'Prompt': prompt_values,
                'LLaMA': llama_values.iloc[:len(prompt_values)] if len(llama_values) > 0 else [],
                'Aya': aya_values.iloc[:len(prompt_values)] if len(aya_values) > 0 else []
            })
            
            # Convert to long format for seaborn
            box_data_long = pd.melt(box_data, var_name='Source', value_name=metric)
            
            # Create box plot
            sns.boxplot(x='Source', y=metric, data=box_data_long)
            plt.title(f'{metric.replace("_", " ").title()} Distribution', fontsize=14)
            plt.ylabel(f'{metric.replace("_", " ").title()} Score', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f'{metric}_boxplot.png'))
            plt.savefig(os.path.join(analysis_dir, f'{metric}_boxplot.pdf'))
            plt.close()
            
            print(f"  Created box plot for {metric}")
            
        except Exception as e:
            print(f"  Error creating box plot for {metric}: {e}")
    
    # 2. Create correlation matrices and heatmaps
    print("\n=== Creating Correlation Matrices ===")
    
    # Prepare dataframes for correlation analysis
    try:
        # Ensure all dataframes have the same length
        min_len = min(len(prompt_df), len(llama_df), len(aya_df))
        
        prompt_corr = prompt_df[metrics].iloc[:min_len].copy()
        llama_corr = llama_df[metrics].iloc[:min_len].copy()
        aya_corr = aya_df[metrics].iloc[:min_len].copy()
        
        # Rename columns to identify source
        for df, prefix in [(prompt_corr, 'prompt_'), (llama_corr, 'llama_'), (aya_corr, 'aya_')]:
            df.columns = [prefix + col for col in df.columns]
        
        # Combine for cross-correlations
        combined_corr = pd.concat([prompt_corr, llama_corr, aya_corr], axis=1)
        
        # Calculate correlations
        corr_matrix = combined_corr.corr()
        
        # Save correlation matrix
        corr_matrix.to_csv(os.path.join(analysis_dir, 'correlation_matrix.csv'))
        
        # Create heatmap
        plt.figure(figsize=(15, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                   vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
        plt.title('Correlation Matrix of Toxicity Metrics', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'correlation_heatmap.png'))
        plt.savefig(os.path.join(analysis_dir, 'correlation_heatmap.pdf'))
        plt.close()
        
        print("  Created correlation matrix and heatmap")
        
        # Create specific correlation matrices
        
        # 1. Within-dataset correlations
        for df, name in [(prompt_corr, 'prompt'), (llama_corr, 'llama'), (aya_corr, 'aya')]:
            corr = df.corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                       vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
            plt.title(f'{name.title()} Toxicity Metrics Correlation', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f'{name}_correlation_heatmap.png'))
            plt.savefig(os.path.join(analysis_dir, f'{name}_correlation_heatmap.pdf'))
            plt.close()
            
            print(f"  Created correlation heatmap for {name}")
        
        # 2. Cross-dataset correlations
        # Prompt vs LLaMA
        cross_corr = pd.DataFrame()
        for p_metric in prompt_corr.columns:
            for l_metric in llama_corr.columns:
                cross_corr.loc[p_metric, l_metric] = prompt_corr[p_metric].corr(llama_corr[l_metric])
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cross_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   vmin=-1, vmax=1, center=0, linewidths=.5)
        plt.title('Prompt vs LLaMA Toxicity Correlation', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'prompt_vs_llama_correlation.png'))
        plt.savefig(os.path.join(analysis_dir, 'prompt_vs_llama_correlation.pdf'))
        plt.close()
        
        print("  Created Prompt vs LLaMA correlation heatmap")
        
        # Prompt vs Aya
        cross_corr = pd.DataFrame()
        for p_metric in prompt_corr.columns:
            for a_metric in aya_corr.columns:
                cross_corr.loc[p_metric, a_metric] = prompt_corr[p_metric].corr(aya_corr[a_metric])
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cross_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   vmin=-1, vmax=1, center=0, linewidths=.5)
        plt.title('Prompt vs Aya Toxicity Correlation', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'prompt_vs_aya_correlation.png'))
        plt.savefig(os.path.join(analysis_dir, 'prompt_vs_aya_correlation.pdf'))
        plt.close()
        
        print("  Created Prompt vs Aya correlation heatmap")
        
        # LLaMA vs Aya
        cross_corr = pd.DataFrame()
        for l_metric in llama_corr.columns:
            for a_metric in aya_corr.columns:
                cross_corr.loc[l_metric, a_metric] = llama_corr[l_metric].corr(aya_corr[a_metric])
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cross_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   vmin=-1, vmax=1, center=0, linewidths=.5)
        plt.title('LLaMA vs Aya Toxicity Correlation', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'llama_vs_aya_correlation.png'))
        plt.savefig(os.path.join(analysis_dir, 'llama_vs_aya_correlation.pdf'))
        plt.close()
        
        print("  Created LLaMA vs Aya correlation heatmap")
        
    except Exception as e:
        print(f"  Error creating correlation matrices: {e}")

def extract_prompts_from_response_file(response_file):
    """Extract prompts from a response file in the model_toxicity_analysis folder"""
    try:
        df = pd.read_csv(response_file)
        if 'prompt' in df.columns:
            return df['prompt']
        return None
    except:
        return None

def main():
    parser = argparse.ArgumentParser(description="Comprehensive statistical analysis of toxicity data")
    parser.add_argument("--prompt_toxicity", required=True, help="Path to prompt toxicity CSV file")
    parser.add_argument("--llama_toxicity", required=True, help="Path to LLaMA toxicity CSV file")
    parser.add_argument("--aya_toxicity", required=True, help="Path to Aya toxicity CSV file")
    parser.add_argument("--output_dir", required=True, help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    # Load data
    prompt_df, llama_df, aya_df, analysis_dir = load_toxic_data(
        args.prompt_toxicity, args.llama_toxicity, args.aya_toxicity, args.output_dir
    )
    
    # Perform statistical comparisons
    results = compare_toxicity_stats(prompt_df, llama_df, aya_df, analysis_dir)
    
    # Create visualizations
    create_visualizations(prompt_df, llama_df, aya_df, results, analysis_dir)
    
    # Try to get prompts from the response files in the model_toxicity_analysis folder
    # We need to look for model_response files that might contain code-switching info
    llama_df_path = os.path.join(args.output_dir, 'llama_responses.csv')
    
    # Read llama_responses.csv for extra analysis
    if os.path.exists(llama_df_path):
        prompts = extract_prompts_from_response_file(llama_df_path)
        
        if prompts is not None:
            print("\n=== Analyzing Code-Switching Patterns ===")
            print("Looking for code-switching patterns in model responses...")
            
            # The prompts might contain code-switching patterns that we can analyze
            # But we don't have specific code-switching metrics in these files
            # We can highlight which prompts had the highest toxicity difference
            
            # Find which prompts had highest toxicity difference
            try:
                if 'toxicity' in prompt_df.columns and 'toxicity' in llama_df.columns:
                    prompt_toxicity = prompt_df['toxicity'].reset_index(drop=True)
                    llama_toxicity = llama_df['toxicity'].reset_index(drop=True)
                    
                    min_len = min(len(prompt_toxicity), len(llama_toxicity))
                    
                    if min_len > 0:
                        toxicity_diff = abs(prompt_toxicity.iloc[:min_len] - llama_toxicity.iloc[:min_len])
                        
                        # Get top 10 prompts with highest toxicity difference
                        top_indices = toxicity_diff.nlargest(10).index.tolist()
                        
                        # Create a table of these prompts and their toxicity
                        top_diff_df = pd.DataFrame({
                            'Prompt': prompts.iloc[top_indices].tolist(),
                            'Prompt Toxicity': prompt_toxicity.iloc[top_indices].tolist(),
                            'LLaMA Toxicity': llama_toxicity.iloc[top_indices].tolist(),
                            'Toxicity Difference': toxicity_diff.iloc[top_indices].tolist()
                        })
                        
                        # Save to file
                        top_diff_df.to_csv(os.path.join(analysis_dir, 'top_toxicity_differences.csv'), index=False)
                        print(f"Saved top toxicity differences to {os.path.join(analysis_dir, 'top_toxicity_differences.csv')}")
                        
                        # Create a scatter plot of prompt toxicity vs model toxicity
                        plt.figure(figsize=(10, 8))
                        plt.scatter(prompt_toxicity.iloc[:min_len], llama_toxicity.iloc[:min_len], alpha=0.6)
                        
                        # Add diagonal line (y=x)
                        max_val = max(prompt_toxicity.iloc[:min_len].max(), llama_toxicity.iloc[:min_len].max())
                        plt.plot([0, max_val], [0, max_val], 'r--')
                        
                        plt.xlabel('Prompt Toxicity')
                        plt.ylabel('LLaMA Response Toxicity')
                        plt.title('Prompt vs LLaMA Toxicity')
                        plt.grid(True, linestyle='--', alpha=0.7)
                        plt.tight_layout()
                        plt.savefig(os.path.join(analysis_dir, 'prompt_vs_llama_toxicity_scatter.png'))
                        plt.savefig(os.path.join(analysis_dir, 'prompt_vs_llama_toxicity_scatter.pdf'))
                        plt.close()
                        
                        print("Created scatter plot of prompt vs LLaMA toxicity")
            except Exception as e:
                print(f"Error analyzing toxicity differences: {e}")
    
    print(f"\nComprehensive analysis complete! Results saved to {analysis_dir}")

if __name__ == "__main__":
    main()
EOF

# Run the analysis
echo "==== Running Model Toxicity Analysis ===="
python src/model_toxicity_analysis_fixed.py \
    --prompt_toxicity "$OUTPUT_DIR/prompt_toxicity.csv" \
    --llama_toxicity "$OUTPUT_DIR/llama_toxicity.csv" \
    --aya_toxicity "$OUTPUT_DIR/aya_toxicity.csv" \
    --output_dir "$OUTPUT_DIR"

echo "Analysis complete! Results saved to $OUTPUT_DIR/analysis" 