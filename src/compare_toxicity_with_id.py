import pandas as pd
import argparse
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def compare_toxicity(prompt_file, response_file, output_dir, model_name="model"):
    """
    Compare toxicity between prompts and model responses using primary keys.
    
    Args:
        prompt_file (str): Path to CSV file with prompt toxicity scores
        response_file (str): Path to CSV file with response toxicity scores
        output_dir (str): Directory to save output files
        model_name (str): Name of the model for labeling
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read input files
    print(f"Reading prompt toxicity file: {prompt_file}")
    prompt_df = pd.read_csv(prompt_file)
    
    print(f"Reading {model_name} response toxicity file: {response_file}")
    response_df = pd.read_csv(response_file)
    
    # Ensure both files have primary key
    if 'prompt_id' not in prompt_df.columns:
        print(f"Error: No 'prompt_id' column in {prompt_file}")
        return
    
    if 'prompt_id' not in response_df.columns:
        print(f"Error: No 'prompt_id' column in {response_file}")
        return
    
    # Merge datasets on primary key
    merged_df = pd.merge(
        prompt_df,
        response_df,
        on="prompt_id",
        suffixes=("_prompt", f"_{model_name.lower()}")
    )
    
    print(f"Merged dataset has {len(merged_df)} rows")
    
    # Toxicity metrics to compare
    metrics = ['toxicity', 'severe_toxicity', 'identity_attack', 'insult', 'profanity', 'threat']
    
    # Create output file for statistical comparison
    stats_file = os.path.join(output_dir, f"toxicity_comparison_{model_name.lower()}.csv")
    stats_rows = []
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Compare each metric
    print("\n===== Statistical Comparison =====")
    for metric in metrics:
        prompt_col = f"{metric}_prompt"
        response_col = f"{metric}_{model_name.lower()}"
        
        # Skip if columns don't exist
        if prompt_col not in merged_df.columns or response_col not in merged_df.columns:
            print(f"Warning: Columns {prompt_col} or {response_col} not found, skipping")
            continue
        
        # Get non-null values for paired tests
        valid_rows = merged_df.dropna(subset=[prompt_col, response_col])
        
        if len(valid_rows) < 10:
            print(f"Warning: Not enough valid data points for {metric}, skipping")
            continue
            
        prompt_values = valid_rows[prompt_col]
        response_values = valid_rows[response_col]
        
        # Basic statistics
        prompt_mean = prompt_values.mean()
        response_mean = response_values.mean()
        mean_diff = prompt_mean - response_mean
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(prompt_values, response_values)
        
        # Wilcoxon signed-rank test (non-parametric alternative to paired t-test)
        try:
            w_stat, w_p_value = stats.wilcoxon(prompt_values, response_values)
        except:
            w_stat, w_p_value = np.nan, np.nan
        
        # Effect size (Cohen's d for paired samples)
        diff = prompt_values - response_values
        d = diff.mean() / diff.std()
        
        # Print results
        print(f"\n{metric.capitalize()}:")
        print(f"  Prompt mean: {prompt_mean:.4f}, {model_name} mean: {response_mean:.4f}, Difference: {mean_diff:.4f}")
        print(f"  Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
        print(f"  Wilcoxon test: W={w_stat:.4f}, p={w_p_value:.4f}")
        print(f"  Effect size (Cohen's d): {d:.4f}")
        
        # Add to stats rows
        stats_rows.append({
            "metric": metric,
            "prompt_mean": prompt_mean,
            f"{model_name.lower()}_mean": response_mean,
            "mean_difference": mean_diff,
            "t_statistic": t_stat,
            "t_p_value": p_value,
            "significant_t_test": p_value < 0.05,
            "wilcoxon_statistic": w_stat,
            "wilcoxon_p_value": w_p_value,
            "significant_wilcoxon": w_p_value < 0.05 if not np.isnan(w_p_value) else np.nan,
            "cohens_d": d,
            "effect_size_interpretation": interpret_effect_size(d)
        })
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(prompt_values, response_values, alpha=0.5)
        
        # Add diagonal line (y=x)
        max_val = max(prompt_values.max(), response_values.max())
        plt.plot([0, max_val], [0, max_val], 'r--')
        
        # Add trend line
        z = np.polyfit(prompt_values, response_values, 1)
        p = np.poly1d(z)
        plt.plot(prompt_values, p(prompt_values), "g-", alpha=0.8)
        
        plt.xlabel(f"Prompt {metric.capitalize()}")
        plt.ylabel(f"{model_name} Response {metric.capitalize()}")
        plt.title(f"Prompt vs {model_name} {metric.capitalize()} (n={len(valid_rows)})")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add correlation coefficient and p-value to plot
        corr, corr_p = stats.pearsonr(prompt_values, response_values)
        plt.annotate(f"r = {corr:.4f} (p = {corr_p:.4f})", 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        plot_file = os.path.join(plots_dir, f"{metric}_{model_name.lower()}_scatter.png")
        plt.savefig(plot_file)
        plt.close()
    
    # Save statistics to CSV
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(stats_file, index=False)
    print(f"\nStatistics saved to {stats_file}")
    
    # Create summary bar chart comparing all metrics
    if len(stats_rows) > 0:
        plt.figure(figsize=(12, 8))
        
        # Extract data for plotting
        metrics = [row['metric'] for row in stats_rows]
        prompt_means = [row['prompt_mean'] for row in stats_rows]
        response_means = [row[f"{model_name.lower()}_mean"] for row in stats_rows]
        
        # Set up bar positions
        x = np.arange(len(metrics))
        width = 0.35
        
        # Create bars
        bar1 = plt.bar(x - width/2, prompt_means, width, label='Prompt')
        bar2 = plt.bar(x + width/2, response_means, width, label=f'{model_name} Response')
        
        # Add labels and title
        plt.xlabel('Toxicity Metric')
        plt.ylabel('Mean Score')
        plt.title(f'Comparison of Toxicity Metrics: Prompt vs {model_name}')
        plt.xticks(x, [m.capitalize() for m in metrics])
        plt.legend()
        
        # Add significance markers
        for i, row in enumerate(stats_rows):
            if row['significant_t_test']:
                height = max(prompt_means[i], response_means[i]) + 0.05
                plt.text(i, height, '*', ha='center', va='bottom', fontsize=20)
        
        plt.ylim(0, max(max(prompt_means), max(response_means)) * 1.2)
        plt.tight_layout()
        
        # Save the plot
        summary_plot_file = os.path.join(plots_dir, f"toxicity_summary_{model_name.lower()}.png")
        plt.savefig(summary_plot_file)
        plt.close()
        print(f"Summary plot saved to {summary_plot_file}")
    
    # Return the merged dataframe for further analysis
    return merged_df

def interpret_effect_size(d):
    """
    Interpret Cohen's d effect size.
    
    Args:
        d (float): Cohen's d value
        
    Returns:
        str: Interpretation of effect size
    """
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"

def main():
    parser = argparse.ArgumentParser(description="Compare toxicity between prompts and model responses")
    parser.add_argument("--prompt_file", required=True, help="Path to CSV file with prompt toxicity scores")
    parser.add_argument("--response_file", required=True, help="Path to CSV file with response toxicity scores")
    parser.add_argument("--output_dir", required=True, help="Directory to save output files")
    parser.add_argument("--model_name", default="Model", help="Name of the model for labeling")
    
    args = parser.parse_args()
    
    compare_toxicity(args.prompt_file, args.response_file, args.output_dir, args.model_name)

if __name__ == "__main__":
    main() 