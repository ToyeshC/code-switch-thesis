import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse

def direct_analysis(prompt_file, llama_file, aya_file, output_dir):
    """
    Analyze toxicity data directly without relying on prompt matching.
    Uses the row index to align data across files.
    """
    # Create output directory
    analysis_dir = os.path.join(output_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Load the data
    print(f"Loading toxicity data...")
    prompt_df = pd.read_csv(prompt_file)
    llama_df = pd.read_csv(llama_file)
    aya_df = pd.read_csv(aya_file)
    
    print(f"Prompt data shape: {prompt_df.shape}")
    print(f"LLaMA data shape: {llama_df.shape}")
    print(f"Aya data shape: {aya_df.shape}")
    
    # Print column names to verify
    print(f"Prompt columns: {prompt_df.columns.tolist()}")
    print(f"LLaMA columns: {llama_df.columns.tolist()}")
    print(f"Aya columns: {aya_df.columns.tolist()}")
    
    # Print a few sample values to understand the data
    print("\nSample prompt data:")
    print(prompt_df.head(3))
    print("\nSample LLaMA data:")
    print(llama_df.head(3))
    print("\nSample Aya data:")
    print(aya_df.head(3))
    
    # Define all toxicity metrics
    metrics = ['toxicity', 'severe_toxicity', 'identity_attack', 'insult', 'profanity', 'threat']
    
    # Create a combined dataframe to store all results
    results = {}
    
    # Calculate statistics for each metric
    print("\nCalculating statistics for each metric...")
    
    # Create summary table
    summary_rows = []
    
    for metric in metrics:
        # Extract the data for this metric
        try:
            prompt_values = prompt_df[metric].dropna()
            llama_values = llama_df[metric].dropna()
            aya_values = aya_df[metric].dropna()
            
            prompt_mean = prompt_values.mean() if not prompt_values.empty else np.nan
            llama_mean = llama_values.mean() if not llama_values.empty else np.nan
            aya_mean = aya_values.mean() if not aya_values.empty else np.nan
            
            # Store in results
            results[metric] = {
                'prompt_mean': prompt_mean,
                'llama_mean': llama_mean,
                'aya_mean': aya_mean,
                'prompt_values': prompt_values.tolist(),
                'llama_values': llama_values.tolist(),
                'aya_values': aya_values.tolist()
            }
            
            # Try to calculate p-values between prompt and models
            try:
                # Ensure there are at least 2 valid paired values for t-test
                prompt_llama_paired = pd.DataFrame({
                    'prompt': prompt_values[:min(len(prompt_values), len(llama_values))],
                    'llama': llama_values[:min(len(prompt_values), len(llama_values))]
                }).dropna()
                
                if len(prompt_llama_paired) >= 2:
                    t_stat_llama, p_val_llama = stats.ttest_rel(
                        prompt_llama_paired['prompt'], 
                        prompt_llama_paired['llama']
                    )
                    results[metric]['llama_p_value'] = p_val_llama
                else:
                    results[metric]['llama_p_value'] = np.nan
            except Exception as e:
                print(f"Error calculating LLaMA p-value for {metric}: {e}")
                results[metric]['llama_p_value'] = np.nan
            
            try:
                # Ensure there are at least 2 valid paired values for t-test
                prompt_aya_paired = pd.DataFrame({
                    'prompt': prompt_values[:min(len(prompt_values), len(aya_values))],
                    'aya': aya_values[:min(len(prompt_values), len(aya_values))]
                }).dropna()
                
                if len(prompt_aya_paired) >= 2:
                    t_stat_aya, p_val_aya = stats.ttest_rel(
                        prompt_aya_paired['prompt'], 
                        prompt_aya_paired['aya']
                    )
                    results[metric]['aya_p_value'] = p_val_aya
                else:
                    results[metric]['aya_p_value'] = np.nan
            except Exception as e:
                print(f"Error calculating Aya p-value for {metric}: {e}")
                results[metric]['aya_p_value'] = np.nan
            
            # Add row to summary table
            summary_rows.append({
                'Metric': metric,
                'Prompt Mean': prompt_mean,
                'LLaMA Mean': llama_mean,
                'Aya Mean': aya_mean,
                'LLaMA p-value': results[metric].get('llama_p_value', np.nan),
                'Aya p-value': results[metric].get('aya_p_value', np.nan)
            })
            
        except Exception as e:
            print(f"Error processing metric {metric}: {e}")
    
    # Create and save the summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(analysis_dir, 'toxicity_summary.csv'), index=False)
    print(f"Saved summary to {os.path.join(analysis_dir, 'toxicity_summary.csv')}")
    print("\nSummary table:")
    print(summary_df)
    
    # Create combined dataframe with all the data
    print("\nCreating combined dataframe with all metrics...")
    
    # Create a DataFrame for each source
    try:
        all_metrics = pd.DataFrame()
        
        # Add a row index for joining
        if not prompt_df.empty:
            prompt_df = prompt_df.reset_index(drop=True)
            prompt_df['row_id'] = prompt_df.index
        
        if not llama_df.empty:
            llama_df = llama_df.reset_index(drop=True)
            llama_df['row_id'] = llama_df.index
        
        if not aya_df.empty:
            aya_df = aya_df.reset_index(drop=True)
            aya_df['row_id'] = aya_df.index
        
        # Rename columns to add source prefix
        prompt_renamed = prompt_df.copy()
        llama_renamed = llama_df.copy()
        aya_renamed = aya_df.copy()
        
        for metric in metrics:
            if metric in prompt_renamed.columns:
                prompt_renamed = prompt_renamed.rename(columns={metric: f"{metric}_prompt"})
            if metric in llama_renamed.columns:
                llama_renamed = llama_renamed.rename(columns={metric: f"{metric}_llama"})
            if metric in aya_renamed.columns:
                aya_renamed = aya_renamed.rename(columns={metric: f"{metric}_aya"})
        
        # Save the original comment column from each
        if 'comment' in prompt_renamed.columns:
            prompt_renamed = prompt_renamed.rename(columns={'comment': 'prompt'})
        
        # Keep only relevant columns
        keep_cols_prompt = ['row_id', 'prompt'] + [f"{m}_prompt" for m in metrics if f"{m}_prompt" in prompt_renamed.columns]
        keep_cols_llama = ['row_id'] + [f"{m}_llama" for m in metrics if f"{m}_llama" in llama_renamed.columns]
        keep_cols_aya = ['row_id'] + [f"{m}_aya" for m in metrics if f"{m}_aya" in aya_renamed.columns]
        
        prompt_renamed = prompt_renamed[keep_cols_prompt] if all(col in prompt_renamed.columns for col in keep_cols_prompt) else prompt_renamed
        llama_renamed = llama_renamed[keep_cols_llama] if all(col in llama_renamed.columns for col in keep_cols_llama) else llama_renamed
        aya_renamed = aya_renamed[keep_cols_aya] if all(col in aya_renamed.columns for col in keep_cols_aya) else aya_renamed
        
        # Join dataframes on row_id
        all_metrics = prompt_renamed
        
        if not llama_renamed.empty:
            # Make sure row_id exists in both dataframes
            common_ids = set(all_metrics['row_id']).intersection(set(llama_renamed['row_id']))
            if common_ids:
                all_metrics = pd.merge(
                    all_metrics, 
                    llama_renamed,
                    on='row_id',
                    how='outer'
                )
            else:
                # Fall back to just appending columns
                for col in llama_renamed.columns:
                    if col != 'row_id':
                        all_metrics[col] = llama_renamed[col].values[:len(all_metrics)] if len(llama_renamed) > 0 else np.nan
        
        if not aya_renamed.empty:
            # Make sure row_id exists in both dataframes
            common_ids = set(all_metrics['row_id']).intersection(set(aya_renamed['row_id']))
            if common_ids:
                all_metrics = pd.merge(
                    all_metrics, 
                    aya_renamed,
                    on='row_id',
                    how='outer'
                )
            else:
                # Fall back to just appending columns
                for col in aya_renamed.columns:
                    if col != 'row_id':
                        all_metrics[col] = aya_renamed[col].values[:len(all_metrics)] if len(aya_renamed) > 0 else np.nan
        
        # Save the combined metrics
        all_metrics.to_csv(os.path.join(analysis_dir, 'all_toxicity_metrics.csv'), index=False)
        print(f"Saved all metrics to {os.path.join(analysis_dir, 'all_toxicity_metrics.csv')}")
        print(f"All metrics dataframe shape: {all_metrics.shape}")
        print("All metrics columns:", all_metrics.columns.tolist())
    
    except Exception as e:
        print(f"Error creating combined metrics file: {e}")
    
    # Create visualizations
    try:
        # Prepare data for plotting
        plot_data = []
        
        for metric in metrics:
            if metric in results:
                prompt_mean = results[metric].get('prompt_mean')
                llama_mean = results[metric].get('llama_mean')
                aya_mean = results[metric].get('aya_mean')
                
                if not np.isnan(prompt_mean):
                    plot_data.append({
                        'Metric': metric.replace('_', ' ').title(),
                        'Source': 'Prompt',
                        'Mean': prompt_mean
                    })
                
                if not np.isnan(llama_mean):
                    plot_data.append({
                        'Metric': metric.replace('_', ' ').title(),
                        'Source': 'LLaMA',
                        'Mean': llama_mean
                    })
                
                if not np.isnan(aya_mean):
                    plot_data.append({
                        'Metric': metric.replace('_', ' ').title(),
                        'Source': 'Aya',
                        'Mean': aya_mean
                    })
        
        plot_df = pd.DataFrame(plot_data)
        
        if not plot_df.empty:
            plt.figure(figsize=(15, 10))
            ax = sns.barplot(x='Metric', y='Mean', hue='Source', data=plot_df)
            plt.title('Toxicity Comparison', fontsize=16)
            plt.xlabel('Toxicity Metric', fontsize=14)
            plt.ylabel('Mean Score', fontsize=14)
            plt.xticks(rotation=45)
            
            # Add value labels on top of bars
            for i, bar in enumerate(ax.patches):
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 0.01,
                        f'{height:.3f}',
                        ha='center',
                        fontsize=10
                    )
            
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, 'toxicity_comparison.png'))
            plt.savefig(os.path.join(analysis_dir, 'toxicity_comparison.pdf'))  # Also save as PDF
            plt.close()
            print(f"Created visualization at {os.path.join(analysis_dir, 'toxicity_comparison.png')}")
    
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Direct analysis of toxicity data without merging")
    parser.add_argument("--prompt_toxicity", required=True, help="Path to prompt toxicity CSV file")
    parser.add_argument("--llama_toxicity", required=True, help="Path to LLaMA toxicity CSV file")
    parser.add_argument("--aya_toxicity", required=True, help="Path to Aya toxicity CSV file")
    parser.add_argument("--output_dir", required=True, help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the analysis
    direct_analysis(args.prompt_toxicity, args.llama_toxicity, args.aya_toxicity, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to {os.path.join(args.output_dir, 'analysis')}")

if __name__ == "__main__":
    main()
