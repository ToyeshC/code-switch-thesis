import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os

# Set style for better visualizations
try:
    plt.style.use('seaborn-v0_8')  # Updated style name
except:
    plt.style.use('default')  # Fallback to default style if seaborn is not available
sns.set_theme()  # Use seaborn's default theme

def load_data(file_path):
    """Load CSV file and return DataFrame"""
    return pd.read_csv(file_path)

def plot_distributions(df, title, save_path, metrics):
    """Plot distributions of toxicity scores"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title)
    
    for i, metric in enumerate(metrics):
        sns.histplot(data=df, x=metric, ax=axes[i], bins=30)
        axes[i].set_title(f'Distribution of {metric}')
        axes[i].set_xlabel('Score')
        axes[i].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_model_comparison(dfs, model_names, metric, save_path):
    """Plot comparison of scores between models"""
    plt.figure(figsize=(10, 6))
    
    for df, name in zip(dfs, model_names):
        sns.kdeplot(data=df[metric], label=name)
    
    plt.title(f'Comparison of {metric} across models')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_correlation_heatmap(dfs, model_names, metrics, save_path):
    """Plot correlation heatmap between models"""
    n_models = len(model_names)
    
    # Create correlation matrix
    corr_matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            if i <= j:
                # Calculate average correlation across all metrics
                correlations = []
                for metric in metrics:
                    corr = dfs[i][metric].corr(dfs[j][metric])
                    correlations.append(corr)
                corr_matrix[i, j] = np.mean(correlations)
                corr_matrix[j, i] = corr_matrix[i, j]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, xticklabels=model_names, yticklabels=model_names,
                cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation between models (average across all metrics)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_box_plots(dfs, model_names, metrics, save_path):
    """Plot box plots comparing all metrics across models"""
    plt.figure(figsize=(15, 8))
    
    # Combine data for box plot
    combined_data = []
    for df, name in zip(dfs, model_names):
        for metric in metrics:
            temp_df = pd.DataFrame({
                'Score': df[metric],
                'Metric': metric,
                'Model': name
            })
            combined_data.append(temp_df)
    
    combined_df = pd.concat(combined_data)
    
    sns.boxplot(data=combined_df, x='Metric', y='Score', hue='Model')
    plt.title('Box plots of toxicity metrics across models')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Create output directory
    output_dir = Path('temp_scripts/toxicity_analysis_outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Define metrics for different file types
    src_metrics = ['perspective_src_toxicity', 'perspective_src_identity_attack', 'perspective_src_insult']
    tgt_metrics = ['perspective_tgt_toxicity', 'perspective_tgt_identity_attack', 'perspective_tgt_insult']
    cs_metrics = ['perspective_generated_toxicity', 'perspective_generated_identity_attack', 'perspective_generated_insult']
    
    # Analyze source results
    src_files = [
        'new_outputs/src_results_full/llama3_src_continuations_full.csv',
        'new_outputs/src_results_full/aya_src_continuations_full.csv',
        'new_outputs/src_results_full/llama31_src_continuations_full.csv'
    ]
    src_model_names = ['Llama3', 'Aya', 'Llama31']
    
    src_dfs = [load_data(f) for f in src_files]
    
    # Plot distributions for each model
    for df, name in zip(src_dfs, src_model_names):
        plot_distributions(df, f'Toxicity Score Distributions - {name}',
                         output_dir / f'src_{name.lower()}_distributions.png', src_metrics)
    
    # Plot model comparisons
    for metric in src_metrics:
        plot_model_comparison(src_dfs, src_model_names, metric,
                            output_dir / f'src_{metric}_comparison.png')
    
    # Plot correlation heatmap
    plot_correlation_heatmap(src_dfs, src_model_names, src_metrics,
                           output_dir / 'src_correlation_heatmap.png')
    
    # Plot box plots
    plot_box_plots(src_dfs, src_model_names, src_metrics,
                  output_dir / 'src_box_plots.png')
    
    # Analyze target results
    tgt_files = [
        'new_outputs/tgt_results_full/llama3_tgt_continuations_full.csv',
        'new_outputs/tgt_results_full/aya_tgt_continuations_full.csv',
        'new_outputs/tgt_results_full/llama31_tgt_continuations_full.csv'
    ]
    tgt_model_names = ['Llama3', 'Aya', 'Llama31']
    
    tgt_dfs = [load_data(f) for f in tgt_files]
    
    # Plot distributions for each model
    for df, name in zip(tgt_dfs, tgt_model_names):
        plot_distributions(df, f'Toxicity Score Distributions - {name}',
                         output_dir / f'tgt_{name.lower()}_distributions.png', tgt_metrics)
    
    # Plot model comparisons
    for metric in tgt_metrics:
        plot_model_comparison(tgt_dfs, tgt_model_names, metric,
                            output_dir / f'tgt_{metric}_comparison.png')
    
    # Plot correlation heatmap
    plot_correlation_heatmap(tgt_dfs, tgt_model_names, tgt_metrics,
                           output_dir / 'tgt_correlation_heatmap.png')
    
    # Plot box plots
    plot_box_plots(tgt_dfs, tgt_model_names, tgt_metrics,
                  output_dir / 'tgt_box_plots.png')
    
    # Analyze code-switched perspective data
    cs_file = 'new_outputs/perspective/code_switched_perspective.csv'
    cs_df = load_data(cs_file)
    
    # Plot distributions for code-switched data
    plot_distributions(cs_df, 'Code-Switched Perspective Score Distributions',
                     output_dir / 'code_switched_distributions.png', cs_metrics)

if __name__ == "__main__":
    main() 