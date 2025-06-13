import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr, zscore
import gc
import torch

# Add project root to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HUGGINGFACE_API_KEY

from transformers import AutoModelForCausalLM, AutoTokenizer

# ========== CONFIGURATION ==========
FILES = [
    # Original files with input columns
    ("new_outputs/perspective_full/aya_continuations_perspective_local_full.csv", "generated", "perspective_aya_continuation_toxicity", "CohereForAI/aya-23-8B"),
    ("new_outputs/perspective_full/llama3_continuations_perspective_local_full.csv", "generated", "perspective_llama3_continuation_toxicity", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("new_outputs/perspective_full/llama31_continuations_perspective_local_full.csv", "generated", "perspective_llama31_continuation_toxicity", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("new_outputs/src_results_full/aya_src_continuations_full.csv", "src", "perspective_src_toxicity", "CohereForAI/aya-23-8B"),
    ("new_outputs/src_results_full/llama3_src_continuations_full.csv", "src", "perspective_src_toxicity", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("new_outputs/src_results_full/llama31_src_continuations_full.csv", "src", "perspective_src_toxicity", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("new_outputs/tgt_results_full/aya_tgt_continuations_full.csv", "tgt", "perspective_tgt_toxicity", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("new_outputs/tgt_results_full/llama3_tgt_continuations_full.csv", "tgt", "perspective_tgt_toxicity", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("new_outputs/tgt_results_full/llama31_tgt_continuations_full.csv", "tgt", "perspective_tgt_toxicity", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    
    # New perspective_continuations files with input columns
    ("new_outputs/perspective_continuations/aya_src_perspective.csv", "src", "perspective_src_toxicity", "CohereForAI/aya-23-8B"),
    ("new_outputs/perspective_continuations/llama3_src_perspective.csv", "src", "perspective_src_toxicity", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("new_outputs/perspective_continuations/llama31_src_perspective.csv", "src", "perspective_src_toxicity", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("new_outputs/perspective_continuations/aya_tgt_perspective.csv", "tgt", "perspective_tgt_toxicity", "CohereForAI/aya-23-8B"),
    ("new_outputs/perspective_continuations/llama3_tgt_perspective.csv", "tgt", "perspective_tgt_toxicity", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("new_outputs/perspective_continuations/llama31_tgt_perspective.csv", "tgt", "perspective_tgt_toxicity", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    
    # New perspective_continuations files with output columns
    ("new_outputs/perspective_continuations/aya_src_perspective.csv", "generated", "perspective_aya_continuation_toxicity", "CohereForAI/aya-23-8B"),
    ("new_outputs/perspective_continuations/llama3_src_perspective.csv", "generated", "perspective_llama3_continuation_toxicity", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("new_outputs/perspective_continuations/llama31_src_perspective.csv", "generated", "perspective_llama31_continuation_toxicity", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("new_outputs/perspective_continuations/aya_tgt_perspective.csv", "generated", "perspective_aya_continuation_toxicity", "CohereForAI/aya-23-8B"),
    ("new_outputs/perspective_continuations/llama3_tgt_perspective.csv", "generated", "perspective_llama3_continuation_toxicity", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("new_outputs/perspective_continuations/llama31_tgt_perspective.csv", "generated", "perspective_llama31_continuation_toxicity", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
]
N_ROWS = None  # Set to None to use all rows
OUTPUT_DIR = "python_scripts/output/perplexity_toxicity/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set HuggingFace API key for transformers
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_HUB_TOKEN"] = HUGGINGFACE_API_KEY

# ========== HELPER FUNCTIONS ==========
def compute_perplexity(text, model, tokenizer, device='cpu'):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return torch.exp(loss).item()

def compute_correlation_with_significance(df, col1, col2):
    """Compute Pearson correlation and p-value between two columns."""
    valid_data = df[[col1, col2]].dropna()
    if len(valid_data) < 3:
        return np.nan, 1.0
    corr, p_value = pearsonr(valid_data[col1], valid_data[col2])
    return corr, p_value

def remove_outliers_iqr(df, columns):
    """Remove outliers using IQR method."""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def remove_outliers_zscore(df, columns, threshold=3):
    """Remove outliers using Z-score method."""
    df_clean = df.copy()
    for col in columns:
        z_scores = np.abs(zscore(df_clean[col].dropna()))
        df_clean = df_clean[z_scores < threshold]
    return df_clean

def remove_outliers_percentile(df, columns, percentile=5):
    """Remove outliers using percentile method."""
    df_clean = df.copy()
    for col in columns:
        lower_percentile = df_clean[col].quantile(percentile / 100)
        upper_percentile = df_clean[col].quantile(1 - percentile / 100)
        df_clean = df_clean[(df_clean[col] >= lower_percentile) & (df_clean[col] <= upper_percentile)]
    return df_clean

def analyze_outliers_impact(df, perp_col, tox_col, filename):
    """Analyze the impact of outlier removal on correlations."""
    results = {}
    
    # Original correlation
    corr_orig, p_orig = compute_correlation_with_significance(df, perp_col, tox_col)
    results['original'] = {'corr': corr_orig, 'p_value': p_orig, 'n_samples': len(df.dropna())}
    
    # IQR method
    df_iqr = remove_outliers_iqr(df, [perp_col, tox_col])
    corr_iqr, p_iqr = compute_correlation_with_significance(df_iqr, perp_col, tox_col)
    results['iqr'] = {'corr': corr_iqr, 'p_value': p_iqr, 'n_samples': len(df_iqr.dropna())}
    
    # Z-score method
    df_zscore = remove_outliers_zscore(df, [perp_col, tox_col])
    corr_zscore, p_zscore = compute_correlation_with_significance(df_zscore, perp_col, tox_col)
    results['zscore'] = {'corr': corr_zscore, 'p_value': p_zscore, 'n_samples': len(df_zscore.dropna())}
    
    # Percentile method
    df_percentile = remove_outliers_percentile(df, [perp_col, tox_col])
    corr_percentile, p_percentile = compute_correlation_with_significance(df_percentile, perp_col, tox_col)
    results['percentile'] = {'corr': corr_percentile, 'p_value': p_percentile, 'n_samples': len(df_percentile.dropna())}
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Outlier Analysis: {filename}', fontsize=16)
    
    datasets = [
        (df, 'Original', axes[0, 0]),
        (df_iqr, 'IQR', axes[0, 1]),
        (df_zscore, 'Z-score', axes[1, 0]),
        (df_percentile, 'Percentile', axes[1, 1])
    ]
    
    method_mapping = {
        'Original': 'original',
        'IQR': 'iqr',
        'Z-score': 'zscore',
        'Percentile': 'percentile'
    }
    
    for data, title, ax in datasets:
        if len(data) > 0:
            ax.scatter(data[perp_col], data[tox_col], alpha=0.5)
            method_key = method_mapping[title]
            corr_val = results[method_key]['corr']
            p_val = results[method_key]['p_value']
            n_samples = len(data.dropna())
            ax.set_title(f'{title}\nCorr: {corr_val:.3f}, p: {p_val:.3f}, n: {n_samples}')
            ax.set_xlabel('Perplexity')
            ax.set_ylabel('Toxicity')
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f'{filename}_outlier_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

def load_existing_perplexity_results(filepath):
    """Load existing perplexity results if available."""
    out_csv = os.path.join(OUTPUT_DIR, os.path.basename(filepath).replace('.csv', '_perplexity.csv'))
    if os.path.exists(out_csv):
        print(f"  Loading existing perplexity results from {out_csv}")
        return pd.read_csv(out_csv)
    return None

def clear_model_memory():
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def load_model_safely(model_name):
    """Safely load model with error handling for compatibility issues."""
    try:
        print(f"  Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_API_KEY)
        model = AutoModelForCausalLM.from_pretrained(model_name, token=HUGGINGFACE_API_KEY)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        return model, tokenizer, device
    except Exception as e:
        print(f"  ERROR: Failed to load model {model_name}: {e}")
        print(f"  This might be due to incompatible transformers version or missing model components.")
        print(f"  Skipping perplexity calculation for this model.")
        return None, None, None

def create_correlation_heatmap_with_significance(combined_df, output_path, title_suffix=""):
    """Create a correlation heatmap with significance markers."""
    # Define the columns for analysis
    lang_cols = []
    perplexity_cols = []
    toxicity_cols = []
    
    for col in combined_df.columns:
        if 'english' in col.lower() or 'hindi' in col.lower() or 'hinglish' in col.lower():
            lang_cols.append(col)
        elif 'perplexity' in col.lower():
            perplexity_cols.append(col)
        elif 'toxicity' in col.lower():
            toxicity_cols.append(col)
    
    # Combine all columns for correlation matrix
    all_cols = lang_cols + perplexity_cols + toxicity_cols
    correlation_data = combined_df[all_cols]
    
    # Compute correlation matrix and p-values
    corr_matrix = correlation_data.corr()
    p_matrix = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
    
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i != j:
                _, p_val = compute_correlation_with_significance(correlation_data, col1, col2)
                p_matrix.loc[col1, col2] = p_val
            else:
                p_matrix.loc[col1, col2] = 0.0
    
    # Create significance mask (True where p < 0.05)
    sig_mask = p_matrix.astype(float) < 0.05
    
    # Create the heatmap with larger size
    plt.figure(figsize=(16, 14))
    
    # Create annotations with significance markers
    annot_matrix = corr_matrix.copy()
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            if sig_mask.iloc[i, j] and i != j:  # Don't mark diagonal
                annot_matrix.iloc[i, j] = f"{val:.3f}*"
            else:
                annot_matrix.iloc[i, j] = f"{val:.3f}"
    
    # Create heatmap with larger annotations
    sns.heatmap(corr_matrix, 
                annot=annot_matrix, 
                fmt='', 
                cmap='RdBu_r', 
                center=0,
                square=True,
                annot_kws={'size': 10},  # Larger annotation font
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title(f'Correlation Matrix: Languages, Perplexity, and Toxicity{title_suffix}\n(* indicates p < 0.05)', 
              fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return corr_matrix, p_matrix

def apply_outlier_removal_to_combined_data(combined_df, method='original'):
    """Apply outlier removal method to combined dataset."""
    if method == 'original':
        return combined_df
    
    # Get perplexity and toxicity columns
    perp_cols = [col for col in combined_df.columns if 'perplexity' in col.lower()]
    tox_cols = [col for col in combined_df.columns if 'toxicity' in col.lower()]
    
    # Apply outlier removal to all perplexity and toxicity columns
    all_numeric_cols = perp_cols + tox_cols
    
    if method == 'iqr':
        return remove_outliers_iqr(combined_df, all_numeric_cols)
    elif method == 'zscore':
        return remove_outliers_zscore(combined_df, all_numeric_cols)
    elif method == 'percentile':
        return remove_outliers_percentile(combined_df, all_numeric_cols)
    else:
        return combined_df

# Add cross-correlation analysis function
def analyze_cross_correlations(df, input_col, output_col, input_perp_col, output_perp_col, tox_col, model_name):
    """Analyze correlations between input perplexity, output perplexity, and toxicity."""
    results = {
        'input_perp_vs_tox': compute_correlation_with_significance(df, input_perp_col, tox_col),
        'output_perp_vs_tox': compute_correlation_with_significance(df, output_perp_col, tox_col),
        'input_perp_vs_output_perp': compute_correlation_with_significance(df, input_perp_col, output_perp_col)
    }
    
    # Create scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Cross-correlation Analysis: {model_name}', fontsize=14)
    
    # Input Perplexity vs Toxicity
    axes[0].scatter(df[input_perp_col], df[tox_col], alpha=0.5)
    axes[0].set_title(f'Input Perplexity vs Toxicity\nCorr: {results["input_perp_vs_tox"][0]:.3f}, p: {results["input_perp_vs_tox"][1]:.3f}')
    axes[0].set_xlabel('Input Perplexity')
    axes[0].set_ylabel('Toxicity')
    
    # Output Perplexity vs Toxicity
    axes[1].scatter(df[output_perp_col], df[tox_col], alpha=0.5)
    axes[1].set_title(f'Output Perplexity vs Toxicity\nCorr: {results["output_perp_vs_tox"][0]:.3f}, p: {results["output_perp_vs_tox"][1]:.3f}')
    axes[1].set_xlabel('Output Perplexity')
    axes[1].set_ylabel('Toxicity')
    
    # Input vs Output Perplexity
    axes[2].scatter(df[input_perp_col], df[output_perp_col], alpha=0.5)
    axes[2].set_title(f'Input vs Output Perplexity\nCorr: {results["input_perp_vs_output_perp"][0]:.3f}, p: {results["input_perp_vs_output_perp"][1]:.3f}')
    axes[2].set_xlabel('Input Perplexity')
    axes[2].set_ylabel('Output Perplexity')
    
    plt.tight_layout()
    return results, fig

# ========== MAIN SCRIPT ==========
def main():
    all_results = []
    outlier_summary = []
    
    for filepath, input_col, tox_col, model_name in FILES:
        print(f"\nProcessing: {filepath} | Model: {model_name}")
        
        # Check if perplexity results already exist
        existing_df = load_existing_perplexity_results(filepath)
        
        if existing_df is not None:
            # Use existing results
            df = existing_df
            print(f"  Using existing perplexity data with {len(df)} rows")
        else:
            # Try to compute new perplexity values
            model, tokenizer, device = load_model_safely(model_name)
            
            if model is None:
                # Model loading failed, create dummy data or skip
                print(f"  Skipping perplexity calculation due to model loading error")
                print(f"  Creating placeholder data for analysis...")
                
                # Load original data without perplexity
                df = pd.read_csv(filepath, usecols=[input_col, tox_col])
                if N_ROWS:
                    df = df.head(N_ROWS)
                
                # Add placeholder perplexity values (you might want to skip this file entirely)
                df['perplexity'] = np.nan
                
                # Save placeholder results
                out_csv = os.path.join(OUTPUT_DIR, os.path.basename(filepath).replace('.csv', '_perplexity.csv'))
                df.to_csv(out_csv, index=False)
                
                print(f"  Saved placeholder data to {out_csv}")
            else:
                # Model loaded successfully, compute perplexity
                # Load data
                df = pd.read_csv(filepath, usecols=[input_col, tox_col])
                if N_ROWS:
                    df = df.head(N_ROWS)
                
                perplexities = []
                for text in tqdm(df[input_col].fillna("").astype(str), desc=f"Perplexity: {os.path.basename(filepath)}"):
                    try:
                        perplexity = compute_perplexity(text, model, tokenizer, device)
                    except Exception as e:
                        print(f"Error with text: {text[:30]}... | {e}")
                        perplexity = None
                    perplexities.append(perplexity)
                
                df['perplexity'] = perplexities
                
                # Save individual results
                out_csv = os.path.join(OUTPUT_DIR, os.path.basename(filepath).replace('.csv', '_perplexity.csv'))
                df.to_csv(out_csv, index=False)
                
                # Clear model from memory
                del model, tokenizer
                clear_model_memory()
                print(f"  Model cleared from memory")
        
        # Only proceed with analysis if we have valid perplexity data
        if 'perplexity' in df.columns and not df['perplexity'].isna().all():
            # Analyze outliers impact
            filename = os.path.basename(filepath).replace('.csv', '')
            outlier_results = analyze_outliers_impact(df, 'perplexity', tox_col, filename)
            outlier_results['filename'] = filename
            outlier_summary.append(outlier_results)
            
            # Add metadata for combining datasets
            model_short = model_name.split('/')[-1].replace('-', '_')
            input_type = input_col  # 'generated', 'src', or 'tgt'
            
            # Rename columns to be more descriptive
            df_renamed = df.copy()
            df_renamed.columns = [
                f"{input_type}_{model_short}",
                f"toxicity_{input_type}_{model_short}",
                f"perplexity_{input_type}_{model_short}"
            ]
            
            all_results.append(df_renamed)
            
            # Individual scatter plot
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x='perplexity', y=tox_col, data=df, alpha=0.5)
            plt.title(f'Perplexity vs Toxicity\n{os.path.basename(filepath)} | Model: {model_name}')
            plt.xlabel('Perplexity')
            plt.ylabel('Toxicity')
            plt.tight_layout()
            plot_path = os.path.join(OUTPUT_DIR, os.path.basename(filepath).replace('.csv', '_scatter.png'))
            plt.savefig(plot_path)
            plt.close()
            
            # Individual correlation
            corr, p_val = compute_correlation_with_significance(df, 'perplexity', tox_col)
            print(f"  Pearson correlation: {corr:.3f} (p={p_val:.3f})")
            
            # Save correlation value
            corr_file = os.path.join(OUTPUT_DIR, os.path.basename(filepath).replace('.csv', '_correlation.txt'))
            with open(corr_file, 'w') as f:
                f.write(f"Pearson correlation between perplexity and toxicity: {corr:.3f}\n")
                f.write(f"P-value: {p_val:.3f}\n")
                f.write(f"Significant (p < 0.05): {'Yes' if p_val < 0.05 else 'No'}\n")
        else:
            print(f"  Skipping analysis for {filepath} due to missing perplexity data")
    
    # Save outlier analysis summary
    if outlier_summary:
        outlier_df = pd.DataFrame([
            {
                'filename': result['filename'],
                'original_corr': result['original']['corr'],
                'original_p': result['original']['p_value'],
                'original_n': result['original']['n_samples'],
                'iqr_corr': result['iqr']['corr'],
                'iqr_p': result['iqr']['p_value'],
                'iqr_n': result['iqr']['n_samples'],
                'zscore_corr': result['zscore']['corr'],
                'zscore_p': result['zscore']['p_value'],
                'zscore_n': result['zscore']['n_samples'],
                'percentile_corr': result['percentile']['corr'],
                'percentile_p': result['percentile']['p_value'],
                'percentile_n': result['percentile']['n_samples'],
            }
            for result in outlier_summary
        ])
        outlier_df.to_csv(os.path.join(OUTPUT_DIR, 'outlier_analysis_summary.csv'), index=False)
    
    # Combine all results for comprehensive analysis
    if all_results:
        print("\nCreating comprehensive correlation heatmaps...")
        
        # Combine all data
        combined_df = pd.concat(all_results, axis=1)
        
        # Create correlation heatmaps for all four conditions
        outlier_methods = [
            ('original', 'Original Data'),
            ('iqr', 'IQR Outlier Removal'),
            ('zscore', 'Z-score Outlier Removal'),
            ('percentile', 'Percentile Outlier Removal')
        ]
        
        correlation_matrices = {}
        p_value_matrices = {}
        
        for method, method_name in outlier_methods:
            print(f"  Creating heatmap for: {method_name}")
            
            # Apply outlier removal method
            processed_df = apply_outlier_removal_to_combined_data(combined_df, method)
            
            # Create heatmap
            output_path = os.path.join(OUTPUT_DIR, f'correlation_heatmap_{method}.png')
            title_suffix = f" - {method_name}"
            
            corr_matrix, p_matrix = create_correlation_heatmap_with_significance(
                processed_df, output_path, title_suffix
            )
            
            # Store matrices
            correlation_matrices[method] = corr_matrix
            p_value_matrices[method] = p_matrix
            
            # Save individual matrices
            corr_matrix.to_csv(os.path.join(OUTPUT_DIR, f'correlation_matrix_{method}.csv'))
            p_matrix.to_csv(os.path.join(OUTPUT_DIR, f'p_value_matrix_{method}.csv'))
        
        print(f"  Created 4 correlation heatmaps with different outlier removal methods")
    
    # Add cross-correlation analysis for each model
    cross_correlation_results = []
    
    for model_name in ['CohereForAI/aya-23-8B', 'meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Meta-Llama-3.1-8B-Instruct']:
        model_short = model_name.split('/')[-1].replace('-', '_')
        
        # Get all files for this model
        model_files = [f for f in FILES if f[3] == model_name]
        
        for filepath, input_col, tox_col, _ in model_files:
            if 'perspective_continuations' in filepath:
                # Load data with perplexity scores
                df = pd.read_csv(os.path.join(OUTPUT_DIR, os.path.basename(filepath).replace('.csv', '_perplexity.csv')))
                
                # Analyze cross-correlations
                results, fig = analyze_cross_correlations(
                    df,
                    input_col,
                    'generated',
                    'perplexity',
                    'generated_perplexity',
                    tox_col,
                    model_name
                )
                
                # Save plot
                plot_path = os.path.join(OUTPUT_DIR, f'cross_correlation_{model_short}_{os.path.basename(filepath).replace(".csv", "")}.png')
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                # Store results
                cross_correlation_results.append({
                    'model': model_name,
                    'file': filepath,
                    'input_perp_vs_tox_corr': results['input_perp_vs_tox'][0],
                    'input_perp_vs_tox_p': results['input_perp_vs_tox'][1],
                    'output_perp_vs_tox_corr': results['output_perp_vs_tox'][0],
                    'output_perp_vs_tox_p': results['output_perp_vs_tox'][1],
                    'input_vs_output_perp_corr': results['input_perp_vs_output_perp'][0],
                    'input_vs_output_perp_p': results['input_perp_vs_output_perp'][1]
                })
    
    # Save cross-correlation results
    if cross_correlation_results:
        cross_corr_df = pd.DataFrame(cross_correlation_results)
        cross_corr_df.to_csv(os.path.join(OUTPUT_DIR, 'cross_correlation_results.csv'), index=False)
        print("\nCross-correlation analysis completed and saved to 'cross_correlation_results.csv'")
    
    print(f"Results saved in {OUTPUT_DIR}")
    print("\nOutlier Analysis Summary:")
    print("Check 'outlier_analysis_summary.csv' for detailed comparison of correlations before/after outlier removal")
    print("Check correlation_heatmap_*.png files for comprehensive correlation matrices")

if __name__ == '__main__':
    main() 