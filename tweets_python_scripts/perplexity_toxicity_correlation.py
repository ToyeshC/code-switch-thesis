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

try:
    from config import HUGGINGFACE_API_KEY
    os.environ["HF_HUB_TOKEN"] = HUGGINGFACE_API_KEY
except ImportError:
    print("Warning: Could not import HUGGINGFACE_API_KEY from config")

from transformers import AutoModelForCausalLM, AutoTokenizer

# ========== CONFIGURATION ==========
FILES = [
    # (filepath, input_col, toxicity_col, model_name)
    ("tweets_outputs/perspective_small/llama3_continuations_perspective_small.csv", "generated", "perspective_llama3_continuation_toxicity", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("tweets_outputs/perspective_small/llama31_continuations_perspective_small.csv", "generated", "perspective_llama31_continuation_toxicity", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("tweets_outputs/perspective_small/aya_continuations_perspective_small.csv", "generated", "perspective_aya_continuation_toxicity", "CohereForAI/aya-23-8B"),
]

N_ROWS = None  # Set to None to use all rows, or an int for quick testing
OUTPUT_DIR = "tweets_outputs/correlation_analysis/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== HELPER FUNCTIONS ==========
def compute_perplexity(text, model, tokenizer, device='cpu'):
    """Compute perplexity for a given text."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.nan
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        return torch.exp(loss).item()
    except Exception as e:
        print(f"Error computing perplexity: {str(e)}")
        return np.nan

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

def create_correlation_heatmap_with_significance(combined_df, output_path, title_suffix=""):
    """Create a correlation heatmap with significance markers."""
    # Define the columns for analysis
    lang_cols = []
    perplexity_cols = []
    toxicity_cols = []
    
    for col in combined_df.columns:
        if any(lang in col.lower() for lang in ['english', 'hindi', 'hinglish', 'percent']):
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

def load_model_safely(model_name):
    """Load model and tokenizer with error handling."""
    try:
        print(f"Loading model: {model_name}")
        
        # Special handling for different models
        if "llama-3.1" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            from transformers import LlamaConfig
            config = LlamaConfig.from_pretrained(model_name)
            
            # Simplify the rope_scaling to what transformers expects
            if hasattr(config, 'rope_scaling') and config.rope_scaling:
                config.rope_scaling = {
                    "type": "linear",
                    "factor": config.rope_scaling.get('factor', 8.0)
                }
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        # Ensure tokenizer has a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        device = next(model.parameters()).device
        return model, tokenizer, device
        
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        return None, None, None

def clear_model_memory():
    """Clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()

def load_existing_perplexity_results(filepath):
    """Load existing perplexity results if available."""
    out_csv = os.path.join(OUTPUT_DIR, os.path.basename(filepath).replace('.csv', '_perplexity.csv'))
    if os.path.exists(out_csv):
        print(f"  Loading existing perplexity results from {out_csv}")
        return pd.read_csv(out_csv)
    return None

# ========== MAIN SCRIPT ==========
def main():
    """Main function to run comprehensive toxicity-perplexity correlation analysis."""
    all_results = []
    outlier_summary = []
    
    print("Starting comprehensive toxicity-perplexity correlation analysis...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    for filepath, input_col, tox_col, model_name in FILES:
        print(f"\nProcessing: {filepath} | Model: {model_name}")
        
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"  File not found: {filepath}")
            continue
        
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
                try:
                    df = pd.read_csv(filepath, usecols=[input_col, tox_col])
                    if N_ROWS:
                        df = df.head(N_ROWS)
                    
                    # Add placeholder perplexity values (you might want to skip this file entirely)
                    df['perplexity'] = np.nan
                    
                    # Save placeholder results
                    out_csv = os.path.join(OUTPUT_DIR, os.path.basename(filepath).replace('.csv', '_perplexity.csv'))
                    df.to_csv(out_csv, index=False)
                    
                    print(f"  Saved placeholder data to {out_csv}")
                except Exception as e:
                    print(f"  Error loading data: {e}")
                    continue
            else:
                # Model loaded successfully, compute perplexity
                try:
                    # Load data
                    df = pd.read_csv(filepath)
                    if N_ROWS:
                        df = df.head(N_ROWS)
                    
                    print(f"  Computing perplexity for {len(df)} rows...")
                    perplexities = []
                    for text in tqdm(df[input_col].fillna("").astype(str), desc=f"Perplexity: {os.path.basename(filepath)}"):
                        try:
                            perplexity = compute_perplexity(text, model, tokenizer, device)
                        except Exception as e:
                            print(f"Error with text: {text[:30]}... | {e}")
                            perplexity = np.nan
                        perplexities.append(perplexity)
                    
                    df['perplexity'] = perplexities
                    
                    # Save individual results
                    out_csv = os.path.join(OUTPUT_DIR, os.path.basename(filepath).replace('.csv', '_perplexity.csv'))
                    df.to_csv(out_csv, index=False)
                    print(f"  Saved perplexity results to {out_csv}")
                    
                    # Clear model from memory
                    del model, tokenizer
                    clear_model_memory()
                    print(f"  Model cleared from memory")
                except Exception as e:
                    print(f"  Error processing data: {e}")
                    continue
        
        # Only proceed with analysis if we have valid perplexity data
        if 'perplexity' in df.columns and not df['perplexity'].isna().all():
            # Analyze outliers impact
            filename = os.path.basename(filepath).replace('.csv', '')
            print(f"  Analyzing outlier impact for {filename}...")
            outlier_results = analyze_outliers_impact(df, 'perplexity', tox_col, filename)
            outlier_results['filename'] = filename
            outlier_summary.append(outlier_results)
            
            # Add metadata for combining datasets
            model_short = model_name.split('/')[-1].replace('-', '_').replace('.', '_')
            input_type = input_col  # 'generated', 'src', or 'tgt'
            
            # Create a copy with language composition data if available
            df_for_combination = df.copy()
            
            # Add language composition columns if they exist
            lang_cols = ['english_word_count', 'total_hindi_count', 'total_words', 
                        'total_hindi_percent', 'english_percent']
            available_lang_cols = [col for col in lang_cols if col in df.columns]
            
            # Rename columns to be more descriptive for combination
            rename_dict = {
                input_col: f"{input_type}_{model_short}",
                tox_col: f"toxicity_{input_type}_{model_short}",
                'perplexity': f"perplexity_{input_type}_{model_short}"
            }
            
            # Add language columns with model-specific names
            for col in available_lang_cols:
                rename_dict[col] = f"{col}_{model_short}"
            
            df_renamed = df_for_combination[list(rename_dict.keys())].rename(columns=rename_dict)
            all_results.append(df_renamed)
            
            # Individual scatter plot
            print(f"  Creating individual scatter plot...")
            plt.figure(figsize=(10, 8))
            valid_data = df[['perplexity', tox_col]].dropna()
            if len(valid_data) > 0:
                sns.scatterplot(x='perplexity', y=tox_col, data=valid_data, alpha=0.6)
                
                # Add correlation info to plot
                corr, p_val = compute_correlation_with_significance(df, 'perplexity', tox_col)
                plt.title(f'Perplexity vs Toxicity\n{os.path.basename(filepath)}\nModel: {model_name}\nCorr: {corr:.3f}, p: {p_val:.3f}')
                plt.xlabel('Perplexity')
                plt.ylabel('Toxicity Score')
                plt.grid(True, alpha=0.3)
                
                # Add trend line
                if not np.isnan(corr):
                    z = np.polyfit(valid_data['perplexity'], valid_data[tox_col], 1)
                    p = np.poly1d(z)
                    plt.plot(valid_data['perplexity'], p(valid_data['perplexity']), "r--", alpha=0.8)
            
            plt.tight_layout()
            plot_path = os.path.join(OUTPUT_DIR, os.path.basename(filepath).replace('.csv', '_scatter.png'))
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
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
                f.write(f"Sample size: {len(df[['perplexity', tox_col]].dropna())}\n")
        else:
            print(f"  Skipping analysis for {filepath} due to missing perplexity data")
    
    # Save outlier analysis summary
    if outlier_summary:
        print("\nCreating outlier analysis summary...")
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
        print(f"  Saved outlier analysis summary to outlier_analysis_summary.csv")
    
    # Combine all results for comprehensive analysis
    if all_results:
        print("\nCreating comprehensive correlation heatmaps...")
        
        # Combine all data
        combined_df = pd.concat(all_results, axis=1)
        print(f"  Combined dataset shape: {combined_df.shape}")
        
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
            print(f"    Dataset size after {method} filtering: {processed_df.shape}")
            
            # Create heatmap
            output_path = os.path.join(OUTPUT_DIR, f'correlation_heatmap_{method}.png')
            title_suffix = f" - {method_name}"
            
            try:
                corr_matrix, p_matrix = create_correlation_heatmap_with_significance(
                    processed_df, output_path, title_suffix
                )
                
                # Store matrices
                correlation_matrices[method] = corr_matrix
                p_value_matrices[method] = p_matrix
                
                # Save individual matrices
                corr_matrix.to_csv(os.path.join(OUTPUT_DIR, f'correlation_matrix_{method}.csv'))
                p_matrix.to_csv(os.path.join(OUTPUT_DIR, f'p_value_matrix_{method}.csv'))
                
                print(f"    Saved correlation heatmap: correlation_heatmap_{method}.png")
            except Exception as e:
                print(f"    Error creating heatmap for {method}: {e}")
        
        print(f"  Created correlation heatmaps with different outlier removal methods")
        
        # Create a summary comparison plot
        print("  Creating method comparison visualization...")
        if outlier_summary:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Correlation Analysis: Impact of Outlier Removal Methods', fontsize=16)
            
            methods = ['original', 'iqr', 'zscore', 'percentile']
            method_names = ['Original', 'IQR', 'Z-score', 'Percentile']
            
            # Extract correlations for each method
            for i, (method, method_name) in enumerate(zip(methods, method_names)):
                ax = axes[i//2, i%2]
                
                correlations = [result[method]['corr'] for result in outlier_summary if not np.isnan(result[method]['corr'])]
                p_values = [result[method]['p_value'] for result in outlier_summary if not np.isnan(result[method]['p_value'])]
                filenames = [result['filename'] for result in outlier_summary if not np.isnan(result[method]['corr'])]
                
                if correlations:
                    bars = ax.bar(range(len(correlations)), correlations, alpha=0.7)
                    ax.set_title(f'{method_name} Method\nMean Corr: {np.mean(correlations):.3f}')
                    ax.set_ylabel('Correlation')
                    ax.set_xlabel('Dataset')
                    ax.set_xticks(range(len(filenames)))
                    ax.set_xticklabels([f.split('_')[0] for f in filenames], rotation=45)
                    ax.grid(True, alpha=0.3)
                    
                    # Color bars based on significance
                    for j, (bar, p_val) in enumerate(zip(bars, p_values)):
                        if p_val < 0.05:
                            bar.set_color('darkgreen')
                        else:
                            bar.set_color('lightcoral')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'method_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    Saved method comparison plot: method_comparison.png")
    
    # Generate final summary report
    print("\nGenerating final summary report...")
    summary_lines = []
    summary_lines.append("TWEETS DATASET - TOXICITY-PERPLEXITY CORRELATION ANALYSIS REPORT")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"Number of datasets analyzed: {len([f for f in FILES if os.path.exists(f[0])])}")
    summary_lines.append("")
    
    summary_lines.append("DATASETS PROCESSED:")
    summary_lines.append("-" * 20)
    for filepath, input_col, tox_col, model_name in FILES:
        if os.path.exists(filepath):
            summary_lines.append(f"✓ {os.path.basename(filepath)} - {model_name}")
        else:
            summary_lines.append(f"✗ {os.path.basename(filepath)} - FILE NOT FOUND")
    summary_lines.append("")
    
    if outlier_summary:
        summary_lines.append("CORRELATION SUMMARY:")
        summary_lines.append("-" * 20)
        for result in outlier_summary:
            summary_lines.append(f"{result['filename']}:")
            summary_lines.append(f"  Original: r={result['original']['corr']:.3f}, p={result['original']['p_value']:.3f}")
            summary_lines.append(f"  IQR:      r={result['iqr']['corr']:.3f}, p={result['iqr']['p_value']:.3f}")
            summary_lines.append(f"  Z-score:  r={result['zscore']['corr']:.3f}, p={result['zscore']['p_value']:.3f}")
            summary_lines.append(f"  Percentile: r={result['percentile']['corr']:.3f}, p={result['percentile']['p_value']:.3f}")
            summary_lines.append("")
    
    summary_lines.append("GENERATED FILES:")
    summary_lines.append("-" * 16)
    output_files = [
        "outlier_analysis_summary.csv",
        "correlation_heatmap_original.png",
        "correlation_heatmap_iqr.png", 
        "correlation_heatmap_zscore.png",
        "correlation_heatmap_percentile.png",
        "method_comparison.png"
    ]
    
    for file in output_files:
        if os.path.exists(os.path.join(OUTPUT_DIR, file)):
            summary_lines.append(f"  ✓ {file}")
    
    # Individual files
    for filepath, _, _, _ in FILES:
        if os.path.exists(filepath):
            base_name = os.path.basename(filepath).replace('.csv', '')
            individual_files = [
                f"{base_name}_perplexity.csv",
                f"{base_name}_scatter.png",
                f"{base_name}_outlier_analysis.png",
                f"{base_name}_correlation.txt"
            ]
            for file in individual_files:
                if os.path.exists(os.path.join(OUTPUT_DIR, file)):
                    summary_lines.append(f"  ✓ {file}")
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, 'analysis_summary_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\nAnalysis completed!")
    print(f"Results saved in: {OUTPUT_DIR}")
    print(f"Summary report: {report_path}")
    print("\nKey outputs:")
    print("- Individual scatter plots for each model")
    print("- Outlier analysis plots (4 methods per model)")
    print("- Correlation heatmaps (4 outlier removal methods)")
    print("- Method comparison visualization")
    print("- Comprehensive CSV summaries")

if __name__ == '__main__':
    main() 