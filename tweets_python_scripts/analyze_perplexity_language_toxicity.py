import pandas as pd
import numpy as np
import torch
import argparse
from tqdm import tqdm
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats

# Add project root to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available. Plots will be skipped.")

def load_model_and_tokenizer(model_name, device):
    """Load model and tokenizer for perplexity calculation."""
    print(f"Loading {model_name} model...")
    
    try:
        if model_name == "llama3":
            model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        elif model_name == "llama3.1":
            model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        elif model_name == "aya":
            model_path = "CohereForAI/aya-23-8B"
        else:
            print(f"Unknown model: {model_name}")
            return None, None
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Ensure tokenizer has a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading {model_name}: {str(e)}")
        return None, None

def compute_perplexity(text, model, tokenizer, device, model_name):
    """Compute perplexity for a given text."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.nan
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        return torch.exp(outputs.loss).item()
    except Exception as e:
        print(f"Error computing perplexity for {model_name}: {str(e)}")
        return np.nan

def main():
    parser = argparse.ArgumentParser(description="Analyze correlation between language, perplexity, and toxicity for tweets dataset.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--output_csv", type=str, default=None, help="Path to save output CSV with perplexity column.")
    parser.add_argument("--save_plots", action="store_true", help="Save correlation plots if matplotlib/seaborn are available.")
    parser.add_argument("--models", nargs="+", default=["llama3"], 
                      choices=["llama3", "llama3.1", "aya"],
                      help="Models to use for perplexity calculation")
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Process each model
    for model_name in args.models:
        print(f"\nProcessing with {model_name}...")
        model, tokenizer = load_model_and_tokenizer(model_name, device)
        if model is None or tokenizer is None:
            continue

        # Compute perplexity for the generated text
        tqdm.pandas()
        
        # Check which columns exist in the dataset
        if 'generated' in df.columns:
            print("Computing perplexity for 'generated' column...")
            df[f'generated_perplexity_{model_name}'] = df['generated'].progress_apply(
                lambda x: compute_perplexity(x, model, tokenizer, device, model_name))
        
        # If we have continuation columns, compute perplexity for those too
        continuation_col = f'{model_name}_continuation'
        if continuation_col in df.columns:
            print(f"Computing perplexity for '{continuation_col}' column...")
            df[f'{continuation_col}_perplexity'] = df[continuation_col].progress_apply(
                lambda x: compute_perplexity(x, model, tokenizer, device, model_name))

        # Analyze perplexity statistics
        print(f"\nPerplexity Analysis for {model_name}:")
        perplexity_cols = [col for col in df.columns if f'perplexity_{model_name}' in col or f'{model_name}_continuation_perplexity' in col]
        
        if perplexity_cols:
            print("\nMean perplexity:")
            print(df[perplexity_cols].mean())
            
            print("\nMedian perplexity:")
            print(df[perplexity_cols].median())
            
            print("\nPerplexity standard deviation:")
            print(df[perplexity_cols].std())

        # Clean up model from memory
        del model, tokenizer
        torch.cuda.empty_cache()

    # Save output CSV
    output_csv = args.output_csv or args.input_csv.replace('.csv', '_with_perplexity.csv')
    df.to_csv(output_csv, index=False)
    print(f"\nSaved output with perplexity to {output_csv}")

    # Correlation analysis
    print("\nComputing correlations...")
    cols_of_interest = []
    
    # Add language composition columns if they exist
    lang_cols = ['hindi_word_count', 'english_word_count', 'romanized_hindi_count', 
                'total_hindi_count', 'total_words', 'hindi_percent', 
                'romanized_hindi_percent', 'total_hindi_percent', 'english_percent']
    cols_of_interest.extend([c for c in lang_cols if c in df.columns])
    
    # Add toxicity columns if they exist
    tox_cols = [col for col in df.columns if 'toxicity' in col.lower() or 'identity_attack' in col.lower() or 'insult' in col.lower()]
    cols_of_interest.extend(tox_cols)
    
    # Add all perplexity columns
    perplexity_cols = [col for col in df.columns if 'perplexity' in col.lower()]
    cols_of_interest.extend(perplexity_cols)
    
    # Only keep columns that exist and have numeric data
    cols_of_interest = [c for c in cols_of_interest if c in df.columns]
    
    if len(cols_of_interest) > 1:
        # Convert to numeric, replacing non-numeric values with NaN
        numeric_df = df[cols_of_interest].apply(pd.to_numeric, errors='coerce')
        
        corr = numeric_df.corr(method='pearson')
        print("Correlation matrix:")
        print(corr)
        
        # Save correlation matrix
        corr_output = output_csv.replace('.csv', '_correlations.csv')
        corr.to_csv(corr_output)
        print(f"Saved correlation matrix to {corr_output}")

        # Optional: Plot heatmap and scatter plots
        if args.save_plots and HAS_PLOTTING:
            print("\nGenerating plots...")
            
            # Correlation heatmap
            plt.figure(figsize=(15, 12))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', center=0)
            plt.title("Correlation Heatmap: Language, Perplexity, Toxicity (Tweets Dataset)")
            plt.tight_layout()
            heatmap_output = output_csv.replace('.csv', '_correlation_heatmap.png')
            plt.savefig(heatmap_output)
            plt.close()
            
            # Perplexity distribution plots
            for col in perplexity_cols:
                if col in df.columns and not df[col].isna().all():
                    plt.figure(figsize=(10, 6))
                    df[col].dropna().hist(bins=50, alpha=0.7)
                    plt.title(f"Perplexity Distribution: {col}")
                    plt.xlabel("Perplexity")
                    plt.ylabel("Frequency")
                    plt.tight_layout()
                    hist_output = output_csv.replace('.csv', f'_{col}_distribution.png')
                    plt.savefig(hist_output)
                    plt.close()
            
            # Scatter plots of perplexity vs toxicity
            for perp_col in perplexity_cols:
                for tox_col in tox_cols:
                    if perp_col in df.columns and tox_col in df.columns:
                        valid_data = df[[perp_col, tox_col]].dropna()
                        if len(valid_data) > 10:  # Only plot if we have enough data
                            plt.figure(figsize=(10, 6))
                            plt.scatter(valid_data[perp_col], valid_data[tox_col], alpha=0.6)
                            plt.xlabel(perp_col)
                            plt.ylabel(tox_col)
                            plt.title(f"Perplexity vs Toxicity: {perp_col} vs {tox_col}")
                            
                            # Add correlation coefficient to plot
                            corr_val = valid_data[perp_col].corr(valid_data[tox_col])
                            plt.text(0.05, 0.95, f'Correlation: {corr_val:.3f}', 
                                   transform=plt.gca().transAxes, fontsize=12,
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                            
                            plt.tight_layout()
                            scatter_output = output_csv.replace('.csv', f'_{perp_col}_vs_{tox_col}_scatter.png')
                            plt.savefig(scatter_output)
                            plt.close()
            
            print("Plots saved.")
        elif args.save_plots:
            print("matplotlib/seaborn not available, skipping plots.")
    else:
        print("Not enough numeric columns for correlation analysis.")

if __name__ == "__main__":
    main() 