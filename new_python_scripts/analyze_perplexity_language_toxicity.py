import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import os

# Perplexity calculation imports
from transformers import (
    MT5ForConditionalGeneration, MT5Tokenizer,
    AutoModelForCausalLM, AutoTokenizer,
    LlamaForCausalLM, LlamaTokenizer
)
import torch

# Visualization imports (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

def load_model_and_tokenizer(model_name, device):
    """Load different models and their tokenizers."""
    print(f"Loading {model_name}...")
    try:
        if model_name == "mt5-xl":
            tokenizer = MT5Tokenizer.from_pretrained("google/mt5-xl")
            model = MT5ForConditionalGeneration.from_pretrained("google/mt5-xl").to(device)
        elif model_name == "llama3":
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
            model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B").to(device)
        elif model_name == "llama3.1":
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
            model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B").to(device)
        elif model_name == "aya":
            tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-101")
            model = AutoModelForCausalLM.from_pretrained("CohereForAI/aya-101").to(device)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Error loading {model_name}: {str(e)}")
        return None, None

def compute_perplexity(text, model, tokenizer, device, model_name):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.nan
    try:
        if model_name == "mt5-xl":
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
            return torch.exp(outputs.loss).item()
        else:
            # For causal language models (LLaMA, Aya)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
            return torch.exp(outputs.loss).item()
    except Exception as e:
        print(f"Error computing perplexity for {model_name}: {str(e)}")
        return np.nan

def main():
    parser = argparse.ArgumentParser(description="Analyze correlation between language, perplexity, and toxicity.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--output_csv", type=str, default=None, help="Path to save output CSV with perplexity column.")
    parser.add_argument("--save_plots", action="store_true", help="Save correlation plots if matplotlib/seaborn are available.")
    parser.add_argument("--models", nargs="+", default=["mt5-xl"], 
                      choices=["mt5-xl", "llama3", "llama3.1", "aya"],
                      help="Models to use for perplexity calculation")
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Process each model
    for model_name in args.models:
        print(f"\nProcessing with {model_name}...")
        model, tokenizer = load_model_and_tokenizer(model_name, device)
        if model is None or tokenizer is None:
            continue

        # Compute perplexity for each text type
        tqdm.pandas()
        df[f'src_perplexity_{model_name}'] = df['src'].progress_apply(
            lambda x: compute_perplexity(x, model, tokenizer, device, model_name))
        df[f'tgt_perplexity_{model_name}'] = df['tgt'].progress_apply(
            lambda x: compute_perplexity(x, model, tokenizer, device, model_name))
        df[f'generated_perplexity_{model_name}'] = df['generated'].progress_apply(
            lambda x: compute_perplexity(x, model, tokenizer, device, model_name))

        # Analyze perplexity differences
        print(f"\nPerplexity Analysis for {model_name}:")
        perplexity_cols = [f'src_perplexity_{model_name}', 
                         f'tgt_perplexity_{model_name}', 
                         f'generated_perplexity_{model_name}']
        
        print("\nMean perplexity by text type:")
        print(df[perplexity_cols].mean())
        
        print("\nMedian perplexity by text type:")
        print(df[perplexity_cols].median())

        # Statistical tests
        from scipy import stats
        
        print("\nStatistical Tests:")
        # Compare generated vs source
        t_stat, p_val = stats.ttest_rel(
            df[f'generated_perplexity_{model_name}'], 
            df[f'src_perplexity_{model_name}'])
        print(f"\nGenerated vs Source (English):")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_val:.4f}")
        
        # Compare generated vs target
        t_stat, p_val = stats.ttest_rel(
            df[f'generated_perplexity_{model_name}'], 
            df[f'tgt_perplexity_{model_name}'])
        print(f"\nGenerated vs Target (Hindi):")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_val:.4f}")

    # Save output CSV
    output_csv = args.output_csv or args.input_csv.replace('.csv', '_with_perplexity.csv')
    df.to_csv(output_csv, index=False)
    print(f"\nSaved output with perplexity to {output_csv}")

    # Correlation analysis
    print("\nComputing correlations...")
    cols_of_interest = [
        'hindi_percent', 'romanized_hindi_percent', 'total_hindi_percent', 'english_percent',
        'perspective_generated_toxicity',
        'perspective_generated_identity_attack', 'perspective_generated_insult'
    ]
    
    # Add all perplexity columns
    for model_name in args.models:
        cols_of_interest.extend([
            f'src_perplexity_{model_name}',
            f'tgt_perplexity_{model_name}',
            f'generated_perplexity_{model_name}'
        ])
    
    # Only keep columns that exist
    cols_of_interest = [c for c in cols_of_interest if c in df.columns]
    corr = df[cols_of_interest].corr(method='pearson')
    print("Correlation matrix:")
    print(corr)
    corr.to_csv(output_csv.replace('.csv', '_correlations.csv'))
    print(f"Saved correlation matrix to {output_csv.replace('.csv', '_correlations.csv')}")

    # Optional: Plot heatmap and scatter plots
    if args.save_plots and HAS_PLOTTING:
        print("\nGenerating plots...")
        # Heatmap
        plt.figure(figsize=(15, 12))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Heatmap: Language, Perplexity, Toxicity")
        plt.tight_layout()
        plt.savefig(output_csv.replace('.csv', '_correlation_heatmap.png'))
        
        # Perplexity comparison boxplots for each model
        for model_name in args.models:
            plt.figure(figsize=(10, 6))
            perplexity_cols = [
                f'src_perplexity_{model_name}',
                f'tgt_perplexity_{model_name}',
                f'generated_perplexity_{model_name}'
            ]
            perplexity_data = pd.melt(df[perplexity_cols])
            sns.boxplot(x='variable', y='value', data=perplexity_data)
            plt.title(f"Perplexity Distribution by Text Type ({model_name})")
            plt.xlabel("Text Type")
            plt.ylabel("Perplexity")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_csv.replace('.csv', f'_perplexity_comparison_{model_name}.png'))
        
        print("Plots saved.")
    elif args.save_plots:
        print("matplotlib/seaborn not available, skipping plots.")

if __name__ == "__main__":
    main() 