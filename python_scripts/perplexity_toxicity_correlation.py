import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from tqdm import tqdm

# Add project root to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HUGGINGFACE_API_KEY

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ========== CONFIGURATION ==========
FILES = [
    # (filepath, input_col, toxicity_col, model_name)
    ("new_outputs/perspective_full/aya_continuations_perspective_local_full.csv", "generated", "perspective_aya_continuation_toxicity", "CohereForAI/aya-23-8B"),
    ("new_outputs/perspective_full/llama3_continuations_perspective_local_full.csv", "generated", "perspective_llama3_continuation_toxicity", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("new_outputs/perspective_full/llama31_continuations_perspective_local_full.csv", "generated", "perspective_llama31_continuation_toxicity", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("new_outputs/src_results_full/aya_src_continuations_full.csv", "src", "perspective_src_toxicity", "CohereForAI/aya-23-8B"),
    ("new_outputs/src_results_full/llama3_src_continuations_full.csv", "src", "perspective_src_toxicity", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("new_outputs/src_results_full/llama31_src_continuations_full.csv", "src", "perspective_src_toxicity", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("new_outputs/tgt_results_full/aya_tgt_continuations_full.csv", "tgt", "perspective_tgt_toxicity", "CohereForAI/aya-23-8B"),
    ("new_outputs/tgt_results_full/llama3_tgt_continuations_full.csv", "tgt", "perspective_tgt_toxicity", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("new_outputs/tgt_results_full/llama31_tgt_continuations_full.csv", "tgt", "perspective_tgt_toxicity", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
]
N_ROWS = 100  # Set to None to use all rows, or an int for quick testing
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

# ========== MAIN SCRIPT ==========
def main():
    model_cache = {}
    for filepath, input_col, tox_col, model_name in FILES:
        print(f"\nProcessing: {filepath} | Model: {model_name}")
        # Load model only once per model_name
        if model_name not in model_cache:
            print(f"  Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGINGFACE_API_KEY)
            model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=HUGGINGFACE_API_KEY)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
            model.eval()
            model_cache[model_name] = (model, tokenizer, device)
        else:
            model, tokenizer, device = model_cache[model_name]

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
        # Save results
        out_csv = os.path.join(OUTPUT_DIR, os.path.basename(filepath).replace('.csv', '_perplexity.csv'))
        df.to_csv(out_csv, index=False)
        # Plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='perplexity', y=tox_col, data=df, alpha=0.5)
        plt.title(f'Perplexity vs Toxicity\n{os.path.basename(filepath)} | Model: {model_name}')
        plt.xlabel('Perplexity')
        plt.ylabel('Toxicity')
        plt.tight_layout()
        plot_path = out_csv.replace('.csv', '.png')
        plt.savefig(plot_path)
        plt.close()
        # Correlation
        corr = df[['perplexity', tox_col]].corr(method='pearson').iloc[0,1]
        print(f"  Pearson correlation: {corr:.3f}")
        # Save correlation value
        with open(out_csv.replace('.csv', '_correlation.txt'), 'w') as f:
            f.write(f"Pearson correlation between perplexity and toxicity: {corr:.3f}\n")

if __name__ == '__main__':
    main() 