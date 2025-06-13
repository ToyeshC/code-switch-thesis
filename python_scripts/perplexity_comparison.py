import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import sys
import gc

# Add project root to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HUGGINGFACE_API_KEY

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ========== CONFIGURATION ==========
FILES = [
    # Original files
    ("new_outputs/perspective_full/aya_continuations_perspective_local_full.csv", "generated", "perspective_aya_continuation_toxicity", "CohereForAI/aya-23-8B"),
    ("new_outputs/perspective_full/llama3_continuations_perspective_local_full.csv", "generated", "perspective_llama3_continuation_toxicity", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("new_outputs/perspective_full/llama31_continuations_perspective_local_full.csv", "generated", "perspective_llama31_continuation_toxicity", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("new_outputs/src_results_full/aya_src_continuations_full.csv", "src", "perspective_src_toxicity", "CohereForAI/aya-23-8B"),
    ("new_outputs/src_results_full/llama3_src_continuations_full.csv", "src", "perspective_src_toxicity", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("new_outputs/src_results_full/llama31_src_continuations_full.csv", "src", "perspective_src_toxicity", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("new_outputs/tgt_results_full/aya_tgt_continuations_full.csv", "tgt", "perspective_tgt_toxicity", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("new_outputs/tgt_results_full/llama3_tgt_continuations_full.csv", "tgt", "perspective_tgt_toxicity", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("new_outputs/tgt_results_full/llama31_tgt_continuations_full.csv", "tgt", "perspective_tgt_toxicity", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    
    # New perspective_continuations files
    ("new_outputs/perspective_continuations/aya_src_perspective.csv", "src", "perspective_src_toxicity", "CohereForAI/aya-23-8B"),
    ("new_outputs/perspective_continuations/llama3_src_perspective.csv", "src", "perspective_src_toxicity", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("new_outputs/perspective_continuations/llama31_src_perspective.csv", "src", "perspective_src_toxicity", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("new_outputs/perspective_continuations/aya_tgt_perspective.csv", "tgt", "perspective_tgt_toxicity", "CohereForAI/aya-23-8B"),
    ("new_outputs/perspective_continuations/llama3_tgt_perspective.csv", "tgt", "perspective_tgt_toxicity", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("new_outputs/perspective_continuations/llama31_tgt_perspective.csv", "tgt", "perspective_tgt_toxicity", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
]
LANG_COLUMNS = ['English', 'Hindi', 'Hinglish']
MODELS = {
    'llama-3.1': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama-3': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'aya': 'CohereForAI/aya-23-8B'
}
N_ROWS = None  # Set to None to use all rows

# Set HuggingFace API key for transformers
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_HUB_TOKEN"] = HUGGINGFACE_API_KEY

# ========== HELPER FUNCTIONS ==========
def compute_perplexity(text, model, tokenizer, device='cpu'):
    """
    Compute perplexity of a single text using a causal LM.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return torch.exp(loss).item()

# ========== MAIN SCRIPT ==========
def main():
    results = []

    for model_name, model_path in MODELS.items():
        print(f"\nLoading model: {model_name} ({model_path})")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=HUGGINGFACE_API_KEY)
        model = AutoModelForCausalLM.from_pretrained(model_path, use_auth_token=HUGGINGFACE_API_KEY)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()

        # Process each file for this model
        for filepath, input_col, tox_col, _ in [f for f in FILES if f[3] == model_path]:
            print(f"\nProcessing file: {filepath}")
            df = pd.read_csv(filepath)
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

            # Add results
            results.append({
                'file': filepath,
                'model': model_name,
                'input_col': input_col,
                'toxicity_col': tox_col,
                'perplexities': perplexities
            })

            # Save individual results
            out_df = pd.DataFrame({
                input_col: df[input_col],
                tox_col: df[tox_col],
                'perplexity': perplexities
            })
            out_csv = os.path.join('python_scripts/output/perplexity_toxicity', 
                                 os.path.basename(filepath).replace('.csv', '_perplexity.csv'))
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            out_df.to_csv(out_csv, index=False)

        # Clear model from memory
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Save combined results
    combined_results = []
    for result in results:
        df = pd.DataFrame({
            'file': [result['file']] * len(result['perplexities']),
            'model': [result['model']] * len(result['perplexities']),
            'input_col': [result['input_col']] * len(result['perplexities']),
            'toxicity_col': [result['toxicity_col']] * len(result['perplexities']),
            'perplexity': result['perplexities']
        })
        combined_results.append(df)

    if combined_results:
        final_df = pd.concat(combined_results, ignore_index=True)
        final_df.to_csv('python_scripts/output/perplexity_results.csv', index=False)

    # ========== VISUALIZATION ==========
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='toxicity_col', y='perplexity', hue='model', data=final_df)
    plt.title('Perplexity Comparison Across Toxicity Columns and Models')
    plt.ylabel('Perplexity')
    plt.xlabel('Toxicity Column')
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig('python_scripts/output/perplexity_comparison.png')
    plt.show()

if __name__ == '__main__':
    main() 