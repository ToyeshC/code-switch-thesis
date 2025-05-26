import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import sys

# Add project root to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HUGGINGFACE_API_KEY

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ========== CONFIGURATION ==========
CSV_PATH = 'ezswitch/data/hinge/train.csv'
LANG_COLUMNS = ['English', 'Hindi', 'Hinglish']
MODELS = {
    'llama-3.1': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama-3': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'aya': 'CohereForAI/aya-23-8B'
}
N_ROWS = 100  # Set to None to use all rows, or an int for quick testing

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
    # Load data
    df = pd.read_csv(CSV_PATH)
    if N_ROWS:
        df = df.head(N_ROWS)

    results = []

    for model_name, model_path in MODELS.items():
        print(f"\nLoading model: {model_name} ({model_path})")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=HUGGINGFACE_API_KEY)
        model = AutoModelForCausalLM.from_pretrained(model_path, use_auth_token=HUGGINGFACE_API_KEY)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()

        for lang in LANG_COLUMNS:
            print(f"  Processing language: {lang}")
            for text in tqdm(df[lang].dropna(), desc=f"{model_name}-{lang}"):
                try:
                    perplexity = compute_perplexity(str(text), model, tokenizer, device)
                except Exception as e:
                    print(f"Error with text: {text[:30]}... | {e}")
                    perplexity = None
                results.append({
                    'model': model_name,
                    'language': lang,
                    'perplexity': perplexity
                })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv('python_scripts/output/perplexity_results.csv', index=False)

    # ========== VISUALIZATION ==========
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='language', y='perplexity', hue='model', data=results_df)
    plt.title('Perplexity Comparison Across Languages and Models')
    plt.ylabel('Perplexity')
    plt.xlabel('Language')
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig('python_scripts/output/perplexity_comparison.png')
    plt.show()

if __name__ == '__main__':
    main() 