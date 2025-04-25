import pandas as pd
import pickle
import argparse
import os
import sys
import time
import requests
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# --- Perspective API Function (copied from run_perspective_api.py) --- 
def run_perspective_api(text, api_key, quota_limit=60, quota_window=60):
    if not isinstance(text, str) or text.strip() == '':
        # Return default scores for invalid input
        return {"Toxicity": 0.0, "IdentityAttack": 0.0, "Insult": 0.0}, False
        
    url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={api_key}"
    data = {
        "comment": {"text": text},
        "languages": ["en"], # Assuming continuations are primarily English-based for scoring
        "requestedAttributes": {
            "TOXICITY": {},
            "IDENTITY_ATTACK": {},
            "INSULT": {}
        }
    }
    
    try:
        response = requests.post(url, json=data, timeout=10) # Added timeout
        if response.status_code == 429: return None, True # Rate limit
        response.raise_for_status()
        result = response.json()
        scores = {
            "Toxicity": result.get("attributeScores", {}).get("TOXICITY", {}).get("summaryScore", {}).get("value", 0.0),
            "IdentityAttack": result.get("attributeScores", {}).get("IDENTITY_ATTACK", {}).get("summaryScore", {}).get("value", 0.0),
            "Insult": result.get("attributeScores", {}).get("INSULT", {}).get("summaryScore", {}).get("value", 0.0)
        }
        return scores, False
    except requests.exceptions.RequestException as e:
        print(f"\nAPI Request Error: {e}")
        return {"Toxicity": -1.0, "IdentityAttack": -1.0, "Insult": -1.0}, False # Indicate error
    except Exception as e:
        print(f"\nError running Perspective API: {e}")
        return {"Toxicity": -1.0, "IdentityAttack": -1.0, "Insult": -1.0}, False # Indicate error

# --- Function to process a single file (En or Hi PKL) --- 
def process_pkl_file(filepath, api_key, max_rows=None, quota_limit=60, quota_window=60):
    print(f"Processing PKL file: {filepath}")
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading PKL file {filepath}: {e}")
        return None
        
    if not isinstance(data, list):
        print(f"Error: Expected a list in PKL file {filepath}, got {type(data)}")
        return None
        
    results = []
    request_times = []
    processed_count = 0
    
    # Limit rows if needed
    num_to_process = len(data)
    if max_rows is not None and max_rows < num_to_process:
        num_to_process = max_rows
        print(f"Limiting to {max_rows} rows.")

    for idx in tqdm(range(num_to_process), desc=f"Analyzing {os.path.basename(filepath)}"):
        inner_list = data[idx]
        continuation = "<NO_CONTINUATION>" # Default if no valid text found
        
        # Extract the first valid continuation text
        if isinstance(inner_list, list) and len(inner_list) > 0:
            first_item = inner_list[0]
            if isinstance(first_item, dict) and 'generated_text' in first_item:
                 continuation = first_item['generated_text']
                 if not isinstance(continuation, str):
                     continuation = str(continuation) # Ensure string type
            elif isinstance(first_item, str):
                 continuation = first_item # Handle if element is just a string
        
        # Handle API Quota
        current_time = datetime.now()
        request_times = [t for t in request_times if current_time - t < timedelta(seconds=quota_window)]
        while len(request_times) >= quota_limit:
            wait_time = (request_times[0] + timedelta(seconds=quota_window) - current_time).total_seconds()
            if wait_time > 0:
                print(f"\nQuota limit hit. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            current_time = datetime.now() # Re-check time after waiting
            request_times = [t for t in request_times if current_time - t < timedelta(seconds=quota_window)]
        
        # Run Perspective API
        scores, should_wait = run_perspective_api(continuation, api_key, quota_limit, quota_window)
        if should_wait:
             print("\nRate limit hit after check. Waiting...")
             time.sleep(quota_window + 2) # Wait a bit longer
             scores, _ = run_perspective_api(continuation, api_key, quota_limit, quota_window)
             
        if scores is None: # Handle API failure even after wait
             print(f"\nAPI failed for row {idx} even after waiting. Assigning error scores.")
             scores = {"Toxicity": -1.0, "IdentityAttack": -1.0, "Insult": -1.0}
             
        results.append({
            "index": idx, # Use index as temporary key
            "continuation": continuation,
            "perspective_toxicity": scores["Toxicity"],
            "perspective_identity_attack": scores["IdentityAttack"],
            "perspective_insult": scores["Insult"]
        })
        
        request_times.append(datetime.now())
        processed_count += 1
        time.sleep(1) # Basic rate limiting

    print(f"Processed {processed_count} entries from {filepath}")
    return pd.DataFrame(results).set_index("index")

# --- Main Comparison Logic --- 
def main():
    parser = argparse.ArgumentParser(description="Compare Perspective Scores across En, Hi, and CS continuations.")
    parser.add_argument("--en_pkl", required=True, help="Path to English continuations PKL file")
    parser.add_argument("--hi_pkl", required=True, help="Path to Hindi continuations PKL file")
    parser.add_argument("--cs_csv", required=True, help="Path to Code-Switched continuations CSV file (with Perspective scores)")
    parser.add_argument("--api_key", required=True, help="Perspective API key")
    parser.add_argument("--output_dir", required=True, help="Directory to save comparison results")
    parser.add_argument("--max_rows", type=int, default=None, help="Maximum number of rows to process from each file")
    parser.add_argument("--model_name", required=True, help="Short name of the model being compared (e.g., llama3, aya)")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process English and Hindi PKL files to get Perspective scores
    en_df = process_pkl_file(args.en_pkl, args.api_key, args.max_rows)
    hi_df = process_pkl_file(args.hi_pkl, args.api_key, args.max_rows)
    
    if en_df is None or hi_df is None:
        print("Error processing PKL files. Exiting.")
        sys.exit(1)
        
    # Load Code-Switched CSV data
    print(f"Loading Code-Switched data from: {args.cs_csv}")
    try:
        cs_df_full = pd.read_csv(args.cs_csv)
    except Exception as e:
        print(f"Error loading CS CSV file {args.cs_csv}: {e}")
        sys.exit(1)
        
    # Limit CS rows if needed
    if args.max_rows is not None and args.max_rows < len(cs_df_full):
        cs_df = cs_df_full.head(args.max_rows).copy()
    else:
        cs_df = cs_df_full.copy()
    
    # Identify CS perspective score columns (assuming pattern from run_perspective_api.py)
    cs_continuation_col = f"{args.model_name}_continuation"
    cs_tox_col = f"perspective_{cs_continuation_col}_toxicity"
    cs_id_col = f"perspective_{cs_continuation_col}_identity_attack"
    cs_ins_col = f"perspective_{cs_continuation_col}_insult"
    
    # Check if expected columns exist in CS data
    required_cs_cols = [cs_tox_col, cs_id_col, cs_ins_col]
    if not all(col in cs_df.columns for col in required_cs_cols):
        print(f"Error: Required Perspective score columns not found in {args.cs_csv}.")
        print(f"Expected columns like: {required_cs_cols}")
        print(f"Available columns: {list(cs_df.columns)}")
        sys.exit(1)
        
    # Add a matching index to CS data (assuming order matches PKL files)
    cs_df = cs_df.reset_index().rename(columns={'index': 'original_index'})
    cs_df['index'] = cs_df.index
    cs_df = cs_df.set_index('index')
    
    # --- Combine Data --- 
    # Rename columns for clarity before merging
    en_df = en_df.rename(columns={
        'perspective_toxicity': 'toxicity_en',
        'perspective_identity_attack': 'identity_attack_en',
        'perspective_insult': 'insult_en'
    })
    hi_df = hi_df.rename(columns={
        'perspective_toxicity': 'toxicity_hi',
        'perspective_identity_attack': 'identity_attack_hi',
        'perspective_insult': 'insult_hi'
    })
    cs_df_scores = cs_df[[cs_tox_col, cs_id_col, cs_ins_col]].rename(columns={
        cs_tox_col: 'toxicity_cs',
        cs_id_col: 'identity_attack_cs',
        cs_ins_col: 'insult_cs'
    })
    
    # Merge based on the index (assuming order corresponds)
    combined_df = pd.concat([en_df[[col for col in en_df.columns if col.endswith('_en')]], 
                               hi_df[[col for col in hi_df.columns if col.endswith('_hi')]], 
                               cs_df_scores], axis=1)
    
    # --- Analysis & Output --- 
    print("\nCalculating summary statistics...")
    summary = combined_df.agg(['mean', 'std', 'median'])
    print(summary)
    
    summary_file = os.path.join(args.output_dir, f"{args.model_name}_score_summary.csv")
    summary.to_csv(summary_file)
    print(f"Saved summary statistics to {summary_file}")
    
    # --- Plotting --- 
    print("\nGenerating plots...")
    metrics = [("Toxicity", "toxicity"), ("Identity Attack", "identity_attack"), ("Insult", "insult")]
    conditions = ["en", "hi", "cs"]
    condition_labels = {"en": "English", "hi": "Hindi", "cs": "Code-Switched"}
    
    for metric_label, metric_base in metrics:
        plt.figure(figsize=(8, 6))
        plot_data = {
            condition_labels[cond]: combined_df[f"{metric_base}_{cond}"].dropna() 
            for cond in conditions
        }
        plt.boxplot(plot_data.values(), labels=plot_data.keys())
        plt.title(f"{args.model_name.upper()} - {metric_label} Score Comparison")
        plt.ylabel("Perspective Score")
        plt.ylim(0, 1) # Scores are between 0 and 1
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plot_file = os.path.join(args.output_dir, f"{args.model_name}_{metric_base}_boxplot.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"Saved boxplot to {plot_file}")

    print("\nComparison complete.")

if __name__ == "__main__":
    main() 