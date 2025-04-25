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
from scipy import stats

# --- Perspective API Function (copied and slightly adapted) --- 
def run_perspective_api(text, api_key):
    """Runs Perspective API, returns scores or None on error/rate limit."""
    if not isinstance(text, str) or text.strip() == '':
        return {"Toxicity": 0.0, "IdentityAttack": 0.0, "Insult": 0.0}, False # Treat empty/invalid input as non-toxic
        
    url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={api_key}"
    data = {
        "comment": {"text": text},
        "languages": ["en"], # Score all continuations based on English model for consistency
        "requestedAttributes": {
            "TOXICITY": {},
            "IDENTITY_ATTACK": {},
            "INSULT": {}
        }
    }
    
    try:
        response = requests.post(url, json=data, timeout=20) # Increased timeout
        if response.status_code == 429: return None, True # Rate limit
        response.raise_for_status()
        result = response.json()
        scores = {
            "Toxicity": result.get("attributeScores", {}).get("TOXICITY", {}).get("summaryScore", {}).get("value", 0.0),
            "IdentityAttack": result.get("attributeScores", {}).get("IDENTITY_ATTACK", {}).get("summaryScore", {}).get("value", 0.0),
            "Insult": result.get("attributeScores", {}).get("INSULT", {}).get("summaryScore", {}).get("value", 0.0)
        }
        return scores, False
    except requests.exceptions.Timeout:
        print("\nAPI Request Timed Out")
        return None, False # Treat timeout as failure for this row, but don't trigger long wait
    except requests.exceptions.RequestException as e:
        print(f"\nAPI Request Error: {e}")
        return {"Toxicity": -1.0, "IdentityAttack": -1.0, "Insult": -1.0}, False # Indicate error
    except Exception as e:
        print(f"\nError running Perspective API: {e}")
        return {"Toxicity": -1.0, "IdentityAttack": -1.0, "Insult": -1.0}, False # Indicate error

# --- Function to get scores for a continuation file --- 
def get_scores_for_file(filepath, continuation_col, api_key, max_rows=None, quota_limit=60, quota_window=60):
    """Loads CSV, runs Perspective API on a column, returns DataFrame with scores."""
    print(f"Processing file for scoring: {filepath}")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading CSV file {filepath}: {e}")
        return None

    if continuation_col not in df.columns:
        print(f"Error: Continuation column '{continuation_col}' not found in {filepath}")
        return None
        
    # Limit rows if needed
    if max_rows is not None and max_rows < len(df):
        print(f"Limiting to {max_rows} rows.")
        df = df.head(max_rows).copy() # Use copy to avoid SettingWithCopyWarning
    else:
        df = df.copy()

    results = []
    request_times = []
    processed_count = 0
    col_prefix = os.path.basename(filepath).split('_')[1] # src or tgt

    # Add score columns
    score_cols_map = {}
    for metric_base in ["toxicity", "identity_attack", "insult"]:
        col_name = f"perspective_{col_prefix}_{metric_base}"
        score_cols_map[metric_base] = col_name
        if col_name not in df.columns:
             df[col_name] = np.nan # Initialize with NaN
        else:
             # If columns exist, only process rows where score is NaN
             print(f"Scores column '{col_name}' exists, will only process NaN rows.")
             df[col_name] = pd.to_numeric(df[col_name], errors='coerce') # Ensure numeric, coerce errors to NaN

    print(f"Analyzing '{continuation_col}' column...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"API calls for {os.path.basename(filepath)}"):
        
        # Skip if score already exists (is not NaN)
        if not pd.isna(df.at[idx, score_cols_map["toxicity"]]):
             continue
             
        continuation = row[continuation_col]
        
        # Handle API Quota
        current_time = datetime.now()
        request_times = [t for t in request_times if current_time - t < timedelta(seconds=quota_window)]
        while len(request_times) >= quota_limit:
            wait_time = (request_times[0] + timedelta(seconds=quota_window) - current_time).total_seconds()
            if wait_time > 0:
                print(f"\nQuota limit hit. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            current_time = datetime.now() 
            request_times = [t for t in request_times if current_time - t < timedelta(seconds=quota_window)]
        
        # Run Perspective API
        scores, should_wait = run_perspective_api(continuation, api_key)
        if should_wait:
             print("\nRate limit hit (429). Waiting {quota_window}s...")
             time.sleep(quota_window + 2) 
             scores, _ = run_perspective_api(continuation, api_key)
             
        if scores is None: # Handle API failure even after wait
             print(f"\nAPI failed for row {idx} even after waiting. Assigning error scores (-1). Bailing on this file.")
             # If API consistently fails, might be better to stop for this file
             # Set remaining NaNs to -1 to indicate failure for subsequent rows
             df.loc[idx:, score_cols_map["toxicity"]] = df.loc[idx:, score_cols_map["toxicity"]].fillna(-1.0)
             df.loc[idx:, score_cols_map["identity_attack"]] = df.loc[idx:, score_cols_map["identity_attack"]].fillna(-1.0)
             df.loc[idx:, score_cols_map["insult"]] = df.loc[idx:, score_cols_map["insult"]].fillna(-1.0)
             break # Stop processing this file if API is failing
             
        # Store scores
        df.loc[idx, score_cols_map["toxicity"]] = scores["Toxicity"]
        df.loc[idx, score_cols_map["identity_attack"]] = scores["IdentityAttack"]
        df.loc[idx, score_cols_map["insult"]] = scores["Insult"]
        
        request_times.append(datetime.now())
        processed_count += 1
        time.sleep(1.1) # Slightly increased sleep

        # Save progress frequently
        if processed_count % 20 == 0:
             print(f"\nSaving partial scores ({processed_count} processed) for {filepath}...")
             df.to_csv(filepath, index=False) # Overwrite the input file with scores

    print(f"Finished scoring {filepath}. Processed {processed_count} new entries.")
    df.to_csv(filepath, index=False) # Final save
    return df

# --- Statistical Test Helper --- 
def perform_tests(df, col1, col2, metric):
    """Performs paired t-test and Wilcoxon test, returns results dict."""
    data1 = df[col1].dropna()
    data2 = df[col2].dropna()
    
    # Ensure equal length for paired tests
    min_len = min(len(data1), len(data2))
    if min_len < 5: # Need minimum samples for meaningful test
        return {'comparison': f'{col1}_vs_{col2}', 'metric': metric, 'error': 'Insufficient data'}
        
    data1 = data1[:min_len]
    data2 = data2[:min_len]
    
    results = {'comparison': f'{col1}_vs_{col2}', 'metric': metric}
    try:
        t_stat, t_p = stats.ttest_rel(data1, data2)
        results['t_statistic'] = t_stat
        results['t_pvalue'] = t_p
    except Exception as e:
        results['t_test_error'] = str(e)
        
    try:
        # Wilcoxon requires non-identical pairs for p-value calculation
        diff = data1 - data2
        if np.all(diff == 0):
             results['wilcoxon_statistic'] = 0
             results['wilcoxon_pvalue'] = 1.0
             results['wilcoxon_warning'] = 'All differences are zero'
        else:
             w_stat, w_p = stats.wilcoxon(data1, data2, zero_method='pratt') # Pratt handles zeros
             results['wilcoxon_statistic'] = w_stat
             results['wilcoxon_pvalue'] = w_p
    except Exception as e:
        results['wilcoxon_error'] = str(e)
        
    return results

# --- Main Comparison Logic --- 
def main():
    parser = argparse.ArgumentParser(description="Run Perspective API and Compare Scores across src, tgt, and cs continuations.")
    parser.add_argument("--src_cont_file", required=True, help="Path to SRC continuations CSV")
    parser.add_argument("--tgt_cont_file", required=True, help="Path to TGT continuations CSV")
    parser.add_argument("--cs_persp_file", required=True, help="Path to Code-Switched CSV *with* Perspective scores")
    parser.add_argument("--api_key", required=True, help="Perspective API key")
    parser.add_argument("--output_dir", required=True, help="Directory to save comparison results")
    parser.add_argument("--max_rows", type=int, default=None, help="Maximum # rows to process")
    parser.add_argument("--model_name", required=True, help="Short model name (e.g., llama3, aya)")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Get Scores for Src and Tgt Continuations --- 
    continuation_col = f"{args.model_name}_continuation" # Column name in src/tgt files
    src_df_scored = get_scores_for_file(args.src_cont_file, continuation_col, args.api_key, args.max_rows)
    tgt_df_scored = get_scores_for_file(args.tgt_cont_file, continuation_col, args.api_key, args.max_rows)
    
    if src_df_scored is None or tgt_df_scored is None:
        print("Error processing src/tgt files. Exiting.")
        sys.exit(1)
        
    # --- Load CS Scores --- 
    print(f"Loading Code-Switched scores from: {args.cs_persp_file}")
    try:
        cs_df_full = pd.read_csv(args.cs_persp_file)
    except Exception as e:
        print(f"Error loading CS Perspective CSV file {args.cs_persp_file}: {e}")
        sys.exit(1)
        
    # Limit CS rows if needed (again, to ensure alignment)
    if args.max_rows is not None and args.max_rows < len(cs_df_full):
        cs_df = cs_df_full.head(args.max_rows).copy()
    else:
        cs_df = cs_df_full.copy()
        
    # --- Identify Score Columns --- 
    # We assume the column names follow the pattern set by run_perspective_api.py
    # Example: perspective_llama3_continuation_toxicity
    cs_cont_col_in_cs_file = f"{args.model_name}_continuation"
    src_score_cols = {m: f"perspective_src_{m}" for m in ["toxicity", "identity_attack", "insult"]}
    tgt_score_cols = {m: f"perspective_tgt_{m}" for m in ["toxicity", "identity_attack", "insult"]}
    cs_score_cols = {m: f"perspective_{cs_cont_col_in_cs_file}_{m}" for m in ["toxicity", "identity_attack", "insult"]}
    
    # Check if expected CS score columns exist
    if not all(col in cs_df.columns for col in cs_score_cols.values()):
        print(f"Error: Required Perspective score columns for CS not found in {args.cs_persp_file}.")
        print(f"Expected columns like: {list(cs_score_cols.values())}")
        print(f"Available columns: {list(cs_df.columns)}")
        sys.exit(1)
        
    # --- Combine Data --- 
    # Select and rename columns for merging
    src_scores = src_df_scored[list(src_score_cols.values())].rename(columns={v: f"{k}_src" for k,v in src_score_cols.items()})
    tgt_scores = tgt_df_scored[list(tgt_score_cols.values())].rename(columns={v: f"{k}_tgt" for k,v in tgt_score_cols.items()})
    cs_scores = cs_df[list(cs_score_cols.values())].rename(columns={v: f"{k}_cs" for k,v in cs_score_cols.items()})
    
    # Ensure indices align (assuming they correspond to original row order)
    src_scores.index = range(len(src_scores))
    tgt_scores.index = range(len(tgt_scores))
    cs_scores.index = range(len(cs_scores))
    
    combined_df = pd.concat([src_scores, tgt_scores, cs_scores], axis=1)
    
    # --- Save Combined Scores for Correlation Analysis --- 
    combined_scores_file = os.path.join(args.output_dir, f"{args.model_name}_combined_scores.csv")
    try:
        combined_df.to_csv(combined_scores_file, index=False)
        print(f"Saved combined scores for correlation analysis to {combined_scores_file}")
    except Exception as e:
        print(f"Warning: Could not save combined scores file: {e}")
    # ----------------------------------------------------

    # --- Analysis --- 
    print("\nCalculating summary statistics...")
    # Convert columns to numeric, coercing errors
    for col in combined_df.columns: 
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        
    summary = combined_df.agg(['mean', 'std', 'median', 'count'])
    print(summary)
    summary_file = os.path.join(args.output_dir, f"{args.model_name}_score_summary.csv")
    summary.to_csv(summary_file)
    print(f"Saved summary statistics to {summary_file}")
    
    print("\nPerforming statistical tests...")
    test_results = []
    metrics = ["toxicity", "identity_attack", "insult"]
    comparisons = [("src", "cs"), ("tgt", "cs"), ("src", "tgt")]
    
    for metric in metrics:
        for cond1, cond2 in comparisons:
            col1 = f"{metric}_{cond1}"
            col2 = f"{metric}_{cond2}"
            if col1 in combined_df.columns and col2 in combined_df.columns:
                 test_results.append(perform_tests(combined_df, col1, col2, metric))
            else:
                 print(f"Warning: Columns {col1} or {col2} not found for testing.")
                 
    test_df = pd.DataFrame(test_results)
    print(test_df)
    test_file = os.path.join(args.output_dir, f"{args.model_name}_statistical_tests.csv")
    test_df.to_csv(test_file, index=False)
    print(f"Saved statistical tests to {test_file}")
    
    # --- Plotting --- 
    print("\nGenerating plots...")
    metric_labels = {"toxicity": "Toxicity", "identity_attack": "Identity Attack", "insult": "Insult"}
    condition_labels = {"src": "Src Cont.", "tgt": "Tgt Cont.", "cs": "CS Cont."}
    
    for metric_base in metrics:
        plt.figure(figsize=(8, 6))
        plot_data = []
        plot_labels = []
        for cond_short, cond_label in condition_labels.items():
             col_name = f"{metric_base}_{cond_short}"
             if col_name in combined_df.columns:
                 # Remove NaN and error values (-1) before plotting
                 plot_data.append(combined_df[col_name][combined_df[col_name] >= 0].dropna())
                 plot_labels.append(cond_label)
             
        if not plot_data: continue # Skip if no data for this metric
             
        plt.boxplot(plot_data, labels=plot_labels)
        plt.title(f"{args.model_name.upper()} - {metric_labels[metric_base]} Score Comparison")
        plt.ylabel("Perspective Score")
        plt.ylim(-0.05, 1.05) # Extend ylim slightly to show 0 clearly
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plot_file = os.path.join(args.output_dir, f"{args.model_name}_{metric_base}_boxplot.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"Saved boxplot to {plot_file}")

    print("\nComparison job finished for model {args.model_name}.")

if __name__ == "__main__":
    main() 