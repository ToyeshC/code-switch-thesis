import pandas as pd
import argparse
import os
import json
import time
import requests
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from googleapiclient import discovery
from multiprocessing import Pool, cpu_count

# --- Global variables for worker processes ---
API_KEY = None
ATTRIBUTES_TO_CHECK = ['TOXICITY', 'SEVERE_TOXICITY', 'IDENTITY_ATTACK', 'INSULT', 'PROFANITY', 'THREAT']

def initialize_worker(key):
    """Initializes the API key for each worker process."""
    global API_KEY
    API_KEY = key

def analyze_text_chunk(data_chunk):
    """
    Analyzes a chunk of text data. Designed to be called by worker processes.
    `data_chunk` is a list of texts.
    """
    # Each worker builds its own client
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        static_discovery=False,
    )
    
    results = []
    for text in data_chunk:
        scores = {attr: None for attr in ATTRIBUTES_TO_CHECK}
        if isinstance(text, str) and text.strip():
            analyze_request = {
                'comment': {'text': text},
                'requestedAttributes': {attr: {} for attr in ATTRIBUTES_TO_CHECK},
                'languages': ['en']
            }
            try:
                response = client.comments().analyze(body=analyze_request).execute()
                for attr in ATTRIBUTES_TO_CHECK:
                    if 'summaryScore' in response['attributeScores'][attr]:
                        scores[attr] = response['attributeScores'][attr]['summaryScore']['value']
            except Exception as e:
                # Log error but continue, scores will remain None
                # print(f"API Error for worker {os.getpid()}: {e}")
                time.sleep(1) # Sleep on error to avoid spamming
        
        results.append(scores)
        # To stay under the default 100 QPS quota with 16 workers (16 workers / 100 QPS = 0.16s),
        # we sleep for a slightly longer time to be safe.
        time.sleep(0.18) 
        
    return results

def process_data(input_file, output_file, api_key, max_rows=None, quota_limit=60, quota_window=60, text_column=None):
    """
    Process a CSV file with the Perspective API and save the results.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the output CSV file
        api_key (str): The Perspective API key
        max_rows (int): Maximum number of rows to process (None for all rows)
        quota_limit (int): Maximum number of requests per window
        quota_window (int): Time window in seconds for quota limit
        text_column (str, optional): Specific column to analyze. If None, attempt to auto-detect.
    """
    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Limit the number of rows if specified
    if max_rows is not None and max_rows < len(df):
        print(f"Limiting to first {max_rows} rows out of {len(df)} total rows")
        df = df.head(max_rows)
    
    # --- Determine the text column to analyze --- 
    if text_column is None:
        # Auto-detect based on common patterns or filename clues
        if 'llama3_continuation' in df.columns:
            text_column = 'llama3_continuation'
        elif 'llama31_continuation' in df.columns:
            text_column = 'llama31_continuation'
        elif 'aya_continuation' in df.columns:
            text_column = 'aya_continuation'
        elif 'generated' in df.columns:
            text_column = 'generated'
        elif 'sentence' in df.columns:
            text_column = 'sentence'
        else:
            # Fallback: Use the first object/string column found
            object_columns = df.select_dtypes(include=['object']).columns
            if len(object_columns) > 0:
                text_column = object_columns[0]
                print(f"Warning: Could not detect standard text column. Using first object column: '{text_column}'")
            else:
                raise ValueError("Could not automatically determine the text column to analyze.")
    elif text_column not in df.columns:
        raise ValueError(f"Specified text_column '{text_column}' not found in the input file.")
    
    print(f"Analyzing text from column: '{text_column}'")
    # -------------------------------------------
    
    # Add columns for Perspective API scores if they don't exist
    # Prefix with column name to avoid conflicts if run on multiple columns
    score_prefix = f"perspective_{text_column}_"
    score_cols = {
        "toxicity": f"{score_prefix}toxicity",
        "identity_attack": f"{score_prefix}identity_attack",
        "insult": f"{score_prefix}insult"
    }
    
    for col_name in score_cols.values():
        if col_name not in df.columns:
            df[col_name] = 0.0 # Use float for scores
    
    # Initialize quota tracking
    request_times = []
    processed_count = 0
    
    # Process each row
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Running Perspective API on {text_column}"):
        # Skip if already processed (check the toxicity score for this specific column)
        if pd.notna(df.at[i, score_cols["toxicity"]]) and df.at[i, score_cols["toxicity"]] > 0:
             continue
        
        # Get the text to analyze from the correct column
        text = row[text_column]
        
        # Ensure text is a string and not empty/NaN
        if pd.isna(text) or not isinstance(text, str) or text.strip() == '':
            df.at[i, score_cols["toxicity"]] = 0.0
            df.at[i, score_cols["identity_attack"]] = 0.0
            df.at[i, score_cols["insult"]] = 0.0
            continue
        
        # Check if we need to wait for quota reset
        current_time = datetime.now()
        request_times = [t for t in request_times if current_time - t < timedelta(seconds=quota_window)]
        
        if len(request_times) >= quota_limit:
            wait_time = (request_times[0] + timedelta(seconds=quota_window) - current_time).total_seconds()
            if wait_time > 0:
                print(f"\nQuota limit reached. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                request_times = []
        
        # Run the Perspective API
        scores, should_wait = run_perspective_api(text, api_key, quota_limit, quota_window)
        
        if should_wait:
            print("\nRate limit hit. Waiting for quota reset...")
            time.sleep(quota_window)
            scores, _ = run_perspective_api(text, api_key, quota_limit, quota_window)
        
        # Update the DataFrame with scores for the specific column
        if scores:
             df.at[i, score_cols["toxicity"]] = scores["Toxicity"]
             df.at[i, score_cols["identity_attack"]] = scores["IdentityAttack"]
             df.at[i, score_cols["insult"]] = scores["Insult"]
        else:
             # Handle case where API failed even after waiting
             df.at[i, score_cols["toxicity"]] = -1.0 # Use -1 to indicate error
             df.at[i, score_cols["identity_attack"]] = -1.0
             df.at[i, score_cols["insult"]] = -1.0

        # Track the request
        request_times.append(current_time)
        processed_count += 1
        
        # Save progress every 10 rows
        if processed_count % 10 == 0:
            print(f"\nSaving progress after {processed_count} rows...")
            df.to_csv(output_file, index=False)
        
        # Sleep to avoid rate limiting
        time.sleep(1)
    
    # Save the final results
    print(f"Saving results to: {output_file}")
    df.to_csv(output_file, index=False)
    
    print(f"Processed {len(df)} rows")

def compare_toxicity_scores(original_file, code_switched_file, output_dir):
    """
    Compare toxicity scores between original and code-switched data.
    
    Args:
        original_file (str): Path to the original data CSV file
        code_switched_file (str): Path to the code-switched data CSV file
        output_dir (str): Directory to save the comparison results
    """
    print(f"Reading original file: {original_file}")
    original_df = pd.read_csv(original_file)
    
    print(f"Reading code-switched file: {code_switched_file}")
    code_switched_df = pd.read_csv(code_switched_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Compare scores
    metrics = ["Toxicity", "IdentityAttack", "Insult"]
    perspective_metrics = ["perspective_toxicity", "perspective_identity_attack", "perspective_insult"]
    
    # Create a DataFrame for the comparison
    comparison_data = []
    
    for i, metric in enumerate(metrics):
        # Original scores
        original_scores = original_df[metric].values
        
        # Code-switched scores
        code_switched_scores = code_switched_df[metric].values
        
        # Perspective API scores for code-switched data
        perspective_scores = code_switched_df[perspective_metrics[i]].values
        
        # Calculate statistics
        original_mean = np.mean(original_scores)
        original_std = np.std(original_scores)
        
        code_switched_mean = np.mean(code_switched_scores)
        code_switched_std = np.std(code_switched_scores)
        
        perspective_mean = np.mean(perspective_scores)
        perspective_std = np.std(perspective_scores)
        
        # Add to comparison data
        comparison_data.append({
            "Metric": metric,
            "Original_Mean": original_mean,
            "Original_Std": original_std,
            "Code_Switched_Mean": code_switched_mean,
            "Code_Switched_Std": code_switched_std,
            "Perspective_Mean": perspective_mean,
            "Perspective_Std": perspective_std
        })
    
    # Create a DataFrame for the comparison
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save the comparison
    comparison_file = os.path.join(output_dir, "toxicity_comparison.csv")
    comparison_df.to_csv(comparison_file, index=False)
    
    print(f"Saved comparison to: {comparison_file}")
    
    # Create visualizations
    for i, metric in enumerate(metrics):
        # Create a figure
        plt.figure(figsize=(10, 6))
        
        # Create a box plot
        data = [
            original_df[metric].values,
            code_switched_df[metric].values,
            code_switched_df[perspective_metrics[i]].values
        ]
        
        plt.boxplot(data, labels=["Original", "Code-Switched", "Perspective API"])
        plt.title(f"{metric} Comparison")
        plt.ylabel("Score")
        
        # Save the figure
        fig_file = os.path.join(output_dir, f"{metric.lower()}_comparison.png")
        plt.savefig(fig_file)
        plt.close()
        
        print(f"Saved {metric} comparison plot to: {fig_file}")
    
    # Create a correlation heatmap
    plt.figure(figsize=(10, 8))
    
    # Select columns for the heatmap
    heatmap_data = pd.DataFrame({
        "Original_Toxicity": original_df["Toxicity"],
        "Original_IdentityAttack": original_df["IdentityAttack"],
        "Original_Insult": original_df["Insult"],
        "Code_Switched_Toxicity": code_switched_df["Toxicity"],
        "Code_Switched_IdentityAttack": code_switched_df["IdentityAttack"],
        "Code_Switched_Insult": code_switched_df["Insult"],
        "Perspective_Toxicity": code_switched_df["perspective_toxicity"],
        "Perspective_IdentityAttack": code_switched_df["perspective_identity_attack"],
        "Perspective_Insult": code_switched_df["perspective_insult"]
    })
    
    # Calculate correlation
    correlation = heatmap_data.corr()
    
    # Create heatmap
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    
    # Save the figure
    heatmap_file = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_file)
    plt.close()
    
    print(f"Saved correlation heatmap to: {heatmap_file}")

def main(args):
    """Main function to run Perspective API on all relevant columns in parallel."""
    print(f"Reading input file: {args.input_file}")
    df = pd.read_csv(args.input_file)

    # Automatically find columns to process, unless specified
    if args.columns:
        columns_to_analyze = args.columns.split(',')
        print(f"Using specified columns for analysis: {columns_to_analyze}")
    else:
        columns_to_analyze = ['src', 'tgt', 'generated']
        continuation_columns = [col for col in df.columns if col.endswith('_continuation')]
        columns_to_analyze.extend(continuation_columns)
        print(f"Automatically found {len(columns_to_analyze)} columns to analyze.")
    
    num_workers = args.num_workers if args.num_workers > 0 else cpu_count()
    print(f"Using {num_workers} worker processes.")

    for column_name in columns_to_analyze:
        if column_name not in df.columns:
            print(f"Warning: Column '{column_name}' not found in input file. Skipping analysis.")
            continue
        
        print(f"\nAnalyzing column: {column_name}")
        
        texts_to_process = df[column_name].tolist()
        
        # Split data into chunks for each worker
        chunk_size = len(texts_to_process) // num_workers + (len(texts_to_process) % num_workers > 0)
        chunks = [texts_to_process[i:i + chunk_size] for i in range(0, len(texts_to_process), chunk_size)]

        all_scores = []
        # Use a multiprocessing Pool to process chunks in parallel
        with Pool(processes=num_workers, initializer=initialize_worker, initargs=(args.api_key,)) as pool:
            # Use tqdm to show progress for the parallel processing
            for scores_chunk in tqdm(pool.imap(analyze_text_chunk, chunks), total=len(chunks), desc=f"Processing {column_name}"):
                all_scores.extend(scores_chunk)
        
        # Add new columns to the DataFrame
        for attr in ATTRIBUTES_TO_CHECK:
            new_col_name = f"{column_name}_{attr.lower()}"
            df[new_col_name] = [s[attr] for s in all_scores]

    print(f"\nSaving final analysis results to: {args.output_file}")
    df.to_csv(args.output_file, index=False)
    print("Script finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Perspective API in parallel on multiple text columns in a CSV file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file (e.g., continuations.csv).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the final output CSV with all perspective scores.")
    parser.add_argument("--api_key", type=str, required=True, help="Your Perspective API key.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of parallel worker processes. Defaults to number of CPU cores.")
    parser.add_argument("--columns", type=str, default=None, help="Comma-separated list of column names to analyze. If not provided, all relevant columns are used.")
    
    args = parser.parse_args()
    main(args) 