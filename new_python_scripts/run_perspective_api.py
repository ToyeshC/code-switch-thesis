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

def run_perspective_api(text, api_key, quota_limit=60, quota_window=60):
    """
    Run the Perspective API on a text and return the scores.
    
    Args:
        text (str): The text to analyze
        api_key (str): The Perspective API key
        quota_limit (int): Maximum number of requests per window
        quota_window (int): Time window in seconds for quota limit
        
    Returns:
        tuple: (scores, should_wait)
    """
    url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={api_key}"
    
    data = {
        "comment": {"text": text},
        "languages": ["en"],
        "requestedAttributes": {
            "TOXICITY": {},
            "IDENTITY_ATTACK": {},
            "INSULT": {}
        }
    }
    
    try:
        response = requests.post(url, json=data)
        
        # Check if we've hit the quota limit
        if response.status_code == 429:  # Too Many Requests
            return None, True
            
        response.raise_for_status()
        result = response.json()
        
        scores = {
            "Toxicity": result["attributeScores"]["TOXICITY"]["summaryScore"]["value"],
            "IdentityAttack": result["attributeScores"]["IDENTITY_ATTACK"]["summaryScore"]["value"],
            "Insult": result["attributeScores"]["INSULT"]["summaryScore"]["value"]
        }
        
        return scores, False
    except Exception as e:
        print(f"Error running Perspective API: {e}")
        return {
            "Toxicity": 0,
            "IdentityAttack": 0,
            "Insult": 0
        }, False

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

def main():
    parser = argparse.ArgumentParser(description="Run Perspective API on text data")
    parser.add_argument("--input_file", required=True, help="Path to the input CSV file")
    parser.add_argument("--output_file", required=True, help="Path to save the output CSV file")
    parser.add_argument("--api_key", required=True, help="Perspective API key")
    parser.add_argument("--max_rows", type=int, default=None, help="Maximum number of rows to process")
    parser.add_argument("--text_column", type=str, default=None, help="Specific column name containing text to analyze (optional)")
    parser.add_argument("--quota_limit", type=int, default=60, help="Maximum number of requests per window")
    parser.add_argument("--quota_window", type=int, default=60, help="Time window in seconds for quota limit")
    # Optional arguments for comparison - keep these separate for clarity
    parser.add_argument("--original_file", help="Path to the original data CSV file for comparison")
    parser.add_argument("--output_dir", help="Directory to save the comparison results")
    
    args = parser.parse_args()
    
    # Process the data
    process_data(
        args.input_file, 
        args.output_file, 
        args.api_key, 
        args.max_rows, 
        args.quota_limit, 
        args.quota_window,
        args.text_column # Pass the specific text column if provided
    )
    
    # Compare scores if original file and output directory are provided
    # Note: This comparison logic might need adjustment based on the new column names
    if args.original_file and args.output_dir:
        print("\nRunning toxicity comparison...")
        compare_toxicity_scores(args.original_file, args.output_file, args.output_dir)

if __name__ == "__main__":
    main() 