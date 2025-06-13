import pandas as pd
import argparse
import os
import sys
import time
import requests
from tqdm import tqdm
import numpy as np
from datetime import datetime, timedelta

# Add project root to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PERSPECTIVE_API_KEY

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
        response = requests.post(url, json=data, timeout=10)
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

def process_file(input_file, output_file, api_key, continuation_col, quota_limit=60, quota_window=60):
    """Process a single file with Perspective API."""
    print(f"\nProcessing file: {input_file}")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error loading CSV file {input_file}: {e}")
        return False

    if continuation_col not in df.columns:
        print(f"Error: Continuation column '{continuation_col}' not found in {input_file}")
        return False

    # Add score columns if they don't exist
    score_cols = {
        "toxicity": f"perspective_{continuation_col}_toxicity",
        "identity_attack": f"perspective_{continuation_col}_identity_attack",
        "insult": f"perspective_{continuation_col}_insult"
    }
    
    for col_name in score_cols.values():
        if col_name not in df.columns:
            df[col_name] = np.nan
        else:
            print(f"Scores column '{col_name}' exists, will only process NaN rows.")
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

    # Initialize quota tracking
    request_times = []
    processed_count = 0
    total_rows = len(df)
    
    print(f"Analyzing '{continuation_col}' column...")
    for idx, row in tqdm(df.iterrows(), total=total_rows):
        # Check if we need to process this row
        if all(not pd.isna(df.loc[idx, col]) for col in score_cols.values()):
            continue
            
        # Rate limiting
        current_time = datetime.now()
        request_times = [t for t in request_times if current_time - t < timedelta(seconds=quota_window)]
        if len(request_times) >= quota_limit:
            sleep_time = (request_times[0] + timedelta(seconds=quota_window) - current_time).total_seconds()
            if sleep_time > 0:
                print(f"\nRate limit reached. Sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
            request_times = request_times[1:]
        
        # Get scores
        scores, is_rate_limited = run_perspective_api(row[continuation_col], api_key)
        
        if is_rate_limited:
            print("\nRate limit hit. Saving progress and exiting...")
            break
            
        if scores:
            request_times.append(current_time)
            df.loc[idx, score_cols["toxicity"]] = scores["Toxicity"]
            df.loc[idx, score_cols["identity_attack"]] = scores["IdentityAttack"]
            df.loc[idx, score_cols["insult"]] = scores["Insult"]
            processed_count += 1
            
            # Save progress periodically
            if processed_count % 10 == 0:
                print(f"\nSaving progress ({processed_count}/{total_rows})")
                df.to_csv(output_file, index=False)
    
    # Save final results
    print(f"\nSaving final results to {output_file}")
    df.to_csv(output_file, index=False)
    print(f"Analysis complete. Processed {processed_count} comments out of {total_rows}.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run Perspective API on continuation files")
    parser.add_argument("--quota_limit", type=int, default=60, help="API requests per time window")
    parser.add_argument("--quota_window", type=int, default=60, help="Time window in seconds")
    
    args = parser.parse_args()
    
    # Check if API key is available
    if PERSPECTIVE_API_KEY is None:
        print("Error: PERSPECTIVE_API_KEY not found in config.py")
        sys.exit(1)
    
    # Define directories for full dataset
    src_dir = "new_outputs/src_results_full"
    tgt_dir = "new_outputs/tgt_results_full"
    output_base_dir = "new_outputs/perspective_continuations_full"
    
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process all files in both directories
    for directory in [src_dir, tgt_dir]:
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            continue
            
        for filename in os.listdir(directory):
            if not filename.endswith('.csv'):
                continue
                
            # Determine if it's a source or target file
            is_src = 'src' in filename
            model_name = filename.split('_')[0]  # e.g., 'aya', 'llama3', etc.
            
            # Set up input and output paths
            input_file = os.path.join(directory, filename)
            output_file = os.path.join(output_base_dir, f"{model_name}_{'src' if is_src else 'tgt'}_perspective.csv")
            
            # Set continuation column name
            continuation_col = f"{model_name}_continuation"
            
            # Process the file
            print(f"\n{'='*50}")
            print(f"Processing {model_name} {'source' if is_src else 'target'} continuations")
            print(f"{'='*50}")
            
            success = process_file(
                input_file=input_file,
                output_file=output_file,
                api_key=PERSPECTIVE_API_KEY,
                continuation_col=continuation_col,
                quota_limit=args.quota_limit,
                quota_window=args.quota_window
            )
            
            if success:
                print(f"Successfully processed {filename}")
            else:
                print(f"Failed to process {filename}")

if __name__ == "__main__":
    main() 