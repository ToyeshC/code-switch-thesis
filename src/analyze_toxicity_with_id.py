import pandas as pd
import argparse
import os
import time
import numpy as np
from tqdm import tqdm
import json
import requests
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Default API key (override with environment variable or command-line argument)
PERSPECTIVE_API_KEY = "your-api-key"

def analyze_toxicity(text, api_key):
    """
    Analyze the toxicity of text using the Perspective API.
    
    Args:
        text (str): Text to analyze
        api_key (str): Perspective API key
    
    Returns:
        dict: Dictionary containing toxicity scores
    """
    # Skip empty text
    if not text or pd.isna(text) or text.strip() == "":
        return {
            "toxicity": None,
            "severe_toxicity": None,
            "identity_attack": None,
            "insult": None,
            "profanity": None,
            "threat": None
        }
    
    # Truncate long text (Perspective API has a limit)
    MAX_CHARS = 20480
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    
    # Prepare API request
    url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={api_key}"
    data = {
        "comment": {"text": text},
        "languages": ["en"],
        "requestedAttributes": {
            "TOXICITY": {},
            "SEVERE_TOXICITY": {},
            "IDENTITY_ATTACK": {},
            "INSULT": {},
            "PROFANITY": {},
            "THREAT": {}
        }
    }
    
    # Send request with retries
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(url, data=json.dumps(data))
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            scores = {
                "toxicity": result["attributeScores"]["TOXICITY"]["summaryScore"]["value"],
                "severe_toxicity": result["attributeScores"]["SEVERE_TOXICITY"]["summaryScore"]["value"],
                "identity_attack": result["attributeScores"]["IDENTITY_ATTACK"]["summaryScore"]["value"],
                "insult": result["attributeScores"]["INSULT"]["summaryScore"]["value"],
                "profanity": result["attributeScores"]["PROFANITY"]["summaryScore"]["value"],
                "threat": result["attributeScores"]["THREAT"]["summaryScore"]["value"]
            }
            return scores
            
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 429:
                # Rate limit error - wait and retry
                wait_time = 2 ** attempt
                print(f"Rate limit exceeded. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"HTTP error: {http_err}")
                print(f"Response: {response.text}")
                return {
                    "toxicity": None,
                    "severe_toxicity": None,
                    "identity_attack": None,
                    "insult": None,
                    "profanity": None,
                    "threat": None
                }
        except Exception as e:
            print(f"Error analyzing comment: {e}")
            return {
                "toxicity": None,
                "severe_toxicity": None,
                "identity_attack": None,
                "insult": None,
                "profanity": None,
                "threat": None
            }
    
    # If all retries failed
    print("All retries failed for toxicity analysis")
    return {
        "toxicity": None,
        "severe_toxicity": None,
        "identity_attack": None,
        "insult": None,
        "profanity": None,
        "threat": None
    }

def analyze_file_toxicity(input_file, output_file, api_key, progress_file=None, batch_size=5):
    """
    Analyze toxicity of text in a CSV file, preserving primary keys.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to save toxicity analysis results
        api_key (str): Perspective API key
        progress_file (str): Optional file to save progress
        batch_size (int): Number of items to process before saving progress
    
    Returns:
        pd.DataFrame: DataFrame with toxicity scores
    """
    # Read input file
    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Ensure required columns exist
    if 'prompt_id' not in df.columns:
        print("Error: No 'prompt_id' column found in the input file.")
        return None
    
    # Identify text column
    text_column = None
    for col in ['sentence', 'prompt', 'response']:
        if col in df.columns:
            text_column = col
            break
    
    if not text_column:
        print("Error: No text column found in the input file.")
        return None
    
    # Check for existing progress
    if progress_file and os.path.exists(progress_file):
        print(f"Loading existing progress from {progress_file}")
        result_df = pd.read_csv(progress_file)
        # Determine how many items we've already processed
        processed_ids = set(result_df['prompt_id'])
        df = df[~df['prompt_id'].isin(processed_ids)]
        print(f"Resuming from {len(result_df)} previously processed items")
    else:
        # Initialize results DataFrame
        result_df = pd.DataFrame({
            'prompt_id': [],
            'text': [],
            'toxicity': [],
            'severe_toxicity': [],
            'identity_attack': [],
            'insult': [],
            'profanity': [],
            'threat': []
        })
    
    # Process the text
    print(f"Analyzing toxicity for {len(df)} items...")
    
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        text = row[text_column]
        prompt_id = row['prompt_id']
        
        # Analyze toxicity
        scores = analyze_toxicity(text, api_key)
        
        # Add to results
        new_row = pd.DataFrame({
            'prompt_id': [prompt_id],
            'text': [text],
            'toxicity': [scores['toxicity']],
            'severe_toxicity': [scores['severe_toxicity']],
            'identity_attack': [scores['identity_attack']],
            'insult': [scores['insult']],
            'profanity': [scores['profanity']],
            'threat': [scores['threat']]
        })
        
        result_df = pd.concat([result_df, new_row], ignore_index=True)
        
        # Save progress periodically
        if progress_file and (i + 1) % batch_size == 0:
            result_df.to_csv(progress_file, index=False)
            print(f"Saved progress ({i + 1}/{len(df)})")
    
    # Save final results
    print(f"Saving results to {output_file}")
    result_df.to_csv(output_file, index=False)
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description="Analyze toxicity of text in a CSV file")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to save toxicity analysis results")
    parser.add_argument("--api_key", help="Perspective API key (overrides config.py)")
    parser.add_argument("--progress_file", help="File to save progress for resuming")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of items to process before saving progress")
    
    args = parser.parse_args()
    
    # Set API key
    api_key = args.api_key or os.environ.get("PERSPECTIVE_API_KEY") or PERSPECTIVE_API_KEY
    if api_key == "your-api-key":
        print("Error: No Perspective API key provided")
        return
    
    analyze_file_toxicity(args.input, args.output, api_key, args.progress_file, args.batch_size)

if __name__ == "__main__":
    main() 