import pandas as pd
import requests
import argparse
import os
import sys
import time
from tqdm import tqdm

# Add the directory containing config.py to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

def analyze_comments_with_rate_limit(input_file, output_file, api_key, rate_limit=1.0, batch_size=10, continue_from=0):
    """
    Analyze comments using the Perspective API with rate limiting to avoid quota errors.
    
    Args:
        input_file (str): Path to the input CSV file containing comments
        output_file (str): Path to save the analysis results
        api_key (str): Perspective API key
        rate_limit (float): Time in seconds to wait between each API request (1.0 = 60 requests per minute)
        batch_size (int): Number of comments to process in one batch before saving
        continue_from (int): Index to continue from if process was interrupted
    """
    # Read the generated sentences from the CSV file
    print(f"Reading input file: {input_file}")
    
    # Try to read the file with different configurations
    try:
        # First try with header
        df = pd.read_csv(input_file)
        
        # Check if there's a column that might contain the comments
        text_columns = [col for col in df.columns if df[col].dtype == 'object']
        if text_columns:
            comment_column = text_columns[0]
            print(f"Using column '{comment_column}' for analysis")
        else:
            raise ValueError("No suitable text column found")
            
    except Exception as e:
        print(f"Trying to read file without header: {e}")
        # If that fails, try without header
        df = pd.read_csv(input_file, header=None, names=['comment'])
        comment_column = 'comment'
    
    comments = df[comment_column].tolist()
    total_comments = len(comments)
    print(f"Found {total_comments} comments to analyze")
    
    # Check if output file already exists and has content
    existing_results = []
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0 and continue_from > 0:
        try:
            existing_df = pd.read_csv(output_file)
            existing_results = existing_df.to_dict('records')
            print(f"Loaded {len(existing_results)} existing results from {output_file}")
        except Exception as e:
            print(f"Could not load existing results: {e}")
            existing_results = []
    
    # Define the Perspective API endpoint
    url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
    
    # Analyze each comment with rate limiting
    results = existing_results.copy()
    batch_results = []
    
    # Use tqdm for progress tracking
    for i, comment in tqdm(enumerate(comments[continue_from:], start=continue_from), total=total_comments-continue_from):
        if pd.isna(comment) or not isinstance(comment, str):
            print(f"Skipping invalid comment at index {i}")
            continue
            
        data = {
            'comment': {'text': comment},
            'requestedAttributes': {
                'TOXICITY': {},
                'SEVERE_TOXICITY': {},
                'IDENTITY_ATTACK': {},
                'INSULT': {},
                'PROFANITY': {},
                'THREAT': {}
            }
        }
        
        try:
            # Rate limiting: sleep between requests to avoid quota limits
            if i > continue_from:  # Don't wait for the first request
                time.sleep(rate_limit)
                
            response = requests.post(url, params={'key': api_key}, json=data)
            
            if response.status_code == 200:
                result = response.json()
                analysis = {
                    'comment': comment,
                    'toxicity': result['attributeScores']['TOXICITY']['summaryScore']['value'],
                    'severe_toxicity': result['attributeScores']['SEVERE_TOXICITY']['summaryScore']['value'],
                    'identity_attack': result['attributeScores']['IDENTITY_ATTACK']['summaryScore']['value'],
                    'insult': result['attributeScores']['INSULT']['summaryScore']['value'],
                    'profanity': result['attributeScores']['PROFANITY']['summaryScore']['value'],
                    'threat': result['attributeScores']['THREAT']['summaryScore']['value']
                }
                batch_results.append(analysis)
                results.append(analysis)
                
                # Save results in batches to avoid losing progress
                if len(batch_results) >= batch_size:
                    save_progress(results, output_file)
                    print(f"Saved progress ({i+1}/{total_comments})")
                    batch_results = []
                    
            elif response.status_code == 429:  # Too Many Requests
                print(f"Rate limit exceeded at comment {i+1}. Waiting for 60 seconds...")
                time.sleep(60)  # Wait for a minute before retrying
                i -= 1  # Retry this comment
                continue
            else:
                print(f"Error analyzing comment {i+1}/{total_comments}: {response.status_code}")
                print(f"Response: {response.text}")
        except Exception as e:
            print(f"Exception while analyzing comment {i+1}/{total_comments}: {e}")
            # If there's an exception, save progress and continue
            if batch_results:
                save_progress(results, output_file)
                print(f"Saved progress after exception ({i+1}/{total_comments})")
                batch_results = []
    
    # Save any remaining results
    if batch_results:
        save_progress(results, output_file)
    
    # Create final DataFrame with all results
    results_df = pd.DataFrame(results)
    
    # Save the results to the output file
    print(f"Saving final results to {output_file}")
    results_df.to_csv(output_file, index=False)
    print(f"Analysis complete. Processed {len(results)} comments out of {total_comments}.")
    
    return results_df

def save_progress(results, output_file):
    """Save current progress to file"""
    temp_file = output_file + '.temp'
    pd.DataFrame(results).to_csv(temp_file, index=False)
    # Rename to avoid corruption if interrupted during write
    if os.path.exists(temp_file):
        os.replace(temp_file, output_file)

def main():
    parser = argparse.ArgumentParser(description="Analyze comments using the Perspective API with rate limiting")
    parser.add_argument("--input", required=True, help="Path to the input CSV file containing comments")
    parser.add_argument("--output", required=True, help="Path to save the analysis results")
    parser.add_argument("--api_key", required=True, help="Perspective API key")
    parser.add_argument("--rate_limit", type=float, default=1.0, 
                        help="Time in seconds to wait between each API request (1.0 = 60 requests per minute)")
    parser.add_argument("--batch_size", type=int, default=10, 
                        help="Number of comments to process in one batch before saving progress")
    parser.add_argument("--continue_from", type=int, default=0, 
                        help="Index to continue from if process was interrupted")
    
    args = parser.parse_args()
    
    analyze_comments_with_rate_limit(
        args.input, 
        args.output, 
        args.api_key, 
        args.rate_limit, 
        args.batch_size, 
        args.continue_from
    )

if __name__ == "__main__":
    main() 