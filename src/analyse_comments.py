import pandas as pd
import requests
import argparse
import os
import sys

# Add the directory containing config.py to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import the API key from config.py
try:
    from config import PERSPECTIVE_API_KEY
    print("Successfully imported API key from config.py")
except ImportError:
    PERSPECTIVE_API_KEY = None
    print("Warning: Could not import PERSPECTIVE_API_KEY from config.py")

def analyze_comments(input_file, output_file, api_key):
    """
    Analyze comments using the Perspective API for toxicity and other attributes.
    
    Args:
        input_file (str): Path to the input CSV file containing comments
        output_file (str): Path to save the analysis results
        api_key (str): Perspective API key
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
    print(f"Found {len(comments)} comments to analyze")
    
    # Define the Perspective API endpoint
    url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
    
    # Analyze each comment
    results = []
    for i, comment in enumerate(comments):
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
                results.append(analysis)
                print(f"Analyzed comment {i+1}/{len(comments)}: {comment[:50]}...")
            else:
                print(f"Error analyzing comment {i+1}/{len(comments)}: {response.status_code}")
                print(f"Response: {response.text}")
        except Exception as e:
            print(f"Exception while analyzing comment {i+1}/{len(comments)}: {e}")
    
    # Create a DataFrame with the results
    results_df = pd.DataFrame(results)
    
    # Save the results to a CSV file
    print(f"Saving analysis results to {output_file}")
    results_df.to_csv(output_file, index=False)
    print(f"Analysis complete. Processed {len(results)} comments.")
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze comments using the Perspective API")
    parser.add_argument("--input", required=True, help="Path to the input CSV file containing comments")
    parser.add_argument("--output", required=True, help="Path to save the analysis results")
    parser.add_argument("--api_key", help="Perspective API key (overrides config.py)")
    parser.add_argument("--api_key_file", help="Path to a file containing the Perspective API key (overrides config.py)")
    
    args = parser.parse_args()
    
    # Get the API key with priority:
    # 1. Command line argument
    # 2. API key file
    # 3. Environment variable
    # 4. config.py
    api_key = None
    
    if args.api_key:
        api_key = args.api_key
        print("Using API key from command line argument")
    elif args.api_key_file:
        try:
            with open(args.api_key_file, 'r') as f:
                api_key = f.read().strip()
                print(f"Using API key from file: {args.api_key_file}")
        except Exception as e:
            print(f"Error reading API key file: {e}")
    elif "PERSPECTIVE_API_KEY" in os.environ:
        api_key = os.environ["PERSPECTIVE_API_KEY"]
        print("Using API key from environment variable")
    elif PERSPECTIVE_API_KEY:
        api_key = PERSPECTIVE_API_KEY
        print("Using API key from config.py")
    
    if not api_key:
        print("Error: No API key provided. Please provide an API key using --api_key, --api_key_file, set the PERSPECTIVE_API_KEY environment variable, or define it in config.py.")
        exit(1)
    
    analyze_comments(args.input, args.output, api_key) 