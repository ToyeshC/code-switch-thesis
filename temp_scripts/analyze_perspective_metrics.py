import pandas as pd
import numpy as np
from googleapiclient import discovery
import time
from pathlib import Path
import sys
from tqdm import tqdm

# Add the parent directory to Python path to find config.py
sys.path.append(str(Path(__file__).parent.parent))
import config

# Constants
RATE_LIMIT = 100  # requests per minute
BATCH_SIZE = 50  # process in batches to avoid memory issues
SLEEP_TIME = 60 / RATE_LIMIT  # time to sleep between requests

def get_perspective_client():
    """Initialize and return the Perspective API client"""
    return discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=config.PERSPECTIVE_API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

def analyze_text(text, client):
    """Analyze a single text using Perspective API"""
    if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
        return {
            'TOXICITY': np.nan,
            'IDENTITY_ATTACK': np.nan,
            'INSULT': np.nan
        }
    
    try:
        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {
                'TOXICITY': {},
                'IDENTITY_ATTACK': {},
                'INSULT': {}
            }
        }
        
        response = client.comments().analyze(body=analyze_request).execute()
        
        return {
            'TOXICITY': response['attributeScores']['TOXICITY']['summaryScore']['value'],
            'IDENTITY_ATTACK': response['attributeScores']['IDENTITY_ATTACK']['summaryScore']['value'],
            'INSULT': response['attributeScores']['INSULT']['summaryScore']['value']
        }
    except Exception as e:
        print(f"Error analyzing text: {str(e)}")
        return {
            'TOXICITY': np.nan,
            'IDENTITY_ATTACK': np.nan,
            'INSULT': np.nan
        }

def process_batch(df_batch, client):
    """Process a batch of texts and return results"""
    results = []
    for _, row in tqdm(df_batch.iterrows(), total=len(df_batch), desc="Processing batch"):
        # Analyze source text
        src_result = analyze_text(row['src'], client)
        time.sleep(SLEEP_TIME)  # Rate limiting
        
        # Analyze target text
        tgt_result = analyze_text(row['tgt'], client)
        time.sleep(SLEEP_TIME)  # Rate limiting
        
        # Analyze generated text
        gen_result = analyze_text(row['generated'], client)
        time.sleep(SLEEP_TIME)  # Rate limiting
        
        # Combine results with clear column names
        result = {
            'perspective_src_toxicity': src_result['TOXICITY'],
            'perspective_src_identity_attack': src_result['IDENTITY_ATTACK'],
            'perspective_src_insult': src_result['INSULT'],
            'perspective_tgt_toxicity': tgt_result['TOXICITY'],
            'perspective_tgt_identity_attack': tgt_result['IDENTITY_ATTACK'],
            'perspective_tgt_insult': tgt_result['INSULT'],
            'perspective_generated_toxicity': gen_result['TOXICITY'],
            'perspective_generated_identity_attack': gen_result['IDENTITY_ATTACK'],
            'perspective_generated_insult': gen_result['INSULT']
        }
        results.append(result)
    
    return pd.DataFrame(results)

def main():
    # Create output directory
    output_dir = Path('temp_scripts/perspective_analysis_outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Load the data
    print("Loading data...")
    df = pd.read_csv('new_outputs/filtered_output_full.csv')
    
    # Initialize Perspective API client
    print("Initializing Perspective API client...")
    client = get_perspective_client()
    
    # Process data in batches
    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    all_results = []
    
    for i in range(0, len(df), BATCH_SIZE):
        print(f"\nProcessing batch {i//BATCH_SIZE + 1}/{total_batches}")
        batch = df.iloc[i:i+BATCH_SIZE]
        batch_results = process_batch(batch, client)
        all_results.append(batch_results)
        
        # Save intermediate results
        intermediate_df = pd.concat(all_results, ignore_index=True)
        intermediate_df.to_csv(output_dir / 'intermediate_results.csv', index=False)
        
        print(f"Saved intermediate results after batch {i//BATCH_SIZE + 1}")
    
    # Combine all results
    final_results = pd.concat(all_results, ignore_index=True)
    
    # Add the original columns
    final_df = pd.concat([df, final_results], axis=1)
    
    # Save final results
    output_file = output_dir / 'perspective_analysis_results.csv'
    final_df.to_csv(output_file, index=False)
    print(f"\nAnalysis complete! Results saved to {output_file}")

if __name__ == "__main__":
    main() 