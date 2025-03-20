import pandas as pd
import argparse
import os

def add_primary_key(hindi_file, english_file, output_dir):
    """
    Add a primary key to the original Hindi and English prompts and 
    create CSV files that will be used for further processing.
    
    Args:
        hindi_file (str): Path to the file containing Hindi prompts
        english_file (str): Path to the file containing English prompts
        output_dir (str): Directory to save the output CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the Hindi and English prompts
    with open(hindi_file, 'r', encoding='utf-8') as f:
        hindi_prompts = [line.strip() for line in f if line.strip()]
    
    with open(english_file, 'r', encoding='utf-8') as f:
        english_prompts = [line.strip() for line in f if line.strip()]
    
    # Ensure both files have the same number of lines
    if len(hindi_prompts) != len(english_prompts):
        print(f"Warning: Hindi file has {len(hindi_prompts)} prompts but English file has {len(english_prompts)} prompts")
        # Use the smaller number of prompts
        n = min(len(hindi_prompts), len(english_prompts))
        hindi_prompts = hindi_prompts[:n]
        english_prompts = english_prompts[:n]
    
    # Create DataFrames with primary keys
    hindi_df = pd.DataFrame({
        'prompt_id': range(1, len(hindi_prompts) + 1),
        'sentence': hindi_prompts,
        'language': 'hindi'
    })
    
    english_df = pd.DataFrame({
        'prompt_id': range(1, len(english_prompts) + 1),
        'sentence': english_prompts,
        'language': 'english'
    })
    
    # Create combined DataFrame for all prompts
    all_prompts_df = pd.concat([hindi_df, english_df], ignore_index=False)
    
    # Save the DataFrames to CSV files
    hindi_df.to_csv(os.path.join(output_dir, 'hindi_prompts_with_id.csv'), index=False)
    english_df.to_csv(os.path.join(output_dir, 'english_prompts_with_id.csv'), index=False)
    all_prompts_df.to_csv(os.path.join(output_dir, 'all_prompts_with_id.csv'), index=False)
    
    print(f"Created CSV files with primary keys:")
    print(f"  - Hindi prompts: {os.path.join(output_dir, 'hindi_prompts_with_id.csv')}")
    print(f"  - English prompts: {os.path.join(output_dir, 'english_prompts_with_id.csv')}")
    print(f"  - All prompts: {os.path.join(output_dir, 'all_prompts_with_id.csv')}")
    
    return hindi_df, english_df

def main():
    parser = argparse.ArgumentParser(description="Add primary key to Hindi and English prompts")
    parser.add_argument("--hindi", required=True, help="Path to the file containing Hindi prompts")
    parser.add_argument("--english", required=True, help="Path to the file containing English prompts")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output CSV files")
    
    args = parser.parse_args()
    add_primary_key(args.hindi, args.english, args.output_dir)

if __name__ == "__main__":
    main() 