import pandas as pd
import argparse
import os

def add_primary_key(base_file, source_file, base_lang, source_lang, output_dir):
    """
    Add a primary key to the original base and source language prompts and 
    create CSV files that will be used for further processing.
    
    Args:
        base_file (str): Path to the file containing base language prompts
        source_file (str): Path to the file containing source language prompts
        base_lang (str): Name of the base language (e.g., "hindi", "english", etc.)
        source_lang (str): Name of the source language (e.g., "hindi", "english", etc.)
        output_dir (str): Directory to save the output CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the base and source prompts
    with open(base_file, 'r', encoding='utf-8') as f:
        base_prompts = [line.strip() for line in f if line.strip()]
    
    with open(source_file, 'r', encoding='utf-8') as f:
        source_prompts = [line.strip() for line in f if line.strip()]
    
    # Ensure both files have the same number of lines
    if len(base_prompts) != len(source_prompts):
        print(f"Warning: {base_lang} file has {len(base_prompts)} prompts but {source_lang} file has {len(source_prompts)} prompts")
        # Use the smaller number of prompts
        n = min(len(base_prompts), len(source_prompts))
        base_prompts = base_prompts[:n]
        source_prompts = source_prompts[:n]
    
    # Create DataFrames with primary keys
    base_df = pd.DataFrame({
        'prompt_id': range(1, len(base_prompts) + 1),
        'sentence': base_prompts,
        'language': base_lang
    })
    
    source_df = pd.DataFrame({
        'prompt_id': range(1, len(source_prompts) + 1),
        'sentence': source_prompts,
        'language': source_lang
    })
    
    # Create combined DataFrame for all prompts
    all_prompts_df = pd.concat([base_df, source_df], ignore_index=False)
    
    # Save the DataFrames to CSV files
    base_df.to_csv(os.path.join(output_dir, f'{base_lang}_prompts_with_id.csv'), index=False)
    source_df.to_csv(os.path.join(output_dir, f'{source_lang}_prompts_with_id.csv'), index=False)
    all_prompts_df.to_csv(os.path.join(output_dir, 'all_prompts_with_id.csv'), index=False)
    
    print(f"Created CSV files with primary keys:")
    print(f"  - {base_lang.capitalize()} prompts: {os.path.join(output_dir, f'{base_lang}_prompts_with_id.csv')}")
    print(f"  - {source_lang.capitalize()} prompts: {os.path.join(output_dir, f'{source_lang}_prompts_with_id.csv')}")
    print(f"  - All prompts: {os.path.join(output_dir, 'all_prompts_with_id.csv')}")
    
    return base_df, source_df

def main():
    parser = argparse.ArgumentParser(description="Add primary key to language prompts")
    parser.add_argument("--base_file", required=True, help="Path to the file containing base language prompts")
    parser.add_argument("--source_file", required=True, help="Path to the file containing source language prompts")
    parser.add_argument("--base_lang", required=True, help="Name of the base language (e.g., 'hindi', 'english')")
    parser.add_argument("--source_lang", required=True, help="Name of the source language (e.g., 'hindi', 'english')")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output CSV files")
    
    args = parser.parse_args()
    add_primary_key(args.base_file, args.source_file, args.base_lang, args.source_lang, args.output_dir)

if __name__ == "__main__":
    main() 