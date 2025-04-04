import argparse
import pandas as pd
import os
import json

def extract_prompts_to_text(input_csv, output_txt, id_map_file):
    """
    Extract prompts from a CSV file and save them to a text file.
    Also create a mapping file to preserve the prompt IDs.
    
    Args:
        input_csv (str): Path to the input CSV file
        output_txt (str): Path to the output text file
        id_map_file (str): Path to save the ID mapping JSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    os.makedirs(os.path.dirname(id_map_file), exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Ensure required columns exist
    if 'sentence' not in df.columns:
        raise ValueError(f"'sentence' column not found in {input_csv}")
    if 'prompt_id' not in df.columns:
        raise ValueError(f"'prompt_id' column not found in {input_csv}")
    
    # Create ID mapping
    id_mapping = {}
    line_number = 0
    
    # Extract prompts and write to text file
    with open(output_txt, 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            f.write(f"{row['sentence']}\n")
            # Store mapping between line number and prompt_id
            id_mapping[line_number] = int(row['prompt_id'])
            line_number += 1
    
    # Save ID mapping to JSON file
    with open(id_map_file, 'w', encoding='utf-8') as f:
        json.dump(id_mapping, f, indent=2)
    
    print(f"Extracted {len(df)} prompts to {output_txt}")
    print(f"Saved ID mapping to {id_map_file}")

def main():
    parser = argparse.ArgumentParser(description="Extract prompts from CSV files to text files")
    parser.add_argument('--input_csv', required=True, help='Path to the input CSV file')
    parser.add_argument('--output_txt', required=True, help='Path to the output text file')
    parser.add_argument('--id_map', required=True, help='Path to save the ID mapping JSON file')
    
    args = parser.parse_args()
    extract_prompts_to_text(args.input_csv, args.output_txt, args.id_map)

if __name__ == '__main__':
    main() 