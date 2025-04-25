import json
import pandas as pd
import argparse
import os
from tqdm import tqdm

def extract_annotations(json_file, output_file, max_rows=None):
    """
    Extract PromptAnnotations from a JSON file and save them to a CSV file.
    
    Args:
        json_file (str): Path to the JSON file
        output_file (str): Path to save the output CSV file
        max_rows (int): Maximum number of rows to process (None for all rows)
    """
    print(f"Reading JSON file: {json_file}")
    
    # Read the JSON file line by line
    annotations = []
    with open(json_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
                
            try:
                data = json.loads(line)
                if 'PromptAnnotations' in data:
                    annotation = {
                        'Index': data.get('Index', i),
                        'Prompt': data.get('Prompt', ''),
                        'Toxicity': data['PromptAnnotations'].get('Toxicity', 0),
                        'IdentityAttack': data['PromptAnnotations'].get('IdentityAttack', 0),
                        'Bias': data['PromptAnnotations'].get('Bias', 0),
                        'Microaggression': data['PromptAnnotations'].get('Microaggression', 0),
                        'Violence': data['PromptAnnotations'].get('Violence', 0),
                        'SelfHarm': data['PromptAnnotations'].get('SelfHarm', 0),
                        'Insult': data['PromptAnnotations'].get('Insult', 0),
                        'SexualContent': data['PromptAnnotations'].get('SexualContent', 0)
                    }
                    annotations.append(annotation)
            except json.JSONDecodeError:
                print(f"Error decoding JSON at line {i+1}")
                continue
    
    # Convert to DataFrame
    df = pd.DataFrame(annotations)
    
    # Save to CSV
    print(f"Saving annotations to: {output_file}")
    df.to_csv(output_file, index=False)
    
    print(f"Extracted {len(df)} annotations")

def merge_annotations_with_code_switched(annotations_file, code_switched_file, output_file):
    """
    Merge annotations with code-switched data based on the prompt text.
    
    Args:
        annotations_file (str): Path to the annotations CSV file
        code_switched_file (str): Path to the code-switched CSV file
        output_file (str): Path to save the merged output CSV file
    """
    print(f"Reading annotations file: {annotations_file}")
    annotations_df = pd.read_csv(annotations_file)
    
    print(f"Reading code-switched file: {code_switched_file}")
    code_switched_df = pd.read_csv(code_switched_file)
    
    # Create a dictionary for quick lookup
    annotations_dict = {}
    for _, row in annotations_df.iterrows():
        prompt = row['Prompt']
        annotations_dict[prompt] = {
            'Toxicity': row['Toxicity'],
            'IdentityAttack': row['IdentityAttack'],
            'Bias': row['Bias'],
            'Microaggression': row['Microaggression'],
            'Violence': row['Violence'],
            'SelfHarm': row['SelfHarm'],
            'Insult': row['Insult'],
            'SexualContent': row['SexualContent']
        }
    
    # Add annotation columns to code-switched data
    for col in ['Toxicity', 'IdentityAttack', 'Bias', 'Microaggression', 'Violence', 'SelfHarm', 'Insult', 'SexualContent']:
        code_switched_df[col] = 0
    
    # Match annotations with code-switched data
    matched_count = 0
    for i, row in tqdm(code_switched_df.iterrows(), total=len(code_switched_df), desc="Matching annotations"):
        src = row['src']
        if src in annotations_dict:
            for col in annotations_dict[src].keys():
                code_switched_df.at[i, col] = annotations_dict[src][col]
            matched_count += 1
    
    # Save the merged data
    print(f"Saving merged data to: {output_file}")
    code_switched_df.to_csv(output_file, index=False)
    
    print(f"Matched {matched_count} out of {len(code_switched_df)} rows")

def main():
    parser = argparse.ArgumentParser(description="Extract PromptAnnotations from JSON files and merge with code-switched data")
    parser.add_argument("--json_file", required=True, help="Path to the JSON file containing PromptAnnotations")
    parser.add_argument("--code_switched_file", required=True, help="Path to the code-switched CSV file")
    parser.add_argument("--output_file", required=True, help="Path to save the output CSV file")
    parser.add_argument("--max_rows", type=int, default=None, help="Maximum number of rows to process from JSON file")
    
    args = parser.parse_args()
    
    # Create a temporary file for annotations
    temp_annotations_file = "temp_annotations.csv"
    
    # Extract annotations
    extract_annotations(args.json_file, temp_annotations_file, args.max_rows)
    
    # Merge annotations with code-switched data
    merge_annotations_with_code_switched(temp_annotations_file, args.code_switched_file, args.output_file)
    
    # Clean up temporary file
    if os.path.exists(temp_annotations_file):
        os.remove(temp_annotations_file)
        print(f"Removed temporary file: {temp_annotations_file}")

if __name__ == "__main__":
    main() 