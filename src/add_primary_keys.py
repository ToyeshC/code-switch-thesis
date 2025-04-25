import pandas as pd
import argparse
import os

def add_primary_keys(input_file, output_file, key_prefix=''):
    """
    Add primary keys to a dataset and save it.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the output CSV file
        key_prefix (str): Prefix for the primary keys (e.g., 'hi_' for Hindi, 'en_' for English)
    """
    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Generate primary keys
    df['primary_key'] = [f"{key_prefix}{i+1:06d}" for i in range(len(df))]
    
    # Save the data with primary keys
    print(f"Saving data with primary keys to: {output_file}")
    df.to_csv(output_file, index=False)
    
    print(f"Added {len(df)} primary keys with prefix '{key_prefix}'")

def propagate_keys(input_file, output_file, key_column='primary_key'):
    """
    Ensure primary keys are present in the output file.
    If they're missing, add them based on the input file's keys.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the output CSV file
        key_column (str): Name of the primary key column
    """
    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Check if primary keys exist
    if key_column not in df.columns:
        print(f"Warning: {key_column} column not found in {input_file}")
        return False
    
    # Save the data with primary keys
    print(f"Saving data with primary keys to: {output_file}")
    df.to_csv(output_file, index=False)
    
    print(f"Propagated {len(df)} primary keys")
    return True

def main():
    parser = argparse.ArgumentParser(description="Add and propagate primary keys through datasets")
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--output", required=True, help="Path to save the output CSV file")
    parser.add_argument("--key_prefix", default="", help="Prefix for the primary keys")
    parser.add_argument("--propagate", action="store_true", help="Propagate existing keys instead of generating new ones")
    
    args = parser.parse_args()
    
    if args.propagate:
        propagate_keys(args.input, args.output)
    else:
        add_primary_keys(args.input, args.output, args.key_prefix)

if __name__ == "__main__":
    main() 