import pandas as pd
import argparse
import os

def add_primary_keys(input_file, output_file, key_prefix='', max_rows=None):
    """
    Add primary keys to a dataset and save it.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the output CSV file
        key_prefix (str): Prefix for the primary keys (e.g., 'cs_' for code-switched)
        max_rows (int): Maximum number of rows to process (None for all rows)
    """
    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Limit the number of rows if specified
    if max_rows is not None and max_rows < len(df):
        print(f"Limiting to first {max_rows} rows out of {len(df)} total rows")
        df = df.head(max_rows)
    
    # Generate primary keys
    df['primary_key'] = [f"{key_prefix}{i+1:06d}" for i in range(len(df))]
    
    # Save the data with primary keys
    print(f"Saving data with primary keys to: {output_file}")
    df.to_csv(output_file, index=False)
    
    print(f"Added {len(df)} primary keys with prefix '{key_prefix}'")

def main():
    parser = argparse.ArgumentParser(description="Add primary keys to datasets")
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--output", required=True, help="Path to save the output CSV file")
    parser.add_argument("--key_prefix", default="cs_", help="Prefix for the primary keys")
    parser.add_argument("--max_rows", type=int, default=None, help="Maximum number of rows to process")
    
    args = parser.parse_args()
    add_primary_keys(args.input, args.output, args.key_prefix, args.max_rows)

if __name__ == "__main__":
    main() 