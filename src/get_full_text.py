import pandas as pd
import argparse

def extract_full_text(input_file, output_file):
    """
    Extract the full_text column from a CSV file and save it to a new CSV file.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the output CSV file
    """
    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Check if 'full_text' column exists
    if 'full_text' not in df.columns:
        print("Error: 'full_text' column not found in the input CSV file.")
        return
    
    # Extract only the full_text column
    full_text_df = df[['full_text']]
    
    # Save to a new CSV file
    print(f"Saving full_text to {output_file}")
    full_text_df.to_csv(output_file, index=False)
    
    print(f"Successfully extracted {len(full_text_df)} full_text entries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract full_text column from a CSV file")
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--output", required=True, help="Path to save the output CSV file")
    
    args = parser.parse_args()
    
    extract_full_text(args.input, args.output)
