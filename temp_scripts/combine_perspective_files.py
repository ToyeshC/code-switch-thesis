#!/usr/bin/env python3
"""
Script to combine three individual perspective CSV files into a single combined file.
This script reads the llama3, llama31, and aya perspective files and combines them
into the format of the reference continuations.csv file.
"""

import pandas as pd
import os

def main():
    # Define file paths
    input_dir = "new_outputs/perspective_combined_full"
    output_dir = "temp_outputs"
    
    llama3_file = os.path.join(input_dir, "llama3_combined_perspective.csv")
    llama31_file = os.path.join(input_dir, "llama31_combined_perspective.csv")
    aya_file = os.path.join(input_dir, "aya_combined_perspective.csv")
    
    output_file = os.path.join(output_dir, "combined_continuations.csv")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data from the three files...")
    
    # Read the three files
    llama3_df = pd.read_csv(llama3_file)
    llama31_df = pd.read_csv(llama31_file)
    aya_df = pd.read_csv(aya_file)
    
    print(f"Loaded {len(llama3_df)} rows from llama3 file")
    print(f"Loaded {len(llama31_df)} rows from llama31 file")
    print(f"Loaded {len(aya_df)} rows from aya file")
    
    # Verify all files have the same number of rows
    if not (len(llama3_df) == len(llama31_df) == len(aya_df)):
        print("Error: Files have different number of rows!")
        print(f"llama3: {len(llama3_df)}, llama31: {len(llama31_df)}, aya: {len(aya_df)}")
        return
    
    # Start with the basic columns from any of the files (they should be the same)
    basic_columns = [
        'src', 'tgt', 'generated', 'method', 'model', 'direction', 'primary_key',
        'hindi_word_count', 'english_word_count', 'romanized_hindi_count',
        'total_hindi_count', 'total_words', 'hindi_percent', 'romanized_hindi_percent',
        'total_hindi_percent', 'english_percent'
    ]
    
    # Create the combined dataframe starting with basic columns from llama3 file
    combined_df = llama3_df[basic_columns].copy()
    
    # Add continuation columns from each model
    # Extract continuation columns from each file
    llama3_continuation_cols = [
        'llama3_continuation_src', 'llama3_continuation_tgt', 'llama3_continuation_generated'
    ]
    llama31_continuation_cols = [
        'llama31_continuation_src', 'llama31_continuation_tgt', 'llama31_continuation_generated'
    ]
    aya_continuation_cols = [
        'aya_continuation_src', 'aya_continuation_tgt', 'aya_continuation_generated'
    ]
    
    # Rename and add continuation columns
    for col in llama3_continuation_cols:
        if col in llama3_df.columns:
            combined_df[col.replace('llama3_continuation', 'llama3')] = llama3_df[col]
    
    for col in llama31_continuation_cols:
        if col in llama31_df.columns:
            combined_df[col.replace('llama31_continuation', 'llama31')] = llama31_df[col]
    
    for col in aya_continuation_cols:
        if col in aya_df.columns:
            combined_df[col.replace('aya_continuation', 'aya')] = aya_df[col]
    
    # Rename to match the reference file format
    column_rename_map = {
        'llama3_src': 'llama3_src_continuation',
        'llama3_tgt': 'llama3_tgt_continuation', 
        'llama3_generated': 'llama3_generated_continuation',
        'llama31_src': 'llama31_src_continuation',
        'llama31_tgt': 'llama31_tgt_continuation',
        'llama31_generated': 'llama31_generated_continuation',
        'aya_src': 'aya_src_continuation',
        'aya_tgt': 'aya_tgt_continuation',
        'aya_generated': 'aya_generated_continuation'
    }
    
    combined_df = combined_df.rename(columns=column_rename_map)
    
    # Define the final column order to match the reference file
    final_columns = [
        'src', 'tgt', 'generated', 'method', 'model', 'direction', 'primary_key',
        'hindi_word_count', 'english_word_count', 'romanized_hindi_count',
        'total_hindi_count', 'total_words', 'hindi_percent', 'romanized_hindi_percent',
        'total_hindi_percent', 'english_percent',
        'llama3_src_continuation', 'llama3_tgt_continuation', 'llama3_generated_continuation',
        'llama31_src_continuation', 'llama31_tgt_continuation', 'llama31_generated_continuation',
        'aya_src_continuation', 'aya_tgt_continuation', 'aya_generated_continuation'
    ]
    
    # Select only the columns that exist in our combined dataframe
    available_columns = [col for col in final_columns if col in combined_df.columns]
    combined_df = combined_df[available_columns]
    
    print(f"Combined dataframe shape: {combined_df.shape}")
    print(f"Columns in combined file: {list(combined_df.columns)}")
    
    # Save the combined file
    combined_df.to_csv(output_file, index=False)
    print(f"Combined file saved to: {output_file}")
    
    # Print first few rows for verification
    print("\nFirst 3 rows of combined data:")
    print(combined_df.head(3).to_string())

if __name__ == "__main__":
    main() 