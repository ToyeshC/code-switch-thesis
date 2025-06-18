#!/usr/bin/env python3
"""
Script to remove specific perspective analysis columns from the CSV file.
Removes columns related to perspective toxicity, identity attack, and insult measurements.
"""

import pandas as pd
import os

def remove_perspective_columns(csv_file_path):
    """
    Remove perspective analysis columns from the CSV file and overwrite it.
    
    Args:
        csv_file_path (str): Path to the CSV file
    """
    
    # Columns to remove
    columns_to_remove = [
        'perspective_src_toxicity',
        'perspective_src_identity_attack',
        'perspective_src_insult',
        'perspective_tgt_toxicity',
        'perspective_tgt_identity_attack',
        'perspective_tgt_insult',
        'perspective_generated_toxicity',
        'perspective_generated_identity_attack',
        'perspective_generated_insult'
    ]
    
    # Check if file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: File {csv_file_path} does not exist!")
        return
    
    try:
        # Read the CSV file
        print(f"Reading CSV file: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        
        # Show original shape
        print(f"Original shape: {df.shape}")
        print(f"Original columns: {df.columns.tolist()}")
        
        # Check which columns actually exist in the dataframe
        existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]
        missing_columns = [col for col in columns_to_remove if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: The following columns were not found in the CSV: {missing_columns}")
        
        if not existing_columns_to_remove:
            print("No specified columns found in the CSV file. Nothing to remove.")
            return
        
        print(f"Removing columns: {existing_columns_to_remove}")
        
        # Remove the specified columns
        df_cleaned = df.drop(columns=existing_columns_to_remove)
        
        # Show new shape
        print(f"New shape: {df_cleaned.shape}")
        print(f"Columns removed: {len(existing_columns_to_remove)}")
        
        # Save back to the same file (overwrite)
        print(f"Overwriting file: {csv_file_path}")
        df_cleaned.to_csv(csv_file_path, index=False)
        
        print("✅ Successfully removed perspective columns and updated the file!")
        print(f"Final columns in file: {df_cleaned.columns.tolist()}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    # File path
    csv_file_path = "temp_scripts/perspective_analysis_form.csv"
    
    # Run the function
    remove_perspective_columns(csv_file_path) 