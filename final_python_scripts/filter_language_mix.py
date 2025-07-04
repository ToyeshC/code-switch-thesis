import pandas as pd
import argparse
import os

def filter_sentences(input_file, output_file):
    """
    Filter sentences from a CSV file based on language composition criteria.
    
    Removes sentences where:
    1. total_hindi_percent = 0 and english_percent = 100 (English-only)
    2. english_percent = 0 and total_hindi_percent = 100 (Hindi-only)
    3. english_percent - total_hindi_percent >= 45 (English-dominant)
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the filtered CSV file
    """
    # Read the CSV file
    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Check if primary key exists
    if 'primary_key' not in df.columns:
        print("Warning: No primary_key column found in input file. Adding one...")
        df['primary_key'] = [f"cs_{i+1:06d}" for i in range(len(df))]
    
    # Store original count
    original_count = len(df)
    print(f"Original number of sentences: {original_count}")
    
    # Apply filters
    # 1. Remove English-only sentences
    english_only = (df['total_hindi_percent'] == 0) & (df['english_percent'] == 100)
    english_only_count = english_only.sum()
    english_only_keys = df[english_only]['primary_key'].tolist()
    
    # 2. Remove Hindi-only sentences
    hindi_only = (df['english_percent'] == 0) & (df['total_hindi_percent'] == 100)
    hindi_only_count = hindi_only.sum()
    hindi_only_keys = df[hindi_only]['primary_key'].tolist()
    
    # 3. Remove English-dominant sentences
    english_dominant = (df['english_percent'] - df['total_hindi_percent'] >= 45)
    english_dominant_count = english_dominant.sum() - english_only_count  # Avoid double counting
    english_dominant_keys = df[english_dominant & ~english_only]['primary_key'].tolist()
    
    # Create a combined filter (keep rows that don't match any of the conditions)
    keep_mask = ~(english_only | hindi_only | english_dominant)
    
    # Apply the filter
    filtered_df = df[keep_mask].reset_index(drop=True)
    
    # Save the filtered data
    filtered_df.to_csv(output_file, index=False)
    
    # Save the filtered out keys to a separate file
    filtered_keys = {
        'english_only': english_only_keys,
        'hindi_only': hindi_only_keys,
        'english_dominant': english_dominant_keys
    }
    
    # Create output directory for filtered keys if it doesn't exist
    output_dir = os.path.dirname(output_file)
    filtered_keys_file = os.path.join(output_dir, 'filtered_keys.csv')
    
    # Create a DataFrame with all filtered keys and their categories
    filtered_keys_data = []
    for category, keys in filtered_keys.items():
        for key in keys:
            filtered_keys_data.append({
                'primary_key': key,
                'filter_category': category
            })
    
    filtered_keys_df = pd.DataFrame(filtered_keys_data)
    filtered_keys_df.to_csv(filtered_keys_file, index=False)
    
    # Print summary
    print("\n===== Filtering Summary =====")
    print(f"Removed English-only sentences: {english_only_count} ({english_only_count/original_count*100:.2f}%)")
    print(f"Removed Hindi-only sentences: {hindi_only_count} ({hindi_only_count/original_count*100:.2f}%)")
    print(f"Removed English-dominant sentences: {english_dominant_count} ({english_dominant_count/original_count*100:.2f}%)")
    print(f"Total sentences removed: {original_count - len(filtered_df)} ({(original_count - len(filtered_df))/original_count*100:.2f}%)")
    print(f"Remaining sentences: {len(filtered_df)} ({len(filtered_df)/original_count*100:.2f}%)")
    print(f"\nFiltered data saved to: {output_file}")
    print(f"Filtered keys saved to: {filtered_keys_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter sentences based on language composition criteria")
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--output", required=True, help="Path to save the filtered CSV file")
    
    args = parser.parse_args()
    filter_sentences(args.input, args.output) 