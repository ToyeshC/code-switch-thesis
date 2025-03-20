import pandas as pd
import argparse

def filter_sentences(input_file, output_file):
    """
    Filter sentences from a CSV file based on language composition criteria.
    Preserves the primary key throughout the process.
    
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
    
    # Ensure 'prompt_id' is in the dataframe
    if 'prompt_id' not in df.columns:
        print("Error: No 'prompt_id' column found in the input file.")
        return
    
    # Store original count
    original_count = len(df)
    print(f"Original number of sentences: {original_count}")
    
    # Apply filters
    # 1. Remove English-only sentences
    english_only = (df['total_hindi_percent'] == 0) & (df['english_percent'] == 100)
    english_only_count = english_only.sum()
    
    # 2. Remove Hindi-only sentences
    hindi_only = (df['english_percent'] == 0) & (df['total_hindi_percent'] == 100)
    hindi_only_count = hindi_only.sum()
    
    # 3. Remove English-dominant sentences
    english_dominant = (df['english_percent'] - df['total_hindi_percent'] >= 45)
    english_dominant_count = english_dominant.sum() - english_only_count  # Avoid double counting
    
    # Create a combined filter (keep rows that don't match any of the conditions)
    keep_mask = ~(english_only | hindi_only | english_dominant)
    
    # Apply the filter
    filtered_df = df[keep_mask].reset_index(drop=True)
    
    # Save the filtered data
    filtered_df.to_csv(output_file, index=False)
    
    # Save the mapping of which primary keys were kept or filtered out
    filter_status_df = pd.DataFrame({
        'prompt_id': df['prompt_id'],
        'is_kept': keep_mask,
        'filter_reason': 'kept'
    })
    
    # Add reasons for filtering
    filter_status_df.loc[english_only, 'filter_reason'] = 'english_only'
    filter_status_df.loc[hindi_only, 'filter_reason'] = 'hindi_only'
    filter_status_df.loc[english_dominant & ~english_only, 'filter_reason'] = 'english_dominant'
    
    # Save filter status
    filter_status_file = output_file.replace('.csv', '_filter_status.csv')
    filter_status_df.to_csv(filter_status_file, index=False)
    
    # Print summary
    print("\n===== Filtering Summary =====")
    print(f"Removed English-only sentences: {english_only_count} ({english_only_count/original_count*100:.2f}%)")
    print(f"Removed Hindi-only sentences: {hindi_only_count} ({hindi_only_count/original_count*100:.2f}%)")
    print(f"Removed English-dominant sentences: {english_dominant_count} ({english_dominant_count/original_count*100:.2f}%)")
    print(f"Total sentences removed: {original_count - len(filtered_df)} ({(original_count - len(filtered_df))/original_count*100:.2f}%)")
    print(f"Remaining sentences: {len(filtered_df)} ({len(filtered_df)/original_count*100:.2f}%)")
    print(f"\nFiltered data saved to: {output_file}")
    print(f"Filter status saved to: {filter_status_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter sentences based on language composition criteria while preserving primary key")
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--output", required=True, help="Path to save the filtered CSV file")
    
    args = parser.parse_args()
    filter_sentences(args.input, args.output) 