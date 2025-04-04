import pandas as pd
import argparse
import os

def filter_language_mix(input_file, output_file, min_hindi_percent=20, min_english_percent=20, 
                        min_hindi_prob=0.2, min_english_prob=0.2, use_indic_format=False):
    """
    Filter sentences based on their language mixture.
    
    Args:
        input_file (str): Path to the input CSV file with language detection results
        output_file (str): Path to save the filtered results
        min_hindi_percent (float): Minimum percentage of Hindi words required
        min_english_percent (float): Minimum percentage of English words required
        min_hindi_prob (float): Minimum Hindi probability (for Indic LID format)
        min_english_prob (float): Minimum English probability (for Indic LID format)
        use_indic_format (bool): Whether the input is in Indic LID format
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Read the input file
    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Make sure prompt_id is in the DataFrame
    if 'prompt_id' not in df.columns:
        print("Warning: No 'prompt_id' column found. Results may not be traceable.")
    
    # Filter the DataFrame based on format
    if use_indic_format:
        if 'hindi_prob' not in df.columns or 'english_prob' not in df.columns:
            print("Error: Input file does not have the expected Indic LID format.")
            return
        
        # Filter based on probabilities
        filtered_df = df[(df['hindi_prob'] >= min_hindi_prob) & 
                          (df['english_prob'] >= min_english_prob)]
        
        print(f"Filtering based on Indic LID probabilities:")
        print(f"  - Minimum Hindi probability: {min_hindi_prob}")
        print(f"  - Minimum English probability: {min_english_prob}")
    else:
        # Check if we have the required columns
        required_columns = ['hindi_percent', 'english_percent']
        if not all(col in df.columns for col in required_columns):
            # Try alternative column names
            if 'total_hindi_percent' in df.columns:
                df['hindi_percent'] = df['total_hindi_percent']
            if 'hindi_word_count' in df.columns and 'english_word_count' in df.columns and 'total_words' in df.columns:
                # Calculate percentages if they're not already there
                df['hindi_percent'] = (df['hindi_word_count'] + df.get('romanized_hindi_count', 0)) / df['total_words'] * 100
                df['english_percent'] = df['english_word_count'] / df['total_words'] * 100
            else:
                print("Error: Input file does not have the required columns for language filtering.")
                return
        
        # Filter based on language percentages
        filtered_df = df[(df['hindi_percent'] >= min_hindi_percent) & 
                          (df['english_percent'] >= min_english_percent)]
        
        print(f"Filtering based on language percentages:")
        print(f"  - Minimum Hindi percentage: {min_hindi_percent}%")
        print(f"  - Minimum English percentage: {min_english_percent}%")
    
    # Add a flag for code-switched content
    filtered_df['is_code_switched'] = True
    
    # Report filtering results
    print(f"Original dataset size: {len(df)} sentences")
    print(f"Filtered dataset size: {len(filtered_df)} sentences")
    print(f"Removed {len(df) - len(filtered_df)} sentences that didn't meet the criteria")
    
    # Save the filtered DataFrame
    filtered_df.to_csv(output_file, index=False)
    print(f"Saved filtered results to {output_file}")
    
    return filtered_df

def main():
    parser = argparse.ArgumentParser(description="Filter sentences based on language mixture")
    parser.add_argument('--input', required=True, help='Path to the input CSV file with language detection results')
    parser.add_argument('--output', required=True, help='Path to save the filtered results')
    parser.add_argument('--min_hindi_percent', type=float, default=20, help='Minimum percentage of Hindi words required')
    parser.add_argument('--min_english_percent', type=float, default=20, help='Minimum percentage of English words required')
    parser.add_argument('--min_hindi_prob', type=float, default=0.2, help='Minimum Hindi probability (for Indic LID format)')
    parser.add_argument('--min_english_prob', type=float, default=0.2, help='Minimum English probability (for Indic LID format)')
    parser.add_argument('--use_indic_format', action='store_true', help='Whether the input is in Indic LID format')
    
    args = parser.parse_args()
    filter_language_mix(args.input, args.output, args.min_hindi_percent, args.min_english_percent,
                       args.min_hindi_prob, args.min_english_prob, args.use_indic_format)

if __name__ == '__main__':
    main() 