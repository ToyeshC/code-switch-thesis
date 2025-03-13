import pandas as pd
import fasttext
import argparse
import os
from tqdm import tqdm

def main():
    """
    Process a CSV file containing generated sentences, detect languages using FastText,
    and count Hindi and English words in each sentence.
    """
    parser = argparse.ArgumentParser(
        description="Detect languages and count Hindi/English words in generated sentences"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the CSV file containing generated sentences",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save the processed CSV file (defaults to input_file_processed.csv)",
    )
    parser.add_argument(
        "--fasttext_model",
        type=str,
        default="lid.176.bin",
        help="Path to the FastText language identification model",
    )
    args = parser.parse_args()

    # Set default output file if not provided
    if args.output_file is None:
        base_name = os.path.splitext(args.input_file)[0]
        args.output_file = f"{base_name}_processed.csv"

    # Load the FastText language identification model
    print(f"Loading FastText model from {args.fasttext_model}...")
    model = fasttext.load_model(args.fasttext_model)

    # Read the input CSV file
    print(f"Reading input file: {args.input_file}")
    df = pd.read_csv(args.input_file)
    
    # Initialize new columns for word counts
    df['hindi_word_count'] = 0
    df['english_word_count'] = 0
    
    # Process each sentence
    print("Processing sentences and counting words by language...")
    
    # Function to process each sentence
    def process_sentence(sentence):
        if not isinstance(sentence, str) or pd.isna(sentence):
            return 0, 0  # Return zeros for non-string or NaN values
        
        # Tokenize the sentence into words
        words = sentence.split()
        
        # Count Hindi and English words
        hindi_count = 0
        english_count = 0
        
        for word in words:
            # Get language prediction from FastText
            # The model returns a tuple with (labels, probabilities)
            prediction = model.predict(word, k=1)
            lang = prediction[0][0].replace('__label__', '')
            
            # Count words by language
            if lang == 'hi':
                hindi_count += 1
            elif lang == 'en':
                english_count += 1
            
        return hindi_count, english_count
    
    # Apply the processing function to the 'generated' column if it exists
    if 'generated' in df.columns:
        # Use tqdm for progress tracking
        for i in tqdm(range(len(df))):
            hindi_count, english_count = process_sentence(df.loc[i, 'generated'])
            df.loc[i, 'hindi_word_count'] = hindi_count
            df.loc[i, 'english_word_count'] = english_count
    else:
        print("Warning: 'generated' column not found. Please specify the column containing sentences.")
        # Try to find a column that might contain sentences
        text_columns = [col for col in df.columns if df[col].dtype == 'object']
        if text_columns:
            print(f"Found potential text columns: {text_columns}")
            print(f"Processing the first text column: {text_columns[0]}")
            for i in tqdm(range(len(df))):
                hindi_count, english_count = process_sentence(df.loc[i, text_columns[0]])
                df.loc[i, 'hindi_word_count'] = hindi_count
                df.loc[i, 'english_word_count'] = english_count
    
    # Save the processed DataFrame to a new CSV file
    print(f"Saving processed data to {args.output_file}")
    df.to_csv(args.output_file, index=False)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total sentences processed: {len(df)}")
    print(f"Average Hindi words per sentence: {df['hindi_word_count'].mean():.2f}")
    print(f"Average English words per sentence: {df['english_word_count'].mean():.2f}")
    print(f"Sentences with both Hindi and English: {((df['hindi_word_count'] > 0) & (df['english_word_count'] > 0)).sum()}")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()