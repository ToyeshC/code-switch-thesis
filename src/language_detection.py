import pandas as pd
import fasttext
import argparse
import os
from tqdm import tqdm
import re
import nltk
from nltk.corpus import words as nltk_words

def main():
    """
    Process a CSV file containing generated sentences, detect languages using FastText,
    and count Hindi and English words in each sentence with improved detection for Romanized Hindi.
    Then filter out sentences that don't meet code-switching criteria.
    """
    parser = argparse.ArgumentParser(
        description="Detect languages and count Hindi/English words in generated sentences with Romanized Hindi support"
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
        "--filtered_output_file",
        type=str,
        default=None,
        help="Path to save the filtered CSV file (defaults to input_file_filtered.csv)",
    )
    parser.add_argument(
        "--fasttext_model",
        type=str,
        default="lid.176.bin",
        help="Path to the FastText language identification model",
    )
    parser.add_argument(
        "--hindi_words_file",
        type=str,
        default=None,
        help="Path to a file containing common Romanized Hindi words (one per line)",
    )
    parser.add_argument(
        "--romanized_threshold",
        type=float,
        default=0.45,
        help="Threshold for romanized Hindi percentage (default: 0.45)",
    )
    args = parser.parse_args()

    # Set default output files if not provided
    if args.output_file is None:
        base_name = os.path.splitext(args.input_file)[0]
        args.output_file = f"{base_name}_processed.csv"
    
    if args.filtered_output_file is None:
        base_name = os.path.splitext(args.input_file)[0]
        args.filtered_output_file = f"{base_name}_filtered.csv"

    # Load the FastText language identification model
    print(f"Loading FastText model from {args.fasttext_model}...")
    model = fasttext.load_model(args.fasttext_model)

    # Load NLTK English words for better English detection
    try:
        english_words = set(nltk_words.words())
        print(f"Loaded {len(english_words)} English words from NLTK")
    except LookupError:
        print("Downloading NLTK words corpus...")
        nltk.download('words')
        english_words = set(nltk_words.words())

    # Common Romanized Hindi words and patterns
    common_hindi_words = {
        # Common Hindi words in Roman script
        'main', 'mein', 'hum', 'tum', 'aap', 'yeh', 'woh', 'kya', 'kyun', 'kaise',
        'hai', 'hain', 'tha', 'the', 'thi', 'thin', 'ho', 'hoga', 'hogi', 'honge',
        'kar', 'karo', 'karna', 'karenge', 'kiya', 'kiye', 'ki', 'ka', 'ke', 'ko',
        'se', 'par', 'mein', 'aur', 'ya', 'ek', 'do', 'teen', 'char', 'paanch',
        'bahut', 'thoda', 'accha', 'bura', 'nahi', 'nahin', 'mat', 'na', 'haan', 'ji',
        'bhai', 'behen', 'beta', 'beti', 'maa', 'papa', 'dada', 'dadi', 'nana', 'nani',
        'dost', 'pyaar', 'mohabbat', 'ishq', 'zindagi', 'khushi', 'gham', 'dukh', 'dard',
        'khana', 'pani', 'chai', 'doodh', 'roti', 'sabzi', 'dal', 'chawal', 'meetha',
        'ghar', 'bahar', 'andar', 'upar', 'neeche', 'aage', 'peeche', 'din', 'raat',
        'subah', 'shaam', 'dopahar', 'kal', 'aaj', 'abhi', 'kabhi', 'hamesha', 'kabhi',
        'jab', 'tab', 'phir', 'lekin', 'magar', 'kyunki', 'isliye', 'toh', 'bhi',
        'sirf', 'bas', 'bilkul', 'ekdum', 'zaroor', 'shayad', 'matlab', 'samajh',
        'dekho', 'suno', 'bolo', 'kaho', 'jao', 'aao', 'ruko', 'chalo', 'karo',
        'apna', 'mera', 'tera', 'uska', 'hamara', 'tumhara', 'unka', 'sabka',
        'kuch', 'sab', 'koi', 'kaun', 'kitna', 'kitne', 'kitni', 'kaisa', 'jaisa',
        'waisa', 'jitna', 'utna', 'bohot', 'zyada', 'kam', 'jyada', 'adhik',
        'pyaara', 'achha', 'bura', 'saaf', 'ganda', 'lamba', 'chota', 'bada',
        'naya', 'purana', 'geela', 'sukha', 'garam', 'thanda', 'meetha', 'teekha',
        'namaste', 'shukriya', 'dhanyavaad', 'alvida', 'phir milenge', 'kaise ho',
        'theek', 'hoon', 'hu', 'ho', 'hai', 'hain', 'tha', 'thi', 'the', 'thi',
        'raha', 'rahi', 'rahe', 'rahenge', 'rahegi', 'rahega', 'gaya', 'gayi', 'gaye',
        'aaya', 'aayi', 'aaye', 'jaayega', 'jaayegi', 'jaayenge', 'karta', 'karti', 'karte',
        'karegi', 'karega', 'karenge',
        'haan', 'nahin', 'nahi', 'bilkul', 'zaroor', 'kabhi', 'humesha', 'kabhi', 'nahi',
        'kya', 'kyun', 'kaise', 'kahan', 'kab', 'kaun', 'kitna', 'kitni', 'kitne',
        'mujhe', 'tumhe', 'use', 'hume', 'aapko', 'unhe', 'inhe', 'mujhko', 'tumko',
        'usko', 'humko', 'aapko', 'unko', 'inko', 'mujhse', 'tumse', 'usse', 'humse',
        'aapse', 'unse', 'inse', 'mera', 'tera', 'uska', 'hamara', 'tumhara', 'aapka',
        'unka', 'inka', 'mere', 'tere', 'uske', 'hamare', 'tumhare', 'aapke', 'unke', 'inke',
        'yaar', 'bhai', 'dost', 'saathi', 'mitr', 'premi', 'premika', 'pati', 'patni',
        'beta', 'beti', 'bacha', 'bachi', 'ladka', 'ladki', 'aadmi', 'aurat', 'log',
        'samay', 'waqt', 'din', 'raat', 'subah', 'shaam', 'dopahar', 'savera', 'sandhya',
        'kal', 'aaj', 'parson', 'abhi', 'tab', 'jab', 'pehle', 'baad', 'phir',
        'lekin', 'magar', 'parantu', 'kintu', 'aur', 'tatha', 'evam', 'ya', 'athava',
        'agar', 'yadi', 'to', 'tab', 'tabhi', 'jabhi', 'kyunki', 'isliye', 'chunki',
        'haan', 'ji', 'accha', 'theek', 'bilkul', 'zaroor', 'avashya', 'pakka',
        'dhanyavad', 'shukriya', 'namaste', 'pranam', 'ram ram', 'jai shri krishna',
        'khuda hafiz', 'allah hafiz', 'sat sri akal', 'radhe radhe', 'jai mata di',
        'shubh', 'mangal', 'kalyan', 'khushi', 'anand', 'sukh', 'dukh', 'kasht', 'peeda',
        'pyaar', 'prem', 'mohabbat', 'ishq', 'dosti', 'yaari', 'mitrata', 'rishtedaari',
        'parivaar', 'ghar', 'makan', 'haveli', 'mahal', 'jhopdi', 'kholi', 'kamra',
        'khana', 'bhojan', 'roti', 'chawal', 'dal', 'sabzi', 'doodh', 'pani', 'jal',
        'chai', 'coffee', 'paani', 'doodh', 'lassi', 'sharbat', 'ras', 'juice',
        'kitaab', 'pustak', 'kalam', 'pen', 'pencil', 'kagaz', 'paper', 'daftar',
        'vidyalaya', 'school', 'college', 'vishwavidyalaya', 'university', 'shiksha',
        'padhai', 'likhna', 'padhna', 'bolna', 'sunna', 'dekhna',
        'karna', 'hona', 'jana', 'aana', 'khana', 'peena', 'sona', 'uthna',
        'baithna', 'khelna', 'hasna', 'rona', 'muskurana', 'nachna', 'gaana',
        'paisa', 'rupaya', 'dhan', 'sampatti', 'ameer', 'gareeb', 'dhani', 'nirdhan',
        'safal', 'asafal', 'jeet', 'haar', 'vijay', 'paraajay', 'achha', 'bura',
        'sundar', 'khoobsurat', 'badsooorat', 'badshakal', 'lamba', 'chota', 'mota', 'patla',
        'bada', 'chhota', 'ooncha', 'neecha', 'geela', 'sukha', 'garam', 'thanda',
        'naya', 'purana', 'taaza', 'baasi', 'saaf', 'ganda', 'swachh', 'maila',
        'meetha', 'namkeen', 'teekha', 'kadwa', 'khatta', 'chatpata', 'masaledar',
        'tez', 'dhima', 'tej', 'mand', 'ucch', 'nimn', 'uttam', 'adhik', 'kam',
        'zyada', 'bahut', 'thoda', 'jyada', 'kum', 'adhik', 'alp', 'vishesh',
        'vishishth', 'khaas', 'aam', 'sadhaaran', 'asaadhaaran', 'alag', 'vibhinn',
        'ek', 'do', 'teen', 'char', 'paanch', 'chhe', 'saat', 'aath', 'nau', 'das',
        'gyarah', 'barah', 'terah', 'chaudah', 'pandrah', 'solah', 'satrah', 'atharah',
        'unnees', 'bees', 'ikkees', 'baees', 'teis', 'chaubees', 'pachees', 'chhabbees',
        'sattaees', 'atthaees', 'untees', 'tees', 'ektees', 'battees', 'taintees',
        'chautees', 'paintees', 'chattees', 'saintees', 'adtees', 'untaalis', 'chalis',
        'iktalis', 'bayalis', 'taintalis', 'chaualis', 'paintalis', 'chiyalis', 'saintalis',
        'adtalis', 'unchaas', 'pachaas', 'ikyavan', 'bavan', 'tirpan', 'chauvan', 'pachpan',
        'chhappan', 'sattavan', 'athavan', 'unsath', 'saath', 'iksath', 'basath', 'tirsath',
        'chausath', 'painsath', 'chhiyasath', 'sadsath', 'adsath', 'unhattar', 'sattar',
        'ikhattar', 'bahattar', 'tihattar', 'chauhattar', 'pachhattar', 'chhihattar',
        'sathattar', 'athattar', 'unaasi', 'assi', 'ikyaasi', 'bayaasi', 'tiraasi',
        'chauraasi', 'pachaasi', 'chhiyaasi', 'sataasi', 'athaasi', 'navaasi', 'nabbe',
        'baanve', 'tiraanve', 'chauraanve', 'pachhaanve', 'chhiyaanve',
        'sattaanve', 'atthaanve', 'ninyaanve', 'sau'
    }

    # Load additional Hindi words from file if provided
    if args.hindi_words_file and os.path.exists(args.hindi_words_file):
        print(f"Loading additional Hindi words from {args.hindi_words_file}...")
        with open(args.hindi_words_file, 'r', encoding='utf-8') as f:
            additional_words = {line.strip().lower() for line in f if line.strip()}
            common_hindi_words.update(additional_words)
        print(f"Added {len(additional_words)} words from file")

    # Patterns for identifying Romanized Hindi
    hindi_patterns = [
        r'\b(kya|kyun|kaise|kaun|kitna|kahan)\b',  # Question words
        r'\b(hai|hain|tha|thi|ho|hoga|hogi)\b',    # Forms of "to be"
        r'\b(ka|ki|ke|ko|se|mein|par)\b',          # Common postpositions
        r'\b(aur|ya|lekin|magar|kyunki|isliye)\b', # Conjunctions
        r'\b(nahi|nahin|mat|na)\b',                # Negations
        r'\b(bahut|thoda|zyada|kam)\b',            # Quantifiers
        r'\b(main|mein|hum|tum|aap|yeh|woh)\b',    # Pronouns
        r'\b(karo|karenge|kiya|kiye|kar|karna)\b', # Forms of "to do"
        r'\b(gaya|gayi|gaye|jao|jaana)\b',         # Forms of "to go"
        r'\b(aaya|aayi|aaye|aao|aana)\b',          # Forms of "to come"
        r'\b(raha|rahi|rahe|rehna)\b',             # Forms of "to stay"
        r'\b(wala|wali|wale)\b',                   # Possessive markers
        r'\b(accha|achha|bura|theek)\b',           # Common adjectives
    ]

    # Compile patterns for faster matching
    hindi_pattern_regex = re.compile('|'.join(hindi_patterns), re.IGNORECASE)

    # Read the input CSV file
    print(f"Reading input file: {args.input_file}")
    df = pd.read_csv(args.input_file)
    
    # Initialize new columns for word counts
    df['hindi_word_count'] = 0
    df['english_word_count'] = 0
    df['romanized_hindi_count'] = 0
    
    # Process each sentence
    print("Processing sentences and counting words by language...")
    
    # Function to process each sentence
    def process_sentence(sentence):
        if not isinstance(sentence, str) or pd.isna(sentence):
            return 0, 0, 0  # Return zeros for non-string or NaN values
        
        # Tokenize the sentence into words
        words = re.findall(r'\b\w+\b', sentence.lower())
        
        # Count Hindi, English, and Romanized Hindi words
        hindi_count = 0
        english_count = 0
        romanized_hindi_count = 0
        
        for word in words:
            word_lower = word.lower()
            
            # Check if it's a common Romanized Hindi word
            if word_lower in common_hindi_words or hindi_pattern_regex.search(word_lower):
                romanized_hindi_count += 1
                continue
                
            # For remaining words, use FastText for language detection
            prediction = model.predict(word, k=1)
            lang = prediction[0][0].replace('__label__', '')
            
            # Count words by language
            if lang == 'hi':
                hindi_count += 1
            # Only count as English if it's in the English dictionary or very likely English
            elif lang == 'en' and (word_lower in english_words or len(word) > 3):
                english_count += 1
            # Words not clearly identified might be Romanized Hindi
            elif len(word) > 2:  # Ignore very short words
                # Check if it follows Hindi word patterns (e.g., ending with common suffixes)
                if (word_lower.endswith(('na', 'ne', 'ni', 'ta', 'ti', 'te', 'ya', 'ye', 'yi', 
                                        'kar', 'wala', 'wali', 'gaya', 'gayi', 'raha', 'rahi')) or
                    any(pattern in word_lower for pattern in ('aa', 'ee', 'oo', 'kh', 'gh', 'ch', 'jh', 'th'))):
                    romanized_hindi_count += 1
                else:
                    english_count += 1
            
        return hindi_count, english_count, romanized_hindi_count
    
    # Apply the processing function to the appropriate column
    text_column = None
    if 'generated' in df.columns:
        text_column = 'generated'
    else:
        print("Warning: 'generated' column not found. Looking for text columns...")
        # Try to find a column that might contain sentences
        text_columns = [col for col in df.columns if df[col].dtype == 'object']
        if text_columns:
            print(f"Found potential text columns: {text_columns}")
            text_column = text_columns[0]
            print(f"Processing the first text column: {text_column}")
    
    if text_column:
        # Use tqdm for progress tracking
        for i in tqdm(range(len(df))):
            hindi_count, english_count, romanized_hindi_count = process_sentence(df.loc[i, text_column])
            df.loc[i, 'hindi_word_count'] = hindi_count
            df.loc[i, 'english_word_count'] = english_count
            df.loc[i, 'romanized_hindi_count'] = romanized_hindi_count
    else:
        print("Error: No suitable text column found for processing.")
        return
    
    # Calculate total Hindi count (Devanagari + Romanized)
    df['total_hindi_count'] = df['hindi_word_count'] + df['romanized_hindi_count']
    
    # Calculate percentages for filtering
    df['total_words'] = df['hindi_word_count'] + df['romanized_hindi_count'] + df['english_word_count']
    df['romanized_hindi_percent'] = df.apply(
        lambda row: (row['romanized_hindi_count'] / row['english_word_count'] * 100) 
        if row['english_word_count'] > 0 else 0, 
        axis=1
    )
    
    # Save the processed DataFrame to a new CSV file
    print(f"Saving processed data to {args.output_file}")
    df.to_csv(args.output_file, index=False)
    
    # Filter sentences based on criteria
    print("Filtering sentences based on code-switching criteria...")
    
    # Create a filter mask based on the criteria
    filter_mask = (
        # Keep sentences with both Hindi and English
        ((df['hindi_word_count'] > 0) & (df['english_word_count'] > 0)) |
        
        # Keep sentences with Romanized Hindi and English if romanized percentage is high enough
        ((df['hindi_word_count'] == 0) & 
         (df['romanized_hindi_count'] > 0) & 
         (df['english_word_count'] > 0) & 
         (df['romanized_hindi_percent'] >= args.romanized_threshold * 100))
    )
    
    # Apply the filter
    filtered_df = df[filter_mask].reset_index(drop=True)
    
    # Save the filtered DataFrame
    print(f"Saving filtered data to {args.filtered_output_file}")
    filtered_df.to_csv(args.filtered_output_file, index=False)
    
    # Print summary statistics
    print("\n===== Summary Statistics for All Sentences =====")
    print(f"Total sentences processed: {len(df)}")
    print(f"Average Devanagari Hindi words per sentence: {df['hindi_word_count'].mean():.2f}")
    print(f"Average Romanized Hindi words per sentence: {df['romanized_hindi_count'].mean():.2f}")
    print(f"Average Total Hindi words per sentence: {df['total_hindi_count'].mean():.2f}")
    print(f"Average English words per sentence: {df['english_word_count'].mean():.2f}")
    print(f"Sentences with both Hindi and English: {((df['hindi_word_count'] > 0) & (df['english_word_count'] > 0)).sum()}")
    print(f"Sentences with Romanized Hindi and English: {((df['hindi_word_count'] == 0) & (df['romanized_hindi_count'] > 0) & (df['english_word_count'] > 0)).sum()}")
    
    print("\n===== Summary Statistics for Filtered Sentences =====")
    print(f"Total sentences after filtering: {len(filtered_df)}")
    print(f"Sentences removed: {len(df) - len(filtered_df)} ({(len(df) - len(filtered_df)) / len(df) * 100:.2f}%)")
    print(f"Average Devanagari Hindi words per sentence: {filtered_df['hindi_word_count'].mean():.2f}")
    print(f"Average Romanized Hindi words per sentence: {filtered_df['romanized_hindi_count'].mean():.2f}")
    print(f"Average Total Hindi words per sentence: {filtered_df['total_hindi_count'].mean():.2f}")
    print(f"Average English words per sentence: {filtered_df['english_word_count'].mean():.2f}")
    
    # Breakdown of filtered sentences
    only_devanagari = ((filtered_df['hindi_word_count'] > 0) & (filtered_df['romanized_hindi_count'] == 0) & (filtered_df['english_word_count'] > 0)).sum()
    only_romanized = ((filtered_df['hindi_word_count'] == 0) & (filtered_df['romanized_hindi_count'] > 0) & (filtered_df['english_word_count'] > 0)).sum()
    both_hindi_types = ((filtered_df['hindi_word_count'] > 0) & (filtered_df['romanized_hindi_count'] > 0) & (filtered_df['english_word_count'] > 0)).sum()
    
    print("\n===== Breakdown of Filtered Sentences =====")
    print(f"Sentences with Devanagari Hindi + English (no Romanized): {only_devanagari} ({only_devanagari/len(filtered_df)*100:.2f}%)")
    print(f"Sentences with Romanized Hindi + English (no Devanagari): {only_romanized} ({only_romanized/len(filtered_df)*100:.2f}%)")
    print(f"Sentences with both Hindi types + English: {both_hindi_types} ({both_hindi_types/len(filtered_df)*100:.2f}%)")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()