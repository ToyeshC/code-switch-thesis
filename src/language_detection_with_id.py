import pandas as pd
import fasttext
import argparse
import os
import json
from tqdm import tqdm
import re
import nltk
from nltk.corpus import words as nltk_words

def extract_code_switched_sentences(text):
    """
    Extract only the code-switched sentences from the input text,
    ignoring metadata, formatting, and other content.
    
    Args:
        text (str): The input text which may contain metadata and other content
        
    Returns:
        str: Only the code-switched sentence part of the text
    """
    # If the text is not a string, return it as is
    if not isinstance(text, str) or not text:
        return text
        
    # Try to identify if the text is in JSON format (contains prompt and completion)
    if text.strip().startswith("{") and ("Prompt" in text or "Completion" in text):
        try:
            # Extract just the completion part which should contain the code-switched text
            completion_match = re.search(r'"Completion":\s*"([^"]+)"', text)
            if completion_match:
                return completion_match.group(1).strip()
        except:
            pass
    
    # If we have a comma-separated line (likely from CSV), try to extract just the sentence
    if "," in text and text.count(",") > 3:  # Likely a CSV line with multiple fields
        # Try to find text that's not surrounded by quotes and metadata
        # This is a heuristic approach - may need adjustment based on your specific format
        parts = text.split('","')
        if len(parts) > 1:
            # Look for the longest part that's likely to be the actual sentence
            parts = [p.strip('"') for p in parts]
            parts = [p for p in parts if len(p) > 10 and not p.startswith("{") and not p.endswith("}")]
            if parts:
                return max(parts, key=len)
    
    # Remove any metadata-like patterns (key-value pairs, JSON fragments)
    text = re.sub(r'"\w+":\s*("[^"]*"|[\d\.]+|\[[^\]]*\]|{[^}]*})', ' ', text)
    
    # Remove excessive punctuation and formatting
    text = re.sub(r'["""",]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def main():
    """
    Process a text file containing sentences, detect languages using FastText,
    and count Hindi and English words in each sentence with improved detection for Romanized Hindi.
    Preserves the prompt IDs from the ID mapping file.
    """
    parser = argparse.ArgumentParser(
        description="Detect languages and count Hindi/English words in code-switched sentences with Romanized Hindi support"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the text file containing code-switched sentences",
    )
    parser.add_argument(
        "--id_map",
        type=str,
        required=False,
        help="Path to the JSON file containing the ID mapping",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the processed CSV file",
    )
    parser.add_argument(
        "--fasttext_model",
        type=str,
        default="lid.176.bin",
        help="Path to the FastText language identification model",
    )
    parser.add_argument(
        "--is_compiled_csv",
        action="store_true",
        help="Whether the input is a compiled CSV with src/tgt/generated columns",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    id_mapping = {}
    if args.id_map:
        # Load the ID mapping
        with open(args.id_map, 'r', encoding='utf-8') as f:
            id_mapping = json.load(f)
        
        # Convert string keys to integers
        id_mapping = {int(k): v for k, v in id_mapping.items()}
    
    # Read the input file
    sentences = []
    prompt_ids = []
    model_info = None
    method_info = None
    
    if args.is_compiled_csv:
        # Load the compiled CSV file (e.g., data/output/hindi/compile_hindi.csv)
        compiled_df = pd.read_csv(args.input_file)
        # Use the 'generated' column as our text
        sentences = compiled_df['generated'].tolist()
        # Use index as prompt_id if no mapping is provided
        prompt_ids = compiled_df.index.tolist()
        # Save additional metadata
        model_info = compiled_df['model'].tolist() if 'model' in compiled_df.columns else None
        method_info = compiled_df['method'].tolist() if 'method' in compiled_df.columns else None
    else:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            raw_sentences = [line.strip() for line in f.readlines()]
        
        # Extract only the code-switched parts from the sentences
        sentences = [extract_code_switched_sentences(sentence) for sentence in raw_sentences]
        prompt_ids = [id_mapping.get(i) for i in range(len(sentences))]
    
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

    # Common Romanized Hindi words - extensive dictionary
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
        'karegi', 'karega', 'karenge', 'karne', 'kiya', 'kitna', 'kitni', 'kitne',
        'kyunki', 'isliye', 'isiliye', 'phir', 'phirse', 'jaise', 'jaisa', 'jab', 'tab',
        'agar', 'magar', 'lekin', 'toh', 'to', 'acha', 'achha', 'bas', 'sirf',
        'kafi', 'bahut', 'bohat', 'thoda', 'thodi', 'pura', 'poora', 'saara', 'sabhi',
        'mujhe', 'tumhe', 'unhe', 'humein', 'apne', 'mere', 'tere', 'uske', 'hamare',
        'tumhare', 'unke', 'kiske', 'jiske', 'sabhi'
    }

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

    # Function to process each sentence
    def process_sentence(sentence, index):
        if not isinstance(sentence, str) or not sentence:
            return None
        
        prompt_id = prompt_ids[index] if index < len(prompt_ids) else None
        if prompt_id is None and id_mapping:
            prompt_id = id_mapping.get(index)
        
        if prompt_id is None:
            prompt_id = index
        
        # Clean the sentence to remove any newlines
        sentence = sentence.replace("\n", " ").replace("\r", " ")
        
        # Tokenize the sentence into words
        words = sentence.split()
        
        # Count Hindi and English words
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
            try:
                clean_word = word_lower.replace("\n", " ").replace("\r", " ")
                prediction = model.predict(clean_word, k=1)
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
            except Exception as e:
                # If error occurs in language detection, just skip the word
                continue
        
        total_words = len(words)
        total_hindi = hindi_count + romanized_hindi_count
        
        # Calculate percentages
        hindi_percent = (total_hindi / total_words * 100) if total_words > 0 else 0
        english_percent = (english_count / total_words * 100) if total_words > 0 else 0
        
        # Determine if sentence is code-switched
        is_code_switched = (total_hindi > 0 and english_count > 0 and 
                           hindi_percent >= 20 and english_percent >= 20)
        
        # Return results
        result_entry = {
            'prompt_id': prompt_id,
            'sentence': sentence,
            'hindi_words': hindi_count,
            'romanized_hindi_words': romanized_hindi_count,
            'english_words': english_count,
            'total_words': total_words,
            'hindi_percent': hindi_percent,
            'english_percent': english_percent,
            'is_code_switched': is_code_switched
        }
        
        # Add model and method info if available
        if model_info and index < len(model_info):
            result_entry['model'] = model_info[index]
        if method_info and index < len(method_info):
            result_entry['method'] = method_info[index]
            
        return result_entry
    
    # Process sentences
    results = []
    for i, sentence in enumerate(tqdm(sentences)):
        result = process_sentence(sentence, i)
        if result:
            results.append(result)
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False)
    print(f"Language detection complete. Results saved to {args.output_file}")

if __name__ == "__main__":
    main() 