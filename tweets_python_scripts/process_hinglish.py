import pandas as pd
import re

def process_hinglish_file(input_file, output_file):
    # Lists to store data
    data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
            
        # Process meta line
        if line.startswith('meta'):
            parts = line.split('\t')
            primary_key = parts[1]
            sentiment = parts[2]
            
            # Initialize counters for the sentence
            hindi_words = 0
            english_words = 0
            romanized_hindi_words = 0
            total_words = 0
            sentence_words = []
            
            # Process the sentence
            i += 1
            while i < len(lines) and lines[i].strip():
                line_parts = lines[i].strip().split('\t')
                # Skip lines that don't have the expected format
                if len(line_parts) != 2:
                    i += 1
                    continue
                    
                word, lang = line_parts
                if lang == 'Hin':
                    romanized_hindi_words += 1
                elif lang == 'Eng':
                    english_words += 1
                total_words += 1
                sentence_words.append(word)
                i += 1
            
            # Calculate percentages
            hindi_percent = 0  # Since we can't determine Devanagari Hindi
            romanized_hindi_percent = (romanized_hindi_words / total_words * 100) if total_words > 0 else 0
            total_hindi_percent = romanized_hindi_percent  # Since all Hindi is considered romanized
            english_percent = (english_words / total_words * 100) if total_words > 0 else 0
            
            # Create the generated text (full sentence)
            generated = ' '.join(sentence_words)
            
            # Add to data list
            data.append({
                'generated': generated,
                'primary_key': primary_key,
                'sentiment': sentiment,
                'hindi_word_count': 0,  # Since we can't determine Devanagari Hindi
                'english_word_count': english_words,
                'romanized_hindi_count': romanized_hindi_words,
                'total_hindi_count': romanized_hindi_words,
                'total_words': total_words,
                'hindi_percent': hindi_percent,
                'romanized_hindi_percent': romanized_hindi_percent,
                'total_hindi_percent': total_hindi_percent,
                'english_percent': english_percent
            })
        else:
            i += 1
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    input_file = "data/Semeval_2020_task9_data/Hinglish/Hinglish_train_14k_split_conll.txt"
    output_file = "tweets_outputs/processed_hinglish.csv"
    process_hinglish_file(input_file, output_file) 