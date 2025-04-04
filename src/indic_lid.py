import argparse
import pandas as pd
import os
import json
from tqdm import tqdm
import re

try:
    from indicnlp.langdetect import IndicLangDetect
    INDIC_LID_AVAILABLE = True
except ImportError:
    print("Warning: indicnlp.langdetect not found. Using fallback language detection.")
    INDIC_LID_AVAILABLE = False

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

def fallback_language_detection(text):
    """
    A simple fallback language detection when Indic LID is not available.
    
    Args:
        text (str): The text to detect language for
        
    Returns:
        dict: Language detection results
    """
    # Clean input text
    text = text.replace("\n", " ").replace("\r", " ")
    
    # Simple pattern matching for Hindi vs English
    # Count Hindi characters vs English characters
    hindi_pattern = re.compile(r'[\u0900-\u097F]')  # Unicode range for Devanagari
    english_pattern = re.compile(r'[a-zA-Z]')
    
    hindi_chars = len(re.findall(hindi_pattern, text))
    english_chars = len(re.findall(english_pattern, text))
    total_chars = len(text)
    
    # Determine language based on character counts
    if hindi_chars > english_chars:
        detected_lang = 'hi'
        hindi_prob = min(1.0, hindi_chars / (total_chars * 0.7) if total_chars > 0 else 0)
        english_prob = min(1.0, english_chars / (total_chars * 0.7) if total_chars > 0 else 0)
    else:
        detected_lang = 'en'
        english_prob = min(1.0, english_chars / (total_chars * 0.7) if total_chars > 0 else 0)
        hindi_prob = min(1.0, hindi_chars / (total_chars * 0.7) if total_chars > 0 else 0)
    
    is_code_switched = hindi_chars > 0 and english_chars > 0 and hindi_prob > 0.2 and english_prob > 0.2
    
    return {
        'detected_language': detected_lang,
        'hindi_prob': hindi_prob,
        'english_prob': english_prob,
        'is_code_switched': is_code_switched
    }

def detect_languages_with_indic(input_file, output_file, id_map_file=None, is_compiled_csv=False):
    """
    Detect languages in a file using Indic LID.
    
    Args:
        input_file (str): Path to the input text file or CSV file
        output_file (str): Path to the output CSV file with language detection
        id_map_file (str, optional): Path to the ID mapping JSON file
        is_compiled_csv (bool): Whether the input is a compiled CSV with 'generated' column
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    id_mapping = {}
    if id_map_file:
        # Load the ID mapping
        with open(id_map_file, 'r', encoding='utf-8') as f:
            id_mapping = json.load(f)
        
        # Convert string keys to integers
        id_mapping = {int(k): v for k, v in id_mapping.items()}
    
    lines = []
    prompt_ids = []
    
    # Handle different input types
    if is_compiled_csv:
        # Load the compiled CSV file (e.g., data/output/hindi/compile_hindi.csv)
        compiled_df = pd.read_csv(input_file)
        # Use the 'generated' column as our text
        lines = compiled_df['generated'].tolist()
        # Use index as prompt_id if no mapping is provided
        prompt_ids = compiled_df.index.tolist()
        # Save additional metadata
        model_info = compiled_df['model'].tolist() if 'model' in compiled_df.columns else None
        method_info = compiled_df['method'].tolist() if 'method' in compiled_df.columns else None
    else:
        # Read the input file as regular text
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_lines = [line.strip() for line in f.readlines()]
        
        # Extract only the code-switched parts from the sentences
        lines = [extract_code_switched_sentences(line) for line in raw_lines]
        prompt_ids = [id_mapping.get(i) for i in range(len(lines))]
        model_info = None
        method_info = None
    
    # Initialize language detector
    if INDIC_LID_AVAILABLE:
        detector = IndicLangDetect()
        print("Using Indic LID for language detection")
    else:
        print("Indic LID not available, using fallback detection")
    
    results = []
    
    # Process each line
    print(f"Detecting languages in {input_file}...")
    for i, line in enumerate(tqdm(lines)):
        if not isinstance(line, str) or not line:
            continue
            
        # Get the original prompt ID
        prompt_id = prompt_ids[i] if i < len(prompt_ids) else None
        if prompt_id is None and id_mapping:
            prompt_id = id_mapping.get(i)
            if prompt_id is None:
                print(f"Warning: No prompt ID found for line {i}")
                continue
        
        # Clean the text to remove newlines
        clean_line = line.replace("\n", " ").replace("\r", " ")
        
        # Detect language
        try:
            if INDIC_LID_AVAILABLE:
                detected_lang = detector.detect(clean_line)
                confidence = detector.detect_prob(clean_line)
                lang_probs = detector.batch_detect_probs([clean_line])[0]
                
                # Extract probabilities for Hindi and English
                hindi_prob = lang_probs.get('hi', 0)
                english_prob = lang_probs.get('en', 0)
                
                # Calculate language distribution
                is_code_switched = False
                if hindi_prob > 0.2 and english_prob > 0.2:
                    is_code_switched = True
            else:
                # Use fallback detection
                result = fallback_language_detection(clean_line)
                detected_lang = result['detected_language']
                hindi_prob = result['hindi_prob']
                english_prob = result['english_prob']
                is_code_switched = result['is_code_switched']
            
            result_entry = {
                'prompt_id': prompt_id if prompt_id is not None else i,
                'text': clean_line,
                'detected_language': detected_lang,
                'hindi_prob': hindi_prob,
                'english_prob': english_prob,
                'total_words': len(clean_line.split()),
                'is_code_switched': is_code_switched
            }
            
            # Add model and method info if available
            if model_info and i < len(model_info):
                result_entry['model'] = model_info[i]
            if method_info and i < len(method_info):
                result_entry['method'] = method_info[i]
                
            results.append(result_entry)
            
        except Exception as e:
            print(f"Error processing line {i}: {e}")
            print(f"Problematic line: '{line}'")
            # Add a placeholder entry
            results.append({
                'prompt_id': prompt_id if prompt_id is not None else i,
                'text': clean_line,
                'detected_language': 'unknown',
                'hindi_prob': 0,
                'english_prob': 0,
                'total_words': len(clean_line.split()) if isinstance(clean_line, str) else 0,
                'is_code_switched': False
            })
    
    # Create DataFrame and save results
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Language detection complete. Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Detect languages using Indic LID")
    parser.add_argument('--input', required=True, help='Path to the input text file or CSV')
    parser.add_argument('--output', required=True, help='Path to the output CSV file')
    parser.add_argument('--id_map', required=False, help='Path to the ID mapping JSON file')
    parser.add_argument('--is_compiled_csv', action='store_true', 
                      help='Whether the input is a compiled CSV with src/tgt/generated columns')
    
    args = parser.parse_args()
    detect_languages_with_indic(args.input, args.output, args.id_map, args.is_compiled_csv)

if __name__ == '__main__':
    main() 