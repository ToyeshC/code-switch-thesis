import argparse
import pandas as pd
import os
import json
from tqdm import tqdm
import re
import sys

# Add paths to look for the models and modules
# Will be overridden by command line argument if provided
indiclid_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "indic_nlp_library")

# Initialize these variables at the global scope
INDIC_LID_LIBRARY_AVAILABLE = False
INDIC_LID_MODELS_AVAILABLE = False

def setup_indic_nlp(indic_nlp_path=None):
    """
    Set up the Indic NLP library
    
    Args:
        indic_nlp_path (str): Path to the Indic NLP library
        
    Returns:
        tuple: (INDIC_LID_LIBRARY_AVAILABLE, INDIC_LID_MODELS_AVAILABLE)
    """
    global indiclid_path
    
    # Use provided path if available
    if indic_nlp_path and os.path.exists(indic_nlp_path):
        indiclid_path = indic_nlp_path
        print(f"Using provided Indic NLP path: {indiclid_path}")
    
    # Add path to Python path
    if os.path.exists(indiclid_path):
        sys.path.append(indiclid_path)
        print(f"Added {indiclid_path} to Python path")
    
    # Check for IndicLID from AI4Bharat
    indic_lid_library_available = False
    indic_lid_models_available = False
    
    try:
        from indicnlp.langdetect import IndicLangDetect
        indic_lid_library_available = True
        print("Successfully imported indicnlp.langdetect")
        
        # Check if the library can be initialized
        try:
            detector = IndicLangDetect()
            indic_lid_models_available = True
            print("Successfully initialized IndicLangDetect")
        except Exception as e:
            print(f"Failed to initialize IndicLangDetect: {e}")
    except ImportError as e:
        print(f"Warning: indicnlp.langdetect not found: {e}")
        print("Using fallback language detection.")
    
    # If both conditions are not met, use fallback
    if not (indic_lid_library_available and indic_lid_models_available):
        print("Using fallback detection method for Indic languages")
    
    return indic_lid_library_available, indic_lid_models_available

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
    
    is_code_switched = hindi_chars > 0 and english_chars > 0
    
    return {
        'detected_language': detected_lang,
        'hindi_prob': hindi_prob,
        'english_prob': english_prob,
        'is_code_switched': is_code_switched
    }

def main():
    parser = argparse.ArgumentParser(description="Detect languages using Indic LID")
    parser.add_argument('--input_file', required=True, help='Path to the input text file or CSV')
    parser.add_argument('--output_file', required=True, help='Path to the output CSV file')
    parser.add_argument('--id_map', required=False, help='Path to the ID mapping JSON file')
    parser.add_argument('--is_compiled_csv', action='store_true', 
                      help='Whether the input is a compiled CSV with src/tgt/generated columns')
    parser.add_argument('--indic_nlp_path', required=False, help='Path to the Indic NLP library')
    
    args = parser.parse_args()
    
    # Set up Indic NLP library
    indic_lid_library_available, indic_lid_models_available = setup_indic_nlp(args.indic_nlp_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    id_mapping = {}
    if args.id_map:
        # Load the ID mapping
        with open(args.id_map, 'r', encoding='utf-8') as f:
            id_mapping = json.load(f)
        
        # Convert string keys to integers
        id_mapping = {int(k): v for k, v in id_mapping.items()}
    
    lines = []
    prompt_ids = []
    
    # Handle different input types
    if args.is_compiled_csv:
        # Load the compiled CSV file (e.g., data/output/hindi/compile_hindi.csv)
        compiled_df = pd.read_csv(args.input_file)
        # Use the 'generated' column as our text
        lines = compiled_df['generated'].tolist()
        # Use index as prompt_id if no mapping is provided
        prompt_ids = compiled_df.index.tolist()
        # Save additional metadata
        model_info = compiled_df['model'].tolist() if 'model' in compiled_df.columns else None
        method_info = compiled_df['method'].tolist() if 'method' in compiled_df.columns else None
    else:
        # Read the input file as regular text
        with open(args.input_file, 'r', encoding='utf-8') as f:
            raw_lines = [line.strip() for line in f.readlines()]
        
        # Extract only the code-switched parts from the sentences
        lines = [extract_code_switched_sentences(line) for line in raw_lines]
        prompt_ids = [id_mapping.get(i) for i in range(len(lines))]
        model_info = None
        method_info = None
    
    # Initialize language detector - do not modify the global variables
    detector = None
    using_indic_lid = False  # Local variable for tracking if we're using Indic LID
    
    if indic_lid_library_available and indic_lid_models_available:
        try:
            from indicnlp.langdetect import IndicLangDetect
            detector = IndicLangDetect()
            using_indic_lid = True
            print("Using Indic LID for language detection")
        except Exception as e:
            print(f"Error initializing IndicLangDetect: {e}")
            print("Using fallback detection")
            using_indic_lid = False
    else:
        print("Indic LID not available, using fallback detection")
    
    results = []
    
    # Process each line
    print(f"Detecting languages in {args.input_file}...")
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
        
        # Process the line
        try:
            if using_indic_lid and detector:
                # Get language probabilities using Indic LID
                lang_probs = detector.batch_detect_probs([clean_line])[0]
                detected_lang = detector.detect(clean_line)
                
                # Extract Hindi and English probabilities
                hindi_count = int(lang_probs.get('hi', 0) * 100)
                english_count = int(lang_probs.get('en', 0) * 100)
                total_words = hindi_count + english_count + sum(v for k, v in lang_probs.items() 
                                                               if k not in ['hi', 'en'])
                
                # Calculate percentages
                if total_words > 0:
                    hindi_percent = hindi_count / total_words * 100
                    english_percent = english_count / total_words * 100
                else:
                    hindi_percent = english_percent = 0
                
            else:
                # Use fallback detection
                result = fallback_language_detection(clean_line)
                detected_lang = result['detected_language']
                hindi_count = int(result['hindi_prob'] * 100)
                english_count = int(result['english_prob'] * 100)
                total_words = len(clean_line.split())
                
                # Calculate percentages based on probabilities
                hindi_percent = result['hindi_prob'] * 100
                english_percent = result['english_prob'] * 100
            
            # Determine if the sentence is code-switched
            is_code_switched = hindi_count > 0 and english_count > 0 and hindi_percent > 20 and english_percent > 20
            
            result_entry = {
                'prompt_id': prompt_id if prompt_id is not None else i,
                'sentence': clean_line,
                'hindi_words': hindi_count,
                'english_words': english_count,
                'total_words': total_words,
                'hindi_percent': hindi_percent,
                'english_percent': english_percent,
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
                'sentence': clean_line,
                'hindi_words': 0,
                'english_words': 0,
                'total_words': len(clean_line.split()) if isinstance(clean_line, str) else 0,
                'hindi_percent': 0,
                'english_percent': 0,
                'is_code_switched': False
            })
    
    # Create DataFrame and save results
    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False)
    print(f"Language detection complete. Results saved to {args.output_file}")

if __name__ == '__main__':
    main() 