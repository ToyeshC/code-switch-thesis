import argparse
import csv
import numpy as np
import pandas as pd
import spacy
import langid
from collections import Counter
import re
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def preprocess_text(text):
    """
    Preprocess text by:
    1. Converting to lowercase
    2. Removing special characters but keeping spaces between words
    3. Tokenizing by whitespace
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Tokenize by whitespace and remove empty tokens
    tokens = [token for token in text.split() if token.strip()]
    
    return tokens

def language_classification(tokens):
    """
    Use langid to classify the language of each token.
    If the detected language is neither 'hi' (Hindi) nor 'en' (English),
    mark it as 'other'.
    """
    langs = []
    for token in tokens:
        lang, _ = langid.classify(token)
        if lang not in ['hi', 'en']:
            langs.append('other')
        else:
            langs.append(lang)
    return langs

def compute_cmi(token_langs):
    """
    Compute the Code-Mixing Index (CMI) using the formula:
    CMI = (1 - max(M, E, O) / T) * 100,
    where M = # Hindi words, E = # English words, O = # Other words,
    and T = total number of words.
    """
    total = len(token_langs)
    counts = Counter(token_langs)
    m = counts.get('hi', 0)
    e = counts.get('en', 0)
    o = counts.get('other', 0)
    if total > 0:
        cmi = (1 - max(m, e, o) / total) * 100
    else:
        cmi = 0
    return {'hi': m, 'en': e, 'other': o, 'cmi': cmi}

def compute_switch_points(langs):
    """
    Compute the number of language switch points in the token sequence.
    """
    if not langs:
        return 0
    switches = sum(1 for i in range(1, len(langs)) if langs[i] != langs[i - 1])
    return switches

def compute_m_index(token_langs):
    """
    Compute the M-Index (Multilingual Index) which measures language diversity.
    M-Index = 1 - Σ(p_i^2), where p_i is the proportion of words in language i.
    """
    total = len(token_langs)
    if total == 0:
        return 0
    
    counts = Counter(token_langs)
    proportions = [count/total for count in counts.values()]
    m_index = 1 - sum(p**2 for p in proportions)
    return m_index

def integration_index(hi_count, en_count):
    """
    Compute the Integration Index (I-Index) which measures how well
    the two languages are integrated.
    I-Index = min(hi_count, en_count) / max(hi_count, en_count) if max > 0 else 0
    """
    if max(hi_count, en_count) > 0:
        return min(hi_count, en_count) / max(hi_count, en_count)
    return 0

def remove_zero_switch_points(df):
    """
    Remove rows where either Hindi or English token count is 0.
    Only keep sentences that have both Hindi and English tokens.
    """
    return df[(df['hindi'] > 0) & (df['en'] > 0)].reset_index(drop=True)

# def pos_tagging(text, nlp):
#     """
#     Perform POS tagging on the text using spaCy.
#     Returns a string representation of token-POS pairs.
#     """
#     doc = nlp(text)
#     pos_tags = [(token.text, token.pos_) for token in doc]
#     return str(pos_tags)

def main():
    parser = argparse.ArgumentParser(
        description="Code-Switched Sentence Evaluation Pipeline"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the file containing code-switched sentences (one sentence per line)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.csv",
        help="File to write per-sentence evaluation metrics",
    )
    args = parser.parse_args()

    # Load spaCy multilingual model for POS tagging - commented out as requested
    # try:
    #     nlp = spacy.load("xx_ent_wiki_sm")
    # except Exception as e:
    #     print("Error loading spaCy model. Please run:")
    #     print("    python -m spacy download xx_ent_wiki_sm")
    #     return

    # Read input sentences
    with open(args.input_file, "r", encoding="utf-8") as infile:
        sentences = [line.strip() for line in infile if line.strip()]

    results = []

    # Process each sentence individually.
    for sent in sentences:
        tokens = preprocess_text(sent)
        token_langs = language_classification(tokens)
        metrics = compute_cmi(token_langs)
        switches = compute_switch_points(token_langs)
        metrics["switch_points"] = switches
        metrics["total_tokens"] = len(tokens)
        metrics["m_index"] = compute_m_index(token_langs)
        metrics["integration_index"] = integration_index(metrics["hi"], metrics["en"])
        # Commented out POS tagging
        # pos_tags = pos_tagging(sent, nlp)
        
        result = {
            "sentence": sent,
            "total_tokens": len(tokens),
            "hindi": metrics["hi"],  # Renamed from 'hi' to 'hindi'
            "en": metrics["en"],
            # Renamed 'other' to 'hindi' as per requirement
            # Note: This will overwrite the previous 'hindi' value
            "hindi": metrics["other"],  
            "CMI": metrics["cmi"],
            "m_index": metrics["m_index"],
            "switch_points": switches,
            "integration_index": metrics["integration_index"],
            # Commented out pos_tags
            # "pos_tags": pos_tags,
        }
        results.append(result)

    # Convert results to DataFrame for easier manipulation
    results_df = pd.DataFrame(results)
    
    # Remove rows with zero switch points
    filtered_df = remove_zero_switch_points(results_df)
    
    # Compute overall statistics using the filtered DataFrame
    avg_tokens = filtered_df['total_tokens'].mean()
    avg_cmi = filtered_df['CMI'].mean()
    avg_m_index = filtered_df['m_index'].mean()
    avg_switch_points = filtered_df['switch_points'].mean()
    avg_integration_index = filtered_df['integration_index'].mean()

    print("\n===== Overall Evaluation Metrics =====")
    print(f"Total sentences (after filtering): {len(filtered_df)}")
    print(f"Average tokens per sentence: {avg_tokens:.2f}")
    print(f"Average CMI: {avg_cmi:.2f}")
    print(f"Average M-Index: {avg_m_index:.2f}")
    print(f"Average Switch Points: {avg_switch_points:.2f}")
    print(f"Average Integration Index: {avg_integration_index:.2f}")

    # Save detailed per-sentence metrics to CSV using pandas
    filtered_df.to_csv(args.output, index=False)
    print(f"\nPer-sentence evaluation metrics saved to {args.output}")

if __name__ == "__main__":
    main()
