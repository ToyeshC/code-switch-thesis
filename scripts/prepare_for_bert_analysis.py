#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare data for BERT toxicity analysis by categorizing it into monolingual and code-switched samples.
Ensures only filtered IDs are used and adds language category information.
"""

import os
import argparse
import pandas as pd
import numpy as np

# Set up command line arguments
parser = argparse.ArgumentParser(description='Prepare data for BERT toxicity analysis')
parser.add_argument('--toxicity_file', type=str, required=True, help='CSV file with toxicity scores')
parser.add_argument('--language_file', type=str, required=True, help='CSV file with language detection results')
parser.add_argument('--filtered_file', type=str, required=True, help='CSV file with filtered IDs')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save prepared data')
args = parser.parse_args()

def categorize_language_mix(row):
    """Categorize text as monolingual or code-switched based on language composition."""
    total_words = row['total_words']
    if total_words == 0:
        return 'unknown'
    
    hindi_percent = (row['hindi_words'] / total_words) * 100
    english_percent = (row['english_words'] / total_words) * 100
    
    if hindi_percent > 90:
        return 'monolingual_hindi'
    elif english_percent > 90:
        return 'monolingual_english'
    else:
        # Code-switched with meaningful mix
        if hindi_percent >= 30 and english_percent >= 30:
            return 'code_switched_balanced'
        elif hindi_percent > english_percent:
            return 'code_switched_hindi_dominant'
        else:
            return 'code_switched_english_dominant'

def main():
    print(f"Loading data files...")
    
    # Load datasets
    toxicity_df = pd.read_csv(args.toxicity_file)
    language_df = pd.read_csv(args.language_file)
    filtered_df = pd.read_csv(args.filtered_file)
    
    # Get filtered IDs
    filtered_ids = set(filtered_df['id'])
    print(f"Found {len(filtered_ids)} filtered IDs")
    
    # Filter datasets to include only filtered IDs
    toxicity_df = toxicity_df[toxicity_df['id'].isin(filtered_ids)]
    language_df = language_df[language_df['id'].isin(filtered_ids)]
    
    print(f"After filtering: {len(toxicity_df)} toxicity samples, {len(language_df)} language samples")
    
    # Add language category to language data
    language_df['language_category'] = language_df.apply(categorize_language_mix, axis=1)
    
    # Merge toxicity data with language categories
    merged_df = pd.merge(toxicity_df, 
                        language_df[['id', 'hindi_words', 'english_words', 'other_words', 
                                   'total_words', 'language_category']], 
                        on='id', how='inner')
    
    print(f"Merged dataset has {len(merged_df)} samples")
    
    # Count samples in each category
    category_counts = merged_df['language_category'].value_counts()
    print("Language category distribution:")
    for category, count in category_counts.items():
        print(f"  {category}: {count} samples")
    
    # Save the combined dataset
    merged_df.to_csv(os.path.join(args.output_dir, 'hindi_categorized_toxicity.csv'), index=False)
    print(f"Saved complete dataset to {os.path.join(args.output_dir, 'hindi_categorized_toxicity.csv')}")
    
    # Create separate datasets for monolingual and code-switched text
    monolingual = merged_df[merged_df['language_category'] == 'monolingual_hindi']
    code_switched = merged_df[merged_df['language_category'].str.contains('code_switched')]
    
    monolingual.to_csv(os.path.join(args.output_dir, 'monolingual_hindi_samples.csv'), index=False)
    code_switched.to_csv(os.path.join(args.output_dir, 'code_switched_samples.csv'), index=False)
    
    print(f"Created separate datasets: {len(monolingual)} monolingual Hindi samples, {len(code_switched)} code-switched samples")

if __name__ == "__main__":
    main() 