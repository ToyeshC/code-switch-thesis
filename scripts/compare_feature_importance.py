#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compare feature importance between monolingual and code-switched texts.
Identifies which words or patterns contribute most to toxicity predictions in different language categories.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import defaultdict
import re

# Set up command line arguments
parser = argparse.ArgumentParser(description='Compare feature importance across language categories')
parser.add_argument('--monolingual_dir', type=str, required=True, help='Directory with monolingual explanations')
parser.add_argument('--code_switched_dir', type=str, required=True, help='Directory with code-switched explanations')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save comparison results')
args = parser.parse_args()

def load_explanations(file_path):
    """Load feature explanations from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_language(word):
    """Detect if a word is likely Hindi or English."""
    # Hindi characters typically in the Unicode range for Devanagari
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    if devanagari_pattern.search(word):
        return 'hindi'
    
    # Simple heuristic for Romanized Hindi vs English
    # (In a production system, you'd want a more sophisticated approach)
    common_romanized_hindi_patterns = [
        r'hai$', r'nahi', r'kya', r'aap', r'tum', r'hum', 
        r'ko$', r'se$', r'ka$', r'ki$', r'ke$', r'mai', r'mein'
    ]
    
    for pattern in common_romanized_hindi_patterns:
        if re.search(pattern, word.lower()):
            return 'romanized_hindi'
    
    return 'english'

def analyze_features(explanations):
    """Analyze feature importance by word and language."""
    # Collect feature importance data
    feature_importance = defaultdict(float)
    feature_count = defaultdict(int)
    language_importance = defaultdict(float)
    
    # Process each explanation
    for exp in explanations:
        for word, importance in exp['features'].items():
            word_clean = word.strip()
            if not word_clean:
                continue
                
            # Detect language
            lang = get_language(word_clean)
            
            # Add to overall feature importance
            feature_importance[word_clean] += abs(importance)
            feature_count[word_clean] += 1
            
            # Add to language-specific importance
            language_importance[lang] += abs(importance)
    
    # Calculate average importance
    avg_importance = {
        word: importance / feature_count[word] 
        for word, importance in feature_importance.items()
    }
    
    # Sort by importance
    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Create summary dataframe
    summary_df = pd.DataFrame(sorted_features, columns=['feature', 'avg_importance'])
    
    # Add language column
    summary_df['language'] = summary_df['feature'].apply(get_language)
    
    # Normalize language importance by count
    total_importance = sum(language_importance.values())
    normalized_language_importance = {
        lang: importance / total_importance
        for lang, importance in language_importance.items()
    }
    
    return summary_df, normalized_language_importance

def main():
    # Load explanations
    mono_file = os.path.join(args.monolingual_dir, 'lime_explanations.json')
    cs_file = os.path.join(args.code_switched_dir, 'lime_explanations.json')
    
    print(f"Loading monolingual explanations from {mono_file}")
    mono_explanations = load_explanations(mono_file)
    
    print(f"Loading code-switched explanations from {cs_file}")
    cs_explanations = load_explanations(cs_file)
    
    print(f"Analyzing {len(mono_explanations)} monolingual and {len(cs_explanations)} code-switched explanations")
    
    # Analyze features
    mono_features, mono_lang_importance = analyze_features(mono_explanations)
    cs_features, cs_lang_importance = analyze_features(cs_explanations)
    
    # Save analysis results
    mono_features.to_csv(os.path.join(args.output_dir, 'monolingual_feature_importance.csv'), index=False)
    cs_features.to_csv(os.path.join(args.output_dir, 'code_switched_feature_importance.csv'), index=False)
    
    # Compare top features
    top_n = 20
    mono_top = mono_features.head(top_n)
    cs_top = cs_features.head(top_n)
    
    # Create comparison visualizations
    plt.figure(figsize=(12, 10))
    
    # Language importance comparison
    plt.subplot(2, 1, 1)
    languages = sorted(set(mono_lang_importance.keys()) | set(cs_lang_importance.keys()))
    mono_values = [mono_lang_importance.get(lang, 0) for lang in languages]
    cs_values = [cs_lang_importance.get(lang, 0) for lang in languages]
    
    x = np.arange(len(languages))
    width = 0.35
    
    plt.bar(x - width/2, mono_values, width, label='Monolingual Hindi')
    plt.bar(x + width/2, cs_values, width, label='Code-switched')
    
    plt.xlabel('Language')
    plt.ylabel('Normalized Importance')
    plt.title('Contribution to Toxicity by Language')
    plt.xticks(x, languages)
    plt.legend()
    
    # Feature comparison
    plt.subplot(2, 1, 2)
    
    # Create a set of all features from both datasets
    combined_features = pd.concat([
        mono_top[['feature', 'avg_importance']].assign(source='monolingual'),
        cs_top[['feature', 'avg_importance']].assign(source='code_switched')
    ])
    
    # Create overlapping features plot
    sns.barplot(x='avg_importance', y='feature', hue='source', data=combined_features)
    plt.title('Top Features Contributing to Toxicity Predictions')
    plt.xlabel('Average Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(args.output_dir, 'toxicity_feature_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Create a comparison table of common features
    mono_dict = dict(zip(mono_features['feature'], mono_features['avg_importance']))
    cs_dict = dict(zip(cs_features['feature'], cs_features['avg_importance']))
    
    common_features = set(mono_dict.keys()) & set(cs_dict.keys())
    
    comparison_data = []
    for feature in common_features:
        comparison_data.append({
            'feature': feature,
            'monolingual_importance': mono_dict[feature],
            'code_switched_importance': cs_dict[feature],
            'difference': cs_dict[feature] - mono_dict[feature]
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values(by='difference', ascending=False)
    
    # Save comparison table
    comparison_df.to_csv(os.path.join(args.output_dir, 'common_feature_comparison.csv'), index=False)
    
    # Generate a summary report
    summary = {
        'monolingual_samples': len(mono_explanations),
        'code_switched_samples': len(cs_explanations),
        'monolingual_unique_features': len(mono_features),
        'code_switched_unique_features': len(cs_features),
        'common_features': len(common_features),
        'monolingual_language_importance': mono_lang_importance,
        'code_switched_language_importance': cs_lang_importance,
        'top_features_more_important_in_code_switched': comparison_df.head(10)[['feature', 'difference']].to_dict('records'),
        'top_features_more_important_in_monolingual': comparison_df.tail(10)[['feature', 'difference']].to_dict('records')
    }
    
    # Save summary report
    with open(os.path.join(args.output_dir, 'feature_importance_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 