#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature attribution analysis for toxicity detection in code-switched text.
Implements LIME and SHAP methods to explain which words contribute most to toxicity predictions.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer
import shap
from collections import defaultdict
import re
import json

# Set up command line arguments
parser = argparse.ArgumentParser(description='Feature attribution for toxicity classification')
parser.add_argument('--input_file', type=str, required=True, help='CSV file with text to explain')
parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned model')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save explanations')
parser.add_argument('--method', type=str, choices=['lime', 'shap', 'both'], default='both', 
                    help='Explanation method to use (default: both)')
parser.add_argument('--num_samples', type=int, default=5, 
                    help='Number of samples to explain (use -1 for all)')
parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
parser.add_argument('--num_features', type=int, default=20, 
                    help='Number of features to include in explanations')
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

def load_data(input_file):
    """Load the data from a CSV file."""
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Identify text column
    text_cols = ['text', 'prompt', 'response', 'sentence', 'content']
    for col in text_cols:
        if col in df.columns:
            text_column = col
            break
    else:
        # If no standard text column is found, use first string column
        text_column = df.select_dtypes(include=['object']).columns[0]
    
    print(f"Using '{text_column}' as text column")
    
    # Identify ID column
    id_cols = ['prompt_id', 'id', 'response_id']
    for col in id_cols:
        if col in df.columns:
            id_column = col
            break
    else:
        # If no ID column, create one
        df['id'] = range(len(df))
        id_column = 'id'
    
    # Sample data if specified
    if args.num_samples > 0 and args.num_samples < len(df):
        df = df.sample(args.num_samples, random_state=42)
    
    return df, text_column, id_column

def load_model(model_path):
    """Load the pretrained BERT model and tokenizer."""
    print(f"Loading model from {model_path}...")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    
    # Set model to evaluation mode
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, tokenizer, device

def predict_proba(texts, model, tokenizer, device):
    """
    Predict probability of toxicity for a list of texts.
    Returns a numpy array of shape (len(texts), 2) with probabilities for [non-toxic, toxic].
    """
    # Tokenize inputs
    encoded_inputs = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=args.max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoded_inputs['input_ids'].to(device)
    attention_mask = encoded_inputs['attention_mask'].to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    # Convert to probabilities
    if logits.shape[1] == 1:  # Binary classification
        probs = torch.sigmoid(logits).cpu().numpy()
        # Format as [non-toxic_prob, toxic_prob] for each sample
        return np.hstack([1-probs, probs])
    else:  # Multi-class classification
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

def predict_fn(texts):
    """Wrapper function for LIME and SHAP to use."""
    return predict_proba(texts, model, tokenizer, device)

def lime_explanation(texts, text_column, id_column, ids):
    """Generate LIME explanations for the texts."""
    print("Generating LIME explanations...")
    
    # Initialize LIME explainer
    class_names = ['non-toxic', 'toxic']
    explainer = LimeTextExplainer(class_names=class_names)
    
    explanations = []
    
    for i, text in enumerate(tqdm(texts)):
        # Generate explanation
        exp = explainer.explain_instance(
            text, 
            predict_fn, 
            num_features=args.num_features,
            num_samples=5000
        )
        
        # Get predicted class
        probs = predict_fn([text])[0]
        pred_class = 1 if probs[1] > 0.5 else 0
        
        # Get top features for the predicted class
        feature_dict = dict(exp.as_list(label=pred_class))
        
        # Store explanation
        explanation = {
            id_column: ids[i],
            'text': text,
            'predicted_class': class_names[pred_class],
            'probability': float(probs[pred_class]),
            'features': feature_dict
        }
        
        explanations.append(explanation)
        
        # Generate HTML visualization
        html_path = os.path.join(args.output_dir, f'lime_explanation_{ids[i]}.html')
        exp.save_to_file(html_path)
        
        # Generate visualization plot
        fig = plt.figure(figsize=(10, 6))
        exp.as_pyplot_figure(label=pred_class)
        plt.title(f"LIME Explanation for '{text[:50]}...'")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'lime_explanation_{ids[i]}.png'), dpi=300)
        plt.close(fig)
    
    # Save all explanations to JSON
    with open(os.path.join(args.output_dir, 'lime_explanations.json'), 'w', encoding='utf-8') as f:
        json.dump(explanations, f, ensure_ascii=False, indent=2)
    
    # Aggregate feature importance across all samples
    feature_importance = defaultdict(float)
    feature_count = defaultdict(int)
    
    for exp in explanations:
        for word, importance in exp['features'].items():
            feature_importance[word] += abs(importance)
            feature_count[word] += 1
    
    # Calculate average importance
    avg_importance = {
        word: importance / feature_count[word] 
        for word, importance in feature_importance.items()
    }
    
    # Sort by importance
    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Create summary dataframe
    summary_df = pd.DataFrame(sorted_features, columns=['feature', 'avg_importance'])
    summary_df['count'] = summary_df['feature'].map(feature_count)
    
    # Save summary
    summary_df.to_csv(os.path.join(args.output_dir, 'lime_feature_importance_summary.csv'), index=False)
    
    # Plot top features
    top_n = 20
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='avg_importance', 
        y='feature', 
        data=summary_df.head(top_n),
        palette='viridis'
    )
    plt.title(f'Top {top_n} Features by Average LIME Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'lime_top_features.png'), dpi=300)
    plt.close()
    
    return explanations

def shap_explanation(texts, text_column, id_column, ids):
    """Generate SHAP explanations for the texts."""
    print("Generating SHAP explanations...")
    
    # Initialize SHAP explainer
    # We'll use the transformers pipeline for SHAP
    from transformers import pipeline
    
    # Create a pipeline
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        return_all_scores=True
    )
    
    # Create a function that returns the model's prediction for SHAP
    def f(x):
        outputs = pipe(list(x))
        # Format outputs as probabilities
        probs = []
        for output in outputs:
            if len(output) == 1:  # Binary classification
                p = output[0]['score']
                probs.append([1-p, p])
            else:  # Multi-class
                probs.append([o['score'] for o in output])
        return np.array(probs)
    
    # Initialize explainer
    explainer = shap.Explainer(f, tokenizer)
    
    # Calculate SHAP values
    # Warning: This can be memory-intensive for large datasets
    shap_values = explainer(texts)
    
    explanations = []
    
    for i, text in enumerate(tqdm(texts)):
        # Get prediction
        probs = predict_fn([text])[0]
        pred_class = 1 if probs[1] > 0.5 else 0
        class_names = ['non-toxic', 'toxic']
        
        # Store explanation
        explanation = {
            id_column: ids[i],
            'text': text,
            'predicted_class': class_names[pred_class],
            'probability': float(probs[pred_class])
        }
        
        # Add tokenized features and their importance
        tokens = tokenizer.tokenize(text)
        token_shap_values = shap_values[i].values[0, 1:len(tokens)+1]  # Skip CLS token and limit to actual tokens
        
        # Create dictionary of token -> shap value
        token_importance = {}
        for token, value in zip(tokens, token_shap_values):
            # Clean token (remove ##)
            clean_token = token.replace('##', '')
            token_importance[clean_token] = float(value)
        
        explanation['token_importance'] = token_importance
        
        # Add to explanations list
        explanations.append(explanation)
        
        # Generate visualization
        plt.figure(figsize=(12, 6))
        shap.plots.text(shap_values[i][:, 1:len(tokens)+1, 1], display=False)  # Focus on toxic class
        plt.title(f"SHAP Explanation for '{text[:50]}...'")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'shap_explanation_{ids[i]}.png'), dpi=300)
        plt.close()
    
    # Save all explanations to JSON
    with open(os.path.join(args.output_dir, 'shap_explanations.json'), 'w', encoding='utf-8') as f:
        json.dump(explanations, f, ensure_ascii=False, indent=2)
    
    # Create summary of token importance
    token_importance_summary = defaultdict(list)
    
    for exp in explanations:
        for token, importance in exp['token_importance'].items():
            token_importance_summary[token].append(importance)
    
    # Calculate average importance
    avg_importance = {
        token: np.mean(abs(np.array(values))) 
        for token, values in token_importance_summary.items()
    }
    
    # Sort by importance
    sorted_tokens = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Create summary dataframe
    summary_df = pd.DataFrame(sorted_tokens, columns=['token', 'avg_importance'])
    summary_df['count'] = summary_df['token'].map(lambda x: len(token_importance_summary[x]))
    
    # Save summary
    summary_df.to_csv(os.path.join(args.output_dir, 'shap_token_importance_summary.csv'), index=False)
    
    # Plot top tokens
    top_n = 20
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='avg_importance', 
        y='token', 
        data=summary_df.head(top_n),
        palette='viridis'
    )
    plt.title(f'Top {top_n} Tokens by Average SHAP Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'shap_top_tokens.png'), dpi=300)
    plt.close()
    
    return explanations

def identify_language_specific_patterns(lime_explanations=None, shap_explanations=None):
    """Identify patterns in feature importance specific to Hindi vs English."""
    print("Analyzing language-specific patterns in feature importance...")
    
    # Regular expressions for identifying Hindi and English text
    hindi_pattern = re.compile(r'[\u0900-\u097F]+')  # Unicode range for Devanagari
    english_pattern = re.compile(r'[a-zA-Z]+')
    
    # Function to determine if a word is Hindi, English, or mixed
    def get_language(word):
        if hindi_pattern.search(word):
            if english_pattern.search(word):
                return 'mixed'
            return 'hindi'
        elif english_pattern.search(word):
            return 'english'
        return 'other'
    
    # Analyze LIME explanations
    if lime_explanations:
        # Collect features by language
        language_features = {
            'hindi': defaultdict(list),
            'english': defaultdict(list),
            'mixed': defaultdict(list),
            'other': defaultdict(list)
        }
        
        # Process all LIME explanations
        for explanation in lime_explanations:
            for word, importance in explanation['features'].items():
                lang = get_language(word)
                language_features[lang][word].append(importance)
        
        # Calculate average importance by language
        language_avg_importance = {}
        for lang, features in language_features.items():
            if features:  # Skip empty categories
                avg_importance = {
                    word: np.mean(abs(np.array(values))) 
                    for word, values in features.items()
                }
                
                # Sort by importance
                sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
                
                # Create dataframe
                df = pd.DataFrame(sorted_features, columns=['feature', 'avg_importance'])
                df['count'] = df['feature'].map(lambda x: len(features[x]))
                df['language'] = lang
                
                language_avg_importance[lang] = df
        
        # Combine and save
        combined_df = pd.concat(language_avg_importance.values())
        combined_df.to_csv(os.path.join(args.output_dir, 'lime_language_feature_importance.csv'), index=False)
        
        # Plot top features by language
        for lang, df in language_avg_importance.items():
            if len(df) > 0:
                top_n = min(10, len(df))
                plt.figure(figsize=(12, 6))
                sns.barplot(
                    x='avg_importance', 
                    y='feature', 
                    data=df.head(top_n),
                    palette='viridis'
                )
                plt.title(f'Top {top_n} {lang.capitalize()} Features by LIME Importance')
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f'lime_top_{lang}_features.png'), dpi=300)
                plt.close()
    
    # Analyze SHAP explanations (similar approach)
    if shap_explanations:
        # Collect tokens by language
        language_tokens = {
            'hindi': defaultdict(list),
            'english': defaultdict(list),
            'mixed': defaultdict(list),
            'other': defaultdict(list)
        }
        
        # Process all SHAP explanations
        for explanation in shap_explanations:
            for token, importance in explanation['token_importance'].items():
                lang = get_language(token)
                language_tokens[lang][token].append(importance)
        
        # Calculate average importance by language
        language_avg_importance = {}
        for lang, tokens in language_tokens.items():
            if tokens:  # Skip empty categories
                avg_importance = {
                    token: np.mean(abs(np.array(values))) 
                    for token, values in tokens.items()
                }
                
                # Sort by importance
                sorted_tokens = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
                
                # Create dataframe
                df = pd.DataFrame(sorted_tokens, columns=['token', 'avg_importance'])
                df['count'] = df['token'].map(lambda x: len(tokens[x]))
                df['language'] = lang
                
                language_avg_importance[lang] = df
        
        # Combine and save
        combined_df = pd.concat(language_avg_importance.values())
        combined_df.to_csv(os.path.join(args.output_dir, 'shap_language_token_importance.csv'), index=False)
        
        # Plot top tokens by language
        for lang, df in language_avg_importance.items():
            if len(df) > 0:
                top_n = min(10, len(df))
                plt.figure(figsize=(12, 6))
                sns.barplot(
                    x='avg_importance', 
                    y='token', 
                    data=df.head(top_n),
                    palette='viridis'
                )
                plt.title(f'Top {top_n} {lang.capitalize()} Tokens by SHAP Importance')
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f'shap_top_{lang}_tokens.png'), dpi=300)
                plt.close()

def main():
    # Load data
    df, text_column, id_column = load_data(args.input_file)
    texts = df[text_column].tolist()
    ids = df[id_column].tolist()
    
    # Load model and tokenizer
    global model, tokenizer, device
    model, tokenizer, device = load_model(args.model_path)
    
    # Perform feature attribution analysis
    lime_explanations = None
    shap_explanations = None
    
    if args.method in ['lime', 'both']:
        lime_explanations = lime_explanation(texts, text_column, id_column, ids)
    
    if args.method in ['shap', 'both']:
        shap_explanations = shap_explanation(texts, text_column, id_column, ids)
    
    # Analyze language-specific patterns
    identify_language_specific_patterns(lime_explanations, shap_explanations)
    
    print(f"Feature attribution analysis complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 