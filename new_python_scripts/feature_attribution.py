#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import (
    IntegratedGradients,
    LayerIntegratedGradients,
    Occlusion,
    visualization
)
import json
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import matplotlib.font_manager as fm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class FeatureAttributor:
    def __init__(self, model_dir, device=None):
        """Initialize the feature attribution class with a trained model."""
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        
        # No more initialization of attribution methods - we'll use a simpler approach
    
    def _encode_text(self, text):
        """Tokenize and encode text."""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Force conversion to long for all ID-based tensors
            if "input_ids" in inputs:
                inputs["input_ids"] = inputs["input_ids"].long()
            
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"].long()
                
            if "token_type_ids" in inputs:
                inputs["token_type_ids"] = inputs["token_type_ids"].long()
                
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            return inputs
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return None
    
    def _predict(self, inputs):
        """Run prediction with the model."""
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                return torch.sigmoid(logits).item() if logits.numel() > 0 else 0.5
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return 0.5  # Default neutral prediction
    
    def simple_token_attribution(self, text):
        """
        Perform a simple attribution by measuring the change in prediction
        when each token is removed (similar to occlusion but simpler).
        """
        # Initial encoding and prediction
        inputs = self._encode_text(text)
        if not inputs:
            return {
                "tokens": [],
                "attributions": [],
                "prediction": 0.5,
                "text": text
            }
        
        input_ids = inputs["input_ids"][0].cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        # Get baseline prediction
        try:
            baseline_pred = self._predict(inputs)
        except Exception as e:
            logger.error(f"Error getting baseline prediction: {e}")
            return {
                "tokens": tokens,
                "attributions": [0.0] * len(tokens),
                "prediction": 0.5,
                "text": text
            }
        
        # Initialize attribution scores
        scores = np.zeros(len(tokens))
        
        # Create attention mask for masking tokens one by one
        attention_mask = inputs["attention_mask"][0].clone()
        
        # For each token, mask it and observe the change in prediction
        for i in range(len(tokens)):
            if tokens[i].startswith("##") or tokens[i] in ["[CLS]", "[SEP]", "[PAD]"]:
                continue  # Skip subwords and special tokens
                
            # Create a mask that masks out this token
            mask = attention_mask.clone()
            mask[i] = 0
            
            # Create masked inputs
            masked_inputs = {
                "input_ids": inputs["input_ids"].clone(),
                "attention_mask": mask.unsqueeze(0)
            }
            
            if "token_type_ids" in inputs:
                masked_inputs["token_type_ids"] = inputs["token_type_ids"].clone()
            
            # Get prediction with token masked
            try:
                masked_pred = self._predict(masked_inputs)
                # Calculate attribution: how much the prediction changes when token is removed
                scores[i] = baseline_pred - masked_pred
            except Exception as e:
                logger.error(f"Error computing attribution for token {i}: {e}")
        
        # Normalize scores
        if np.linalg.norm(scores) > 0:
            scores = scores / np.linalg.norm(scores)
        
        return {
            "tokens": tokens,
            "attributions": scores.tolist(),
            "prediction": baseline_pred,
            "text": text
        }
    
    def generate_visualizations(self, result, output_dir, prefix=""):
        """Generate HTML visualizations of attributions."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            if not result or "tokens" not in result or "attributions" not in result:
                return
                
            tokens = result["tokens"]
            attributions = result["attributions"]
            prediction = result["prediction"]
            
            # Skip if no tokens or attributions
            if not tokens or not attributions or len(tokens) != len(attributions):
                return
            
            # Create file path
            file_path = os.path.join(output_dir, f"{prefix}_attribution_viz.html")
            
            # Create a simple HTML visualization
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Token Attribution</title>
                <style>
                    body {{ font-family: sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333; }}
                    .token {{ padding: 2px; margin: 2px; display: inline-block; }}
                    .positive {{ background-color: rgba(0, 255, 0, 0.3); }}
                    .negative {{ background-color: rgba(255, 0, 0, 0.3); }}
                </style>
            </head>
            <body>
                <h1>Token Attribution Visualization</h1>
                <h2>Prediction Score: {prediction:.4f}</h2>
                <div>
            """
            
            for token, score in zip(tokens, attributions):
                if score > 0:
                    intensity = min(abs(score) * 5, 1.0)
                    html_content += f'<span class="token positive" style="background-color: rgba(0, 255, 0, {intensity});">{token}</span>'
                else:
                    intensity = min(abs(score) * 5, 1.0)
                    html_content += f'<span class="token negative" style="background-color: rgba(255, 0, 0, {intensity});">{token}</span>'
            
            html_content += """
                </div>
            </body>
            </html>
            """
            
            with open(file_path, "w") as f:
                f.write(html_content)
            
            logger.info(f"Saved visualization to {file_path}")
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
    
    def analyze_text(self, text, output_dir=None, prefix=""):
        """Run attribution on a text and optionally save visualizations."""
        result = self.simple_token_attribution(text)
        
        # Generate visualizations if output directory provided
        if output_dir and result:
            self.generate_visualizations(result, output_dir, prefix)
        
        return {"simple_attribution": result}

def process_data(
    input_file, 
    model_dir, 
    output_dir, 
    text_column, 
    max_samples=None, 
    random_state=42
):
    """Process data from a CSV file and generate feature attributions."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Check if the specified text column exists
    if text_column not in df.columns:
        available_columns = df.columns.tolist()
        logger.error(f"Column '{text_column}' not found in {input_file}")
        logger.info(f"Available columns: {available_columns}")
        
        # Try to find a suitable text column
        potential_text_columns = [col for col in available_columns if 
                                 any(keyword in col.lower() for keyword in 
                                     ["text", "continuation", "content", "generated"])]
        
        if potential_text_columns:
            text_column = potential_text_columns[0]
            logger.info(f"Using '{text_column}' as the text column instead")
        else:
            logger.error("No suitable text column found. Exiting.")
            return []
    
    # Limit samples if specified
    if max_samples and max_samples < len(df):
        df = df.sample(max_samples, random_state=random_state)
        logger.info(f"Using {max_samples} samples out of {len(df)}")
    
    # Initialize feature attributor
    attributor = FeatureAttributor(model_dir)
    
    # Process each text
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing texts"):
        text = row[text_column]
        
        # Skip empty texts
        if not isinstance(text, str) or text.strip() == "":
            logger.warning(f"Skipping empty text at index {idx}")
            continue
        
        try:
            # Generate prefix for files
            prefix = f"sample_{idx}"
            
            # Run attribution
            attribution_results = attributor.analyze_text(
                text, 
                output_dir=os.path.join(output_dir, "visualizations"),
                prefix=prefix
            )
            
            # Save attribution scores and metadata
            sample_result = {
                "sample_id": idx,
                "text": text,
                "attribution_results": attribution_results
            }
            
            results.append(sample_result)
            
            # Save individual JSON file
            individual_output_path = os.path.join(
                output_dir, 
                "individual_results", 
                f"{prefix}_attribution.json"
            )
            os.makedirs(os.path.dirname(individual_output_path), exist_ok=True)
            
            with open(individual_output_path, "w") as f:
                json.dump(sample_result, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error processing text at index {idx}: {str(e)}")
    
    # Save all results to a single file
    all_results_path = os.path.join(output_dir, "all_attribution_results.json")
    with open(all_results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved all attribution results to {all_results_path}")
    
    # Generate summary statistics
    generate_summary(results, output_dir)
    
    return results

def generate_summary(results, output_dir):
    """Generate summary statistics and plots from attribution results."""
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Set up matplotlib to use a font that supports Devanagari
    # Try to find a suitable font that supports Devanagari
    font_paths = [
        '/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf',  # Common on Linux
        '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf',
        '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
        # Add more potential font paths if needed
    ]
    
    # Try to set a font that supports Devanagari
    font_found = False
    for font_path in font_paths:
        if os.path.exists(font_path):
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [fm.FontProperties(fname=font_path).get_name()]
            font_found = True
            break
    
    if not font_found:
        logger.warning("Could not find a font with Devanagari support. Hindi characters may not display correctly.")
        # Fallback: try to use a basic Unicode font
        plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Extract predictions and token scores
    predictions = []
    token_scores = defaultdict(list)
    
    for result in results:
        # Get prediction from attribution
        attribution = result["attribution_results"].get("simple_attribution", {})
        if attribution and "prediction" in attribution:
            predictions.append(attribution["prediction"])
        
        # Get token scores
        if attribution and "tokens" in attribution and "attributions" in attribution:
            tokens = attribution["tokens"]
            attributions = attribution["attributions"]
            
            # Skip if lengths don't match or empty
            if not tokens or not attributions or len(tokens) != len(attributions):
                continue
            
            # Update token scores, excluding special tokens
            for token, score in zip(tokens, attributions):
                if not token.startswith('[') and not token.endswith(']') and not token.startswith('##'):
                    token_scores[token].append(score)
    
    # Generate prediction distribution plot
    if predictions:
        plt.figure(figsize=(10, 6))
        plt.hist(predictions, bins=20, alpha=0.7)
        plt.title("Distribution of Predictions")
        plt.xlabel("Prediction Score")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(summary_dir, "prediction_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Calculate average attribution for each token
    avg_token_scores = {}
    for token, scores in token_scores.items():
        if len(scores) >= 3:  # Only include tokens that appear in multiple samples
            avg_token_scores[token] = np.mean(scores)
    
    if avg_token_scores:
        # Get top and bottom tokens by attribution
        top_tokens = sorted(avg_token_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        bottom_tokens = sorted(avg_token_scores.items(), key=lambda x: x[1])[:20]
        
        # Save token scores with proper encoding
        with open(os.path.join(summary_dir, "token_scores.json"), "w", encoding='utf-8') as f:
            json.dump({"top": dict(top_tokens), "bottom": dict(bottom_tokens)}, f, indent=2, ensure_ascii=False)
        
        # Function to create a clean label for display
        def clean_token_label(token):
            # Convert token to a readable format
            label = token.replace('▁', ' ').strip()  # Handle special characters
            # Add token score to label
            return f"{label} ({avg_token_scores[token]:.3f})"
        
        # Generate bar plots for top tokens if we have any
        if top_tokens:
            plt.figure(figsize=(15, 10))  # Larger figure for better text rendering
            token_labels = [clean_token_label(t[0]) for t in reversed(top_tokens)]
            scores = [t[1] for t in reversed(top_tokens)]
            
            bars = plt.barh(range(len(scores)), scores)
            plt.yticks(range(len(token_labels)), token_labels)
            plt.title("Top 20 Tokens by Attribution Score")
            plt.xlabel("Average Attribution Score")
            
            # Add value labels on the bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2,
                        f'{width:.3f}',
                        ha='left', va='center', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(os.path.join(summary_dir, "top_tokens.png"), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
        # Generate bar plots for bottom tokens if we have any
        if bottom_tokens:
            plt.figure(figsize=(15, 10))  # Larger figure for better text rendering
            token_labels = [clean_token_label(t[0]) for t in reversed(bottom_tokens)]
            scores = [t[1] for t in reversed(bottom_tokens)]
            
            bars = plt.barh(range(len(scores)), scores)
            plt.yticks(range(len(token_labels)), token_labels)
            plt.title("Bottom 20 Tokens by Attribution Score")
            plt.xlabel("Average Attribution Score")
            
            # Add value labels on the bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2,
                        f'{width:.3f}',
                        ha='left' if width >= 0 else 'right', va='center', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(os.path.join(summary_dir, "bottom_tokens.png"), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
    
    logger.info(f"Generated summary plots in {summary_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate feature attributions for model predictions")
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to input CSV file with text data"
    )
    parser.add_argument(
        "--model_dir", 
        type=str, 
        required=True, 
        help="Directory with trained classifier model"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to save attribution results"
    )
    parser.add_argument(
        "--text_column", 
        type=str, 
        required=True, 
        help="Column containing text data"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None, 
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--random_state", 
        type=int, 
        default=42, 
        help="Random state for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Process data
    process_data(
        input_file=args.input_file,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        text_column=args.text_column,
        max_samples=args.max_samples,
        random_state=args.random_state
    )
    
    logger.info("Feature attribution completed successfully")

if __name__ == "__main__":
    main() 