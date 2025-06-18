import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from transformers import T5Tokenizer, T5ForConditionalGeneration

# --- Style Configuration ---
professional_colors = ['#2E86AB', '#A8DADC', '#457B9D', '#1D3557', '#A2E4B8', '#52B69A']
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


class PerplexityCalculator:
    """Calculates perplexity using a pre-trained model."""
    def __init__(self, model_id='google/mt5-xl', device=None):
        self.model_id = model_id
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing Perplexity Calculator with {self.model_id}")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_id)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device.upper()}")

    def calculate_batch_perplexity(self, texts, batch_size=4):
        perplexities = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                # Calculate loss for the batch
                loss = outputs.loss.item()
                # The perplexity is the exponential of the loss
                perplexity = np.exp(loss)
                # Append perplexity for each item in the batch
                perplexities.extend([perplexity] * len(batch_texts))
        return perplexities


class SemEvalSyntheticComparator:
    """
    Performs a comparative analysis between SemEval tweets and synthetic code-switched data.
    """
    def __init__(self, tweets_file, synthetic_file, output_dir):
        self.tweets_file = tweets_file
        self.synthetic_file = synthetic_file
        self.output_dir = os.path.join(output_dir, "experiment_f")
        
        # Data containers
        self.tweets_df = None
        self.synthetic_df = None
        
        # Analysis results
        self.perplexity_results = {}
        self.correlation_results = {}
        self.eda_results = {}
        self.comparison_results = {}
        
        # Config
        self.toxicity_dimensions = ['toxicity', 'severe_toxicity', 'identity_attack', 'insult', 'profanity']
        self.models = ['llama3', 'llama31', 'aya']

    def create_output_directory(self):
        """Create output directory structure."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'visualizations'), exist_ok=True)
        print(f"Output directory created: {self.output_dir}")

    def load_and_preprocess_data(self):
        """Load and preprocess both datasets."""
        print("Loading datasets...")
        try:
            self.tweets_df = pd.read_csv(self.tweets_file)
            self.tweets_df['dataset_type'] = 'SemEval_Tweets'
            print(f"Loaded {len(self.tweets_df)} tweets")
        except Exception as e:
            print(f"Error loading tweets data: {e}")
            return False

        try:
            self.synthetic_df = pd.read_csv(self.synthetic_file)
            self.synthetic_df['dataset_type'] = 'Synthetic_Data'
            print(f"Loaded {len(self.synthetic_df)} synthetic samples")
        except Exception as e:
            print(f"Error loading synthetic data: {e}")
            return False
            
        return True

    def calculate_tweets_perplexity(self):
        """Calculate perplexity for tweets and their continuations."""
        print("\n=== Step 4: Fluency Analysis (Perplexity) for Tweets ===")
        perplexity_calc = PerplexityCalculator()
        
        print("Calculating perplexity for original tweets...")
        tweets_texts = self.tweets_df['generated'].fillna('').astype(str).tolist()
        self.tweets_df['original_perplexity'] = perplexity_calc.calculate_batch_perplexity(tweets_texts)
        
        for model in self.models:
            print(f"Calculating perplexity for {model} continuations...")
            col_name = f"{model}_generated_continuation"
            perp_col = f"{model}_continuation_perplexity"
            continuations = self.tweets_df[col_name].fillna('').astype(str).tolist()
            self.tweets_df[perp_col] = perplexity_calc.calculate_batch_perplexity(continuations)
        
        self.perplexity_results['tweets'] = self.tweets_df[['original_perplexity'] + [f"{m}_continuation_perplexity" for m in self.models]].describe().to_dict()
        print("Perplexity calculation completed for tweets")

    def analyze_tweets_correlations(self):
        """Analyze correlations in the tweets dataset."""
        print("\n=== Step 5: Correlation Analysis for Tweets ===")
        self.correlation_results['tweets'] = {}
        lang_features = ['total_hindi_percent', 'english_percent']
        tox_features = [f'generated_{dim}' for dim in self.toxicity_dimensions]
        
        correlations = {}
        for lang_feat in lang_features:
            for tox_feat in tox_features:
                df_clean = self.tweets_df[[lang_feat, tox_feat]].dropna()
                if len(df_clean) > 1:
                    r, p = spearmanr(df_clean[lang_feat], df_clean[tox_feat])
                    correlations[f"{lang_feat}_vs_{tox_feat}"] = {'spearman_r': r, 'p_value': p}
        self.correlation_results['tweets']['language_toxicity'] = correlations
        print("Analyzing language composition vs toxicity correlations...")
        
        perp_tox_correlations = {}
        for tox_feat in tox_features:
            df_clean = self.tweets_df[['original_perplexity', tox_feat]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(df_clean) > 1:
                r, p = spearmanr(df_clean['original_perplexity'], df_clean[tox_feat])
                perp_tox_correlations[f"perplexity_vs_{tox_feat}"] = {'spearman_r': r, 'p_value': p}
        self.correlation_results['tweets']['perplexity_toxicity'] = perp_tox_correlations
        print("Analyzing perplexity vs toxicity correlations...")
        print("Correlation analysis completed for tweets")

    def perform_tweets_eda(self):
        """Perform Exploratory Data Analysis on the tweets dataset."""
        print("\n=== Step 6: Comprehensive EDA for Tweets ===")
        self.eda_results['tweets'] = {}
        self.eda_results['tweets']['basic_stats'] = self.tweets_df.describe().to_dict()
        print("Computing basic statistics...")
        self.eda_results['tweets']['language_composition'] = self.tweets_df[['total_hindi_percent', 'english_percent']].describe().to_dict()
        print("Analyzing language composition...")
        if 'sentiment' in self.tweets_df.columns:
            self.eda_results['tweets']['sentiment_distribution'] = self.tweets_df['sentiment'].value_counts().to_dict()
            print("Analyzing sentiment distribution...")
        tox_features = [f'generated_{dim}' for dim in self.toxicity_dimensions]
        self.eda_results['tweets']['toxicity_distributions'] = self.tweets_df[tox_features].describe().to_dict()
        print("Analyzing toxicity distributions...")
        print("EDA completed for tweets")

    def perform_cross_dataset_comparison(self):
        """Compare analyses across tweets and synthetic datasets."""
        print("\n=== Cross-Dataset Comparison ===")
        comparison_results = {}
        comparison_results['toxicity'] = self._compare_feature('generated_toxicity')
        print("Comparing toxicity patterns...")
        comparison_results['language_mixing'] = {
            'hindi_percent': self._compare_feature('total_hindi_percent'),
            'english_percent': self._compare_feature('english_percent')
        }
        print("Comparing language mixing patterns...")
        comparison_results['model_performance'] = self._compare_model_performance()
        print("Comparing model performance...")
        self.comparison_results = comparison_results
        print("Cross-dataset comparison completed")

    def _compare_feature(self, feature_name):
        """Helper to compare a single feature across datasets."""
        return {
            'tweets': self.tweets_df[feature_name].describe().to_dict(),
            'synthetic': self.synthetic_df[feature_name].describe().to_dict()
        }

    def _compare_model_performance(self):
        """Helper to compare model performance (toxicity shift)."""
        model_comparison = {}
        for model in self.models:
            # For synthetic data, the comparison is between the 'generated' text and its continuation
            synth_orig_tox_col = 'generated_toxicity'
            synth_cont_tox_col = f'{model}_generated_continuation_toxicity'
            synth_change = (self.synthetic_df[synth_cont_tox_col] - self.synthetic_df[synth_orig_tox_col]).describe().to_dict()
            
            # For tweets, the comparison is also between the 'generated' text (original tweet) and its continuation
            tweet_orig_tox_col = 'generated_toxicity'
            tweet_cont_tox_col = f'{model}_generated_continuation_toxicity'
            tweet_change = (self.tweets_df[tweet_cont_tox_col] - self.tweets_df[tweet_orig_tox_col]).describe().to_dict()
            
            model_comparison[model] = {
                'synthetic_toxicity_change': synth_change,
                'tweets_toxicity_change': tweet_change
            }
        return model_comparison
        
    def create_visualizations(self):
        """Create comprehensive visualizations for all analyses."""
        print("\n=== Creating Visualizations ===")
        viz_dir = os.path.join(self.output_dir, 'visualizations')

        self._create_toxicity_comparison_plots(viz_dir)
        self._create_language_composition_plots(viz_dir)
        if hasattr(self, 'tweets_df') and 'original_perplexity' in self.tweets_df.columns:
            self._create_perplexity_plots(viz_dir)
        if hasattr(self, 'correlation_results') and 'tweets' in self.correlation_results:
            self._create_correlation_heatmaps(viz_dir)
        
        print("Visualizations created successfully")

    def _create_toxicity_comparison_plots(self, viz_dir):
        if 'generated_toxicity' not in self.tweets_df.columns or 'generated_toxicity' not in self.synthetic_df.columns: return
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(self.tweets_df['generated_toxicity'].dropna(), ax=ax, label='SemEval Tweets', color=professional_colors[0], fill=True)
        sns.kdeplot(self.synthetic_df['generated_toxicity'].dropna(), ax=ax, label='Synthetic Data', color=professional_colors[1], fill=True)
        ax.set_title('Code-Switched Text Toxicity Distribution Comparison')
        ax.set_xlabel('Toxicity Score'); ax.set_ylabel('Density'); ax.legend()
        plt.tight_layout(); plt.savefig(os.path.join(viz_dir, 'toxicity_distribution_comparison.png')); plt.close()

    def _create_language_composition_plots(self, viz_dir):
        lang_features = ['total_hindi_percent', 'english_percent']
        if not all(f in self.tweets_df.columns and f in self.synthetic_df.columns for f in lang_features): return
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for i, feature in enumerate(lang_features):
            sns.kdeplot(self.tweets_df[feature].dropna(), ax=axes[i], label='SemEval Tweets', color=professional_colors[0], fill=True)
            sns.kdeplot(self.synthetic_df[feature].dropna(), ax=axes[i], label='Synthetic Data', color=professional_colors[1], fill=True)
            axes[i].set_title(f'{feature.replace("_", " ").title()} Distribution'); axes[i].set_xlabel('Percentage'); axes[i].legend()
        plt.tight_layout(); plt.savefig(os.path.join(viz_dir, 'language_composition_comparison.png')); plt.close()

    def _create_perplexity_plots(self, viz_dir):
        fig, ax = plt.subplots(figsize=(10, 6))
        perp = self.tweets_df['original_perplexity'].replace([np.inf, -np.inf], np.nan).dropna()
        
        # Filter out non-informative perplexity values (e.g., for empty strings, perplexity is ~1)
        perp_filtered = perp[perp > 1.1]
        
        # Cap at 95th percentile for better visualization of the main distribution
        if not perp_filtered.empty:
            perp_capped = perp_filtered[perp_filtered <= perp_filtered.quantile(0.95)]
            sns.histplot(perp_capped, ax=ax, bins=30, color=professional_colors[0])
            ax.set_title('Tweet Perplexity Distribution (Perplexity > 1.1, Capped at 95th Percentile)')
        else:
            ax.text(0.5, 0.5, "No data to display for perplexity > 1.1", horizontalalignment='center', verticalalignment='center')
            ax.set_title('Tweet Perplexity Distribution')
            
        ax.set_xlabel('Perplexity')
        plt.tight_layout(); plt.savefig(os.path.join(viz_dir, 'tweet_perplexity_distribution.png')); plt.close()

    def _create_correlation_heatmaps(self, viz_dir):
        if 'language_toxicity' not in self.correlation_results.get('tweets', {}): return
        corr_data = [{'lang': k.split('_vs_')[0], 'tox': k.split('_vs_')[1], 'r': v.get('spearman_r', 0)} 
                     for k, v in self.correlation_results['tweets']['language_toxicity'].items()]
        if not corr_data: return
        pivot = pd.DataFrame(corr_data).pivot_table(index='lang', columns='tox', values='r')
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Language vs. Toxicity Correlation in Tweets'); plt.xlabel('Toxicity Dimension'); plt.ylabel('Language Feature')
        plt.tight_layout(); plt.savefig(os.path.join(viz_dir, 'tweets_correlation_heatmap.png')); plt.close()

    def run_analysis(self):
        """Run the complete comparative analysis."""
        print("="*70 + "\nCOMPARATIVE ANALYSIS: SEMEVAL TWEETS vs SYNTHETIC DATA\n" + "="*70)
        self.create_output_directory()
        if not self.load_and_preprocess_data():
             print("Failed to load data. Exiting.")
             return

        self.calculate_tweets_perplexity()
        self.analyze_tweets_correlations()
        self.perform_tweets_eda()
        self.perform_cross_dataset_comparison()
        self.create_visualizations()
        self.save_results()
        
        print("\n" + "="*70 + "\nANALYSIS COMPLETED SUCCESSFULLY\n" + f"Results saved to: {self.output_dir}\n" + "="*70)

    def save_results(self):
        """Save all analysis results to files."""
        print("\n=== Saving Results ===")
        
        if hasattr(self, 'tweets_df') and 'original_perplexity' in self.tweets_df.columns:
            path = os.path.join(self.output_dir, 'tweets_perplexity_results.csv')
            self.tweets_df.to_csv(path, index=False)
            print(f"Saved tweets perplexity data to {path}")
        
        if self.correlation_results:
            path = os.path.join(self.output_dir, 'tweets_correlation_results.json')
            with open(path, 'w') as f: json.dump(self.correlation_results, f, indent=2, default=str)
            print(f"Saved correlation results to {path}")
            
        if self.eda_results:
            path = os.path.join(self.output_dir, 'tweets_eda_results.json')
            with open(path, 'w') as f: json.dump(self.eda_results, f, indent=2, default=str)
            print(f"Saved EDA results to {path}")
            
        if self.comparison_results:
            path = os.path.join(self.output_dir, 'cross_dataset_comparison.json')
            with open(path, 'w') as f: json.dump(self.comparison_results, f, indent=2, default=str)
            print(f"Saved comparison results to {path}")
            
        print("All results saved successfully")


def main():
    parser = argparse.ArgumentParser(description='Comparative Analysis: SemEval Tweets vs Synthetic Data')
    parser.add_argument('--tweets_file', required=True, help='Path to tweets CSV file')
    parser.add_argument('--synthetic_file', required=True, help='Path to synthetic data CSV file')
    parser.add_argument('--output_dir', required=True, help='Base output directory')
    args = parser.parse_args()
    
    analyzer = SemEvalSyntheticComparator(args.tweets_file, args.synthetic_file, args.output_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 