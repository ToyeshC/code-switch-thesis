#!/usr/bin/env python3
"""
Evaluating Code-Switching Generation Methods for Fluency and Toxicity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, f_oneway, shapiro
import argparse
import os
from datetime import datetime
import warnings
from itertools import combinations
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from huggingface_hub import login
import sys

# Add the project root to the Python path to allow importing 'config'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
try:
    import config
except ImportError:
    print("Error: config.py not found. Make sure it's in the project root.")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set matplotlib backend to avoid display issues on cluster
import matplotlib
matplotlib.use('Agg')

# Set professional color scheme
plt.style.use('default')
sns.set_style("whitegrid")
professional_colors = ['#1D3557', '#457B9D', '#A8DADC'] # Dark blue to light blue
sns.set_palette(professional_colors)

class CodeSwitchingEvaluator:
    """Class to evaluate code-switching generation methods."""
    
    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_dir = os.path.join(output_dir, "experiment_c")
        self.df = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Perspective API toxicity dimensions for generated text
        self.toxicity_dimensions = [
            'generated_toxicity', 'generated_severe_toxicity', 'generated_identity_attack', 
            'generated_insult', 'generated_profanity', 'generated_threat'
        ]
        
        # Generation methods
        self.methods = ['baseline', 'silver', 'gold']
        
        # Results storage
        self.statistical_results = pd.DataFrame()

    def login_huggingface(self):
        """Logs in to Hugging Face using the API key from config."""
        print("Logging in to Hugging Face...")
        try:
            if config.HUGGINGFACE_API_KEY:
                login(token=config.HUGGINGFACE_API_KEY)
                print("Hugging Face login successful.")
            else:
                print("Warning: HUGGINGFACE_API_KEY not found in config.py. Proceeding without login.")
        except Exception as e:
            print(f"Hugging Face login failed: {e}")

    def load_data(self):
        """Load the perspective analysis CSV file."""
        print(f"Loading data from {self.input_file}...")
        try:
            self.df = pd.read_csv(self.input_file)
            print(f"Loaded {len(self.df)} rows.")
            
            # Filter for the relevant generation methods
            self.df = self.df[self.df['method'].isin(self.methods)]
            print(f"Filtered to {len(self.df)} rows for methods: {self.methods}")

            # Check for required columns
            required_cols = ['method', 'generated'] + self.toxicity_dimensions
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                print(f"Error: Missing required columns: {missing_cols}")
                return False
            
            # Drop rows with missing generated text
            self.df.dropna(subset=['generated'], inplace=True)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def calculate_perplexity(self):
        """Calculate perplexity for generated code-switched text using mT5-xl."""
        print(f"Calculating perplexity using google/mt5-xl on {self.device}...")
        
        model_name = 'google/mt5-xl'
        try:
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
            model.eval()
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return False

        perplexities = []
        with torch.no_grad():
            for i, text in enumerate(self.df['generated']):
                if i % 100 == 0:
                    print(f"  Processing text {i+1}/{len(self.df)}...")
                
                if not isinstance(text, str) or not text.strip():
                    perplexities.append(np.nan)
                    continue

                try:
                    inputs = tokenizer(text, return_tensors='pt').to(self.device)
                    # For T5, labels should be the same as input_ids for loss calculation
                    labels = inputs['input_ids']
                    outputs = model(**inputs, labels=labels)
                    loss = outputs.loss
                    perplexity = torch.exp(loss).item()
                    perplexities.append(perplexity)
                except Exception as e:
                    print(f"Could not compute perplexity for row {i}: {e}")
                    perplexities.append(np.nan)

        self.df['perplexity'] = perplexities
        print("Perplexity calculation complete.")
        print(self.df[['method', 'perplexity']].groupby('method').describe())
        return True

    def perform_statistical_analysis(self):
        """Compare perplexity and toxicity across generation methods."""
        print("Performing statistical analysis...")
        results = []
        metrics = ['perplexity'] + self.toxicity_dimensions

        for metric in metrics:
            print(f"  Analyzing metric: {metric}")
            
            # Get data for each method
            method_data = [
                self.df[self.df['method'] == method][metric].dropna().values 
                for method in self.methods
            ]
            
            if any(len(data) < 3 for data in method_data):
                print(f"    Insufficient data for {metric}")
                continue

            # Test for normality
            is_normal = all(shapiro(data).pvalue > 0.05 for data in method_data if len(data) > 2)

            # Choose test
            if is_normal:
                test_used = "ANOVA"
                try:
                    stat, p_value = f_oneway(*method_data)
                except Exception as e:
                    print(f"    ANOVA failed: {e}")
                    stat, p_value = np.nan, 1.0
            else:
                test_used = "Kruskal-Wallis"
                try:
                    stat, p_value = kruskal(*method_data)
                except Exception as e:
                    print(f"    Kruskal-Wallis failed: {e}")
                    stat, p_value = np.nan, 1.0
            
            result = {
                'metric': metric,
                'test_used': test_used,
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            # Add mean scores
            for i, method in enumerate(self.methods):
                result[f'mean_{method}'] = np.mean(method_data[i])

            results.append(result)

        self.statistical_results = pd.DataFrame(results)
        return self.statistical_results

    def create_visualizations(self):
        """Create visualizations for fluency and toxicity analysis."""
        print("Creating visualizations...")

        # 1. Perplexity Distribution
        plt.figure(figsize=(10, 7))
        sns.boxplot(data=self.df, x='method', y='perplexity', order=self.methods)
        plt.title('Fluency (Perplexity) of Generated Text by Method', fontsize=16, fontweight='bold')
        plt.xlabel('Generation Method', fontsize=12, fontweight='bold')
        plt.ylabel('Perplexity (Lower is Better)', fontsize=12, fontweight='bold')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'perplexity_by_method.png'), dpi=300)
        plt.close()

        # 2. Toxicity Distributions
        for dim in self.toxicity_dimensions:
            plt.figure(figsize=(10, 7))
            sns.boxplot(data=self.df, x='method', y=dim, order=self.methods)
            title_dim = dim.replace("generated_", "").replace("_", " ").title()
            plt.title(f'{title_dim} Scores by Generation Method', fontsize=16, fontweight='bold')
            plt.xlabel('Generation Method', fontsize=12, fontweight='bold')
            plt.ylabel('Toxicity Score', fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{dim}_by_method.png'), dpi=300)
            plt.close()
            
        # 3. Statistical Significance Heatmap
        if not self.statistical_results.empty:
            pivot_data = self.statistical_results.pivot(index='metric', columns='test_used', values='p_value')
            pivot_data.index.name = 'Metric'
            pivot_data = pivot_data.sort_index()

            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='Blues_r',
                        linewidths=0.5, cbar_kws={'label': 'p-value'})
            plt.title('P-values for Differences Across Generation Methods', fontsize=16, fontweight='bold')
            plt.xlabel('Statistical Test Used', fontsize=12, fontweight='bold')
            plt.ylabel('Metric', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'statistical_significance_heatmap.png'), dpi=300)
            plt.close()

        print("Visualizations saved successfully.")
        
    def save_results(self):
        """Save all analysis results to files."""
        print("Saving results...")
        
        # Save dataframe with perplexity
        df_file = os.path.join(self.output_dir, 'generation_method_analysis_data.csv')
        self.df.to_csv(df_file, index=False)
        
        # Save statistical results
        stats_file = os.path.join(self.output_dir, 'statistical_analysis_results.csv')
        self.statistical_results.to_csv(stats_file, index=False)
        
        # Save summary report
        summary_file = os.path.join(self.output_dir, 'generation_method_analysis_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("CODE-SWITCHING GENERATION METHOD ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {self.input_file}\n")
            f.write(f"Total rows analyzed: {len(self.df)}\n\n")
            
            f.write("RESEARCH QUESTIONS:\n")
            f.write("1. How effective are code-switching generation methods at creating toxic content?\n")
            f.write("2. How do generation methods compare in terms of fluency (perplexity)?\n\n")

            f.write("KEY FINDINGS (STATISTICAL SIGNIFICANCE p < 0.05):\n")
            f.write("-" * 40 + "\n")
            if not self.statistical_results.empty:
                for _, row in self.statistical_results.iterrows():
                    metric_name = row['metric'].replace("_", " ").title()
                    sig_text = "YES" if row['significant'] else "NO"
                    f.write(f"- Significant difference in '{metric_name}' across methods? {sig_text} (p={row['p_value']:.4f})\n")

            f.write("\nMEAN SCORES BY METHOD:\n")
            f.write("-" * 40 + "\n")
            summary_stats = self.df.groupby('method')[['perplexity'] + self.toxicity_dimensions].mean()
            f.write(summary_stats.to_string(float_format="%.4f"))
            
            lowest_perplexity_method = summary_stats['perplexity'].idxmin()
            highest_toxicity_method = summary_stats['generated_toxicity'].idxmax()
            
            f.write(f"\n\n- Best fluency (lowest perplexity): '{lowest_perplexity_method}'")
            f.write(f"\n- Highest average toxicity: '{highest_toxicity_method}'\n")

            f.write(f"\nGENERATED FILES:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  - {os.path.basename(df_file)}\n")
            f.write(f"  - {os.path.basename(stats_file)}\n")
            f.write(f"  - perplexity_by_method.png\n")
            for dim in self.toxicity_dimensions:
                f.write(f"  - {dim}_by_method.png\n")
            f.write(f"  - statistical_significance_heatmap.png\n")

        print(f"Results saved to {self.output_dir}")

    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting Code-Switching Generation Method Evaluation...")
        print("=" * 50)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Results will be saved to: {self.output_dir}")
        
        self.login_huggingface()
        
        if not self.load_data():
            print("Failed to load data. Exiting.")
            return False
            
        if not self.calculate_perplexity():
            print("Failed to calculate perplexity. Exiting.")
            return False
            
        self.perform_statistical_analysis()
        self.create_visualizations()
        self.save_results()
        
        print("\nAnalysis completed successfully!")
        return True

def main():
    parser = argparse.ArgumentParser(description='Evaluate Code-Switching Generation Methods')
    parser.add_argument('--input_file', type=str, default='final_outputs/perspective_analysis.csv',
                       help='Path to the perspective analysis CSV file')
    parser.add_argument('--output_dir', type=str, default='final_outputs',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        return 1
    
    evaluator = CodeSwitchingEvaluator(args.input_file, args.output_dir)
    
    if evaluator.run_analysis():
        return 0
    else:
        print("Analysis failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 