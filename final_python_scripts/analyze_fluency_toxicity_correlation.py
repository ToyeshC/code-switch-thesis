#!/usr/bin/env python3
"""
Analyzing Fluency of Code-Switched Content and its Correlation with Toxicity
Research Question 4: How does code-switching affect the fluency of LLM-generated text, 
and how does this fluency correlate with toxicity?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal, f_oneway, shapiro, ttest_ind, mannwhitneyu, pearsonr, spearmanr
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

# Set professional color scheme (matching existing scripts)
plt.style.use('default')
sns.set_style("whitegrid")
professional_colors = ['#2E86AB', '#A8DADC', '#457B9D', '#1D3557', '#A2E4B8', '#52B69A']
sns.set_palette(professional_colors)

class FluencyToxicityAnalyzer:
    """Class to analyze fluency and toxicity correlation in code-switched content"""
    
    def __init__(self, perspective_file, hinge_file, output_dir):
        self.perspective_file = perspective_file
        self.hinge_file = hinge_file
        self.output_dir = os.path.join(output_dir, "experiment_d")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Data storage
        self.perspective_df = None
        self.hinge_df = None
        
        # Perspective API toxicity dimensions
        self.toxicity_dimensions = [
            'toxicity', 'severe_toxicity', 'identity_attack', 
            'insult', 'profanity', 'threat'
        ]
        
        # LLM models
        self.models = ['llama3', 'llama31', 'aya']
        
        # Text types for analysis
        self.text_types = {
            'English (src)': 'src',
            'Hindi (tgt)': 'tgt', 
            'Code-switched (generated)': 'generated'
        }
        
        # Results storage
        self.perplexity_results = pd.DataFrame()
        self.statistical_results = pd.DataFrame()
        self.correlation_results = pd.DataFrame()
        
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

    def create_output_directory(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

    def load_and_preprocess_data(self):
        """Load and preprocess both HINGE and perspective analysis datasets"""
        print("Loading datasets...")
        
        # Load perspective analysis data
        self.perspective_df = pd.read_csv(self.perspective_file)
        print(f"Loaded perspective analysis data: {len(self.perspective_df)} rows")
        
        # Load HINGE data
        hinge_raw = pd.read_csv(self.hinge_file)
        print(f"Loaded HINGE data: {len(hinge_raw)} rows")
        
        # Preprocess HINGE data to match perspective analysis format
        self.hinge_df = self._preprocess_hinge_data(hinge_raw)
        
        # Save preprocessed HINGE data
        hinge_processed_file = os.path.join(self.output_dir, "hinge_preprocessed.csv")
        self.hinge_df.to_csv(hinge_processed_file, index=False)
        print(f"Saved preprocessed HINGE data to: {hinge_processed_file}")
        
        return True

    def _preprocess_hinge_data(self, hinge_raw):
        """Preprocess HINGE data to match the perspective analysis format"""
        print("Preprocessing HINGE data to match perspective analysis format...")
        
        # Rename columns to match perspective analysis format
        hinge_processed = pd.DataFrame()
        hinge_processed['src'] = hinge_raw['English']
        hinge_processed['tgt'] = hinge_raw['Hindi'] 
        hinge_processed['generated'] = hinge_raw['Hinglish']
        hinge_processed['method'] = 'hinge_baseline'
        hinge_processed['model'] = 'human_annotated'
        hinge_processed['direction'] = 'en_to_hi_cs'
        hinge_processed['primary_key'] = [f"hinge_{i:06d}" for i in range(len(hinge_raw))]
        hinge_processed['average_rating'] = hinge_raw['Average rating']
        hinge_processed['disagreement'] = hinge_raw['Disagreement']
        
        # Calculate language statistics for generated (Hinglish) text
        print("Calculating language statistics for HINGE data...")
        lang_stats = self._calculate_language_statistics(hinge_processed['generated'])
        hinge_processed = pd.concat([hinge_processed, lang_stats], axis=1)
        
        # Initialize toxicity columns as NaN (to be filled later if needed)
        for text_type in ['src', 'tgt', 'generated']:
            for dimension in self.toxicity_dimensions:
                hinge_processed[f"{text_type}_{dimension}"] = np.nan
        
        # Initialize continuation columns as empty strings
        for model in self.models:
            for text_type in ['src', 'tgt', 'generated']:
                hinge_processed[f"{model}_{text_type}_continuation"] = ""
                for dimension in self.toxicity_dimensions:
                    hinge_processed[f"{model}_{text_type}_continuation_{dimension}"] = np.nan
        
        print(f"Preprocessed HINGE data: {len(hinge_processed)} rows")
        return hinge_processed

    def _calculate_language_statistics(self, texts):
        """Calculate language mixing statistics for texts"""
        stats_data = []
        
        for text in texts:
            if pd.isna(text) or not isinstance(text, str):
                stats_data.append({
                    'hindi_word_count': 0,
                    'english_word_count': 0,
                    'romanized_hindi_count': 0,
                    'total_hindi_count': 0,
                    'total_words': 0,
                    'hindi_percent': 0.0,
                    'romanized_hindi_percent': 0.0,
                    'total_hindi_percent': 0.0,
                    'english_percent': 0.0
                })
                continue
            
            words = text.split()
            total_words = len(words)
            
            # Simple heuristic for language detection
            hindi_words = sum(1 for word in words if any(ord(char) > 127 for char in word))
            english_words = total_words - hindi_words
            
            # For Hinglish, assume romanized Hindi is roughly half of non-Devanagari words
            romanized_hindi = max(0, english_words // 2) if english_words > 0 else 0
            english_words = max(0, english_words - romanized_hindi)
            
            total_hindi = hindi_words + romanized_hindi
            
            stats_data.append({
                'hindi_word_count': hindi_words,
                'english_word_count': english_words,
                'romanized_hindi_count': romanized_hindi,
                'total_hindi_count': total_hindi,
                'total_words': total_words,
                'hindi_percent': (hindi_words / total_words * 100) if total_words > 0 else 0,
                'romanized_hindi_percent': (romanized_hindi / total_words * 100) if total_words > 0 else 0,
                'total_hindi_percent': (total_hindi / total_words * 100) if total_words > 0 else 0,
                'english_percent': (english_words / total_words * 100) if total_words > 0 else 0
            })
        
        return pd.DataFrame(stats_data)

    def calculate_perplexity_scores(self):
        """Calculate perplexity scores using mT5-XL for all text types"""
        print(f"Loading mT5-XL model on {self.device}...")
        
        model_name = 'google/mt5-xl'
        try:
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
            model.eval()
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return False

        # Combine datasets for perplexity calculation
        all_data = []
        
        # Add perspective analysis data
        for _, row in self.perspective_df.iterrows():
            for text_type_name, text_col in self.text_types.items():
                if pd.notna(row[text_col]) and isinstance(row[text_col], str):
                    all_data.append({
                        'text': row[text_col],
                        'source': 'perspective_analysis',
                        'text_type': text_type_name,
                        'primary_key': row.get('primary_key', 'unknown'),
                        'method': row.get('method', 'unknown'),
                        'model': row.get('model', 'unknown')
                    })
        
        # Add HINGE data
        for _, row in self.hinge_df.iterrows():
            for text_type_name, text_col in self.text_types.items():
                if pd.notna(row[text_col]) and isinstance(row[text_col], str):
                    all_data.append({
                        'text': row[text_col],
                        'source': 'hinge',
                        'text_type': text_type_name,
                        'primary_key': row.get('primary_key', 'unknown'),
                        'method': row.get('method', 'hinge_baseline'),
                        'model': row.get('model', 'human_annotated')
                    })
        
        print(f"Calculating perplexity for {len(all_data)} texts...")
        
        # Calculate perplexity scores
        perplexity_scores = []
        with torch.no_grad():
            for i, item in enumerate(all_data):
                if i % 100 == 0:
                    print(f"  Processing text {i+1}/{len(all_data)}...")
                
                text = item['text']
                try:
                    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(self.device)
                    labels = inputs['input_ids']
                    outputs = model(**inputs, labels=labels)
                    loss = outputs.loss
                    perplexity = torch.exp(loss).item()
                    perplexity_scores.append(perplexity)
                except Exception as e:
                    print(f"Could not compute perplexity for text {i}: {e}")
                    perplexity_scores.append(np.nan)
        
        # Create results dataframe
        for i, score in enumerate(perplexity_scores):
            all_data[i]['perplexity'] = score
        
        self.perplexity_results = pd.DataFrame(all_data)
        print("Perplexity calculation complete.")
        return True

    def analyze_monolingual_vs_codeswitched_fluency(self):
        """Compare perplexity between monolingual and code-switched content"""
        print("Analyzing monolingual vs code-switched fluency...")
        
        # Get perplexity scores for different text types
        english_perp = self.perplexity_results[
            self.perplexity_results['text_type'] == 'English (src)'
        ]['perplexity'].dropna()
        
        hindi_perp = self.perplexity_results[
            self.perplexity_results['text_type'] == 'Hindi (tgt)'
        ]['perplexity'].dropna()
        
        cs_perp = self.perplexity_results[
            self.perplexity_results['text_type'] == 'Code-switched (generated)'
        ]['perplexity'].dropna()
        
        print(f"Sample sizes - English: {len(english_perp)}, Hindi: {len(hindi_perp)}, Code-switched: {len(cs_perp)}")
        
        # Statistical tests
        results = []
        
        if len(english_perp) >= 3 and len(cs_perp) >= 3:
            # English vs Code-switched
            eng_normal = shapiro(english_perp[:5000] if len(english_perp) > 5000 else english_perp).pvalue > 0.05
            cs_normal = shapiro(cs_perp[:5000] if len(cs_perp) > 5000 else cs_perp).pvalue > 0.05
            
            if eng_normal and cs_normal:
                stat, p_val = ttest_ind(english_perp, cs_perp)
                test_used = "Independent t-test"
            else:
                stat, p_val = mannwhitneyu(english_perp, cs_perp)
                test_used = "Mann-Whitney U"
            
            results.append({
                'comparison': 'English vs Code-switched',
                'test_used': test_used,
                'statistic': stat,
                'p_value': p_val,
                'significant': p_val < 0.05,
                'mean_english': np.mean(english_perp),
                'mean_cs': np.mean(cs_perp),
                'effect_size': (np.mean(cs_perp) - np.mean(english_perp)) / np.sqrt((np.var(english_perp) + np.var(cs_perp)) / 2)
            })
        
        if len(hindi_perp) >= 3 and len(cs_perp) >= 3:
            # Hindi vs Code-switched
            hin_normal = shapiro(hindi_perp[:5000] if len(hindi_perp) > 5000 else hindi_perp).pvalue > 0.05
            cs_normal = shapiro(cs_perp[:5000] if len(cs_perp) > 5000 else cs_perp).pvalue > 0.05
            
            if hin_normal and cs_normal:
                stat, p_val = ttest_ind(hindi_perp, cs_perp)
                test_used = "Independent t-test"
            else:
                stat, p_val = mannwhitneyu(hindi_perp, cs_perp)
                test_used = "Mann-Whitney U"
            
            results.append({
                'comparison': 'Hindi vs Code-switched',
                'test_used': test_used,
                'statistic': stat,
                'p_value': p_val,
                'significant': p_val < 0.05,
                'mean_hindi': np.mean(hindi_perp),
                'mean_cs': np.mean(cs_perp),
                'effect_size': (np.mean(cs_perp) - np.mean(hindi_perp)) / np.sqrt((np.var(hindi_perp) + np.var(cs_perp)) / 2)
            })
        
        if len(english_perp) >= 3 and len(hindi_perp) >= 3 and len(cs_perp) >= 3:
            # ANOVA across all three
            eng_normal = shapiro(english_perp[:5000] if len(english_perp) > 5000 else english_perp).pvalue > 0.05
            hin_normal = shapiro(hindi_perp[:5000] if len(hindi_perp) > 5000 else hindi_perp).pvalue > 0.05
            cs_normal = shapiro(cs_perp[:5000] if len(cs_perp) > 5000 else cs_perp).pvalue > 0.05
            
            if eng_normal and hin_normal and cs_normal:
                stat, p_val = f_oneway(english_perp, hindi_perp, cs_perp)
                test_used = "One-way ANOVA"
            else:
                stat, p_val = kruskal(english_perp, hindi_perp, cs_perp)
                test_used = "Kruskal-Wallis"
            
            results.append({
                'comparison': 'English vs Hindi vs Code-switched',
                'test_used': test_used,
                'statistic': stat,
                'p_value': p_val,
                'significant': p_val < 0.05,
                'mean_english': np.mean(english_perp),
                'mean_hindi': np.mean(hindi_perp),
                'mean_cs': np.mean(cs_perp)
            })
        
        self.statistical_results = pd.DataFrame(results)
        return self.statistical_results

    def analyze_model_specific_fluency(self):
        """Analyze fluency of LLM continuations by model and prompt type"""
        print("Analyzing model-specific fluency on continuations...")
        
        # Get continuation data from perspective analysis
        continuation_data = []
        
        for _, row in self.perspective_df.iterrows():
            for model in self.models:
                for text_type_name, text_col in self.text_types.items():
                    cont_col = f"{model}_{text_col}_continuation"
                    if cont_col in row and pd.notna(row[cont_col]) and isinstance(row[cont_col], str):
                        continuation_data.append({
                            'text': row[cont_col],
                            'model': model,
                            'prompt_type': text_type_name,
                            'primary_key': row.get('primary_key', 'unknown'),
                            'method': row.get('method', 'unknown')
                        })
        
        if not continuation_data:
            print("No continuation data found")
            return pd.DataFrame()
        
        print(f"Calculating perplexity for {len(continuation_data)} continuations...")
        
        # Load model for continuation analysis
        model_name = 'google/mt5-xl'
        try:
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
            model.eval()
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return pd.DataFrame()
        
        # Calculate perplexity for continuations
        with torch.no_grad():
            for i, item in enumerate(continuation_data):
                if i % 50 == 0:
                    print(f"  Processing continuation {i+1}/{len(continuation_data)}...")
                
                text = item['text']
                try:
                    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(self.device)
                    labels = inputs['input_ids']
                    outputs = model(**inputs, labels=labels)
                    loss = outputs.loss
                    perplexity = torch.exp(loss).item()
                    item['perplexity'] = perplexity
                except Exception as e:
                    print(f"Could not compute perplexity for continuation {i}: {e}")
                    item['perplexity'] = np.nan
        
        continuation_df = pd.DataFrame(continuation_data)
        
        # Combine with main perplexity results
        continuation_df['source'] = 'continuations'
        continuation_df['text_type'] = continuation_df['prompt_type'] + ' (continuation)'
        
        self.perplexity_results = pd.concat([self.perplexity_results, continuation_df], ignore_index=True)
        
        return continuation_df

    def analyze_correlations(self):
        """Analyze correlations between fluency, language mix, and toxicity"""
        print("Analyzing correlations between fluency, language composition, and toxicity...")
        
        correlation_data = []
        
        # Merge perplexity with language statistics and toxicity scores
        for _, row in self.perspective_df.iterrows():
            primary_key = row.get('primary_key', 'unknown')
            
            # Get perplexity scores for this item
            for text_type_name, text_col in self.text_types.items():
                perp_row = self.perplexity_results[
                    (self.perplexity_results['primary_key'] == primary_key) &
                    (self.perplexity_results['text_type'] == text_type_name) &
                    (self.perplexity_results['source'] == 'perspective_analysis')
                ]
                
                if len(perp_row) > 0 and pd.notna(perp_row.iloc[0]['perplexity']):
                    perplexity = perp_row.iloc[0]['perplexity']
                    
                    # Get language composition metrics
                    if text_col == 'generated':  # Code-switched text
                        lang_metrics = {
                            'hindi_percent': row.get('hindi_percent', np.nan),
                            'english_percent': row.get('english_percent', np.nan),
                            'romanized_hindi_percent': row.get('romanized_hindi_percent', np.nan),
                            'total_hindi_percent': row.get('total_hindi_percent', np.nan)
                        }
                    else:
                        # For monolingual texts, set appropriate percentages
                        if text_col == 'src':  # English
                            lang_metrics = {
                                'hindi_percent': 0.0,
                                'english_percent': 100.0,
                                'romanized_hindi_percent': 0.0,
                                'total_hindi_percent': 0.0
                            }
                        else:  # Hindi
                            lang_metrics = {
                                'hindi_percent': 100.0,
                                'english_percent': 0.0,
                                'romanized_hindi_percent': 0.0,
                                'total_hindi_percent': 100.0
                            }
                    
                    # Get toxicity scores
                    toxicity_scores = {}
                    for dimension in self.toxicity_dimensions:
                        toxicity_col = f"{text_col}_{dimension}"
                        if toxicity_col in row:
                            toxicity_scores[dimension] = row[toxicity_col]
                    
                    # Combine all metrics
                    data_point = {
                        'primary_key': primary_key,
                        'text_type': text_type_name,
                        'perplexity': perplexity,
                        'method': row.get('method', 'unknown'),
                        'model': row.get('model', 'unknown'),
                        **lang_metrics,
                        **toxicity_scores
                    }
                    
                    correlation_data.append(data_point)
        
        if not correlation_data:
            print("No correlation data available")
            return pd.DataFrame()
        
        correlation_df = pd.DataFrame(correlation_data)
        
        # Calculate correlations
        correlation_results = []
        
        # Variables to correlate
        fluency_vars = ['perplexity']
        language_vars = ['hindi_percent', 'english_percent', 'romanized_hindi_percent', 'total_hindi_percent']
        toxicity_vars = [dim for dim in self.toxicity_dimensions if dim in correlation_df.columns]
        
        all_vars = fluency_vars + language_vars + toxicity_vars
        
        # Calculate pairwise correlations
        for i, var1 in enumerate(all_vars):
            for var2 in all_vars[i+1:]:
                if var1 in correlation_df.columns and var2 in correlation_df.columns:
                    data1 = correlation_df[var1].dropna()
                    data2 = correlation_df[var2].dropna()
                    
                    # Get overlapping indices
                    common_idx = data1.index.intersection(data2.index)
                    if len(common_idx) >= 10:  # Minimum sample size
                        data1_common = data1[common_idx]
                        data2_common = data2[common_idx]
                        
                        # Pearson correlation
                        try:
                            pearson_r, pearson_p = pearsonr(data1_common, data2_common)
                        except:
                            pearson_r, pearson_p = np.nan, np.nan
                        
                        # Spearman correlation
                        try:
                            spearman_r, spearman_p = spearmanr(data1_common, data2_common)
                        except:
                            spearman_r, spearman_p = np.nan, np.nan
                        
                        correlation_results.append({
                            'variable_1': var1,
                            'variable_2': var2,
                            'n_samples': len(common_idx),
                            'pearson_r': pearson_r,
                            'pearson_p': pearson_p,
                            'spearman_r': spearman_r,
                            'spearman_p': spearman_p,
                            'pearson_significant': pearson_p < 0.05 if not np.isnan(pearson_p) else False,
                            'spearman_significant': spearman_p < 0.05 if not np.isnan(spearman_p) else False
                        })
        
        self.correlation_results = pd.DataFrame(correlation_results)
        return correlation_df

    def create_visualizations(self):
        """Create comprehensive visualizations with proper outlier handling"""
        print("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_style("whitegrid")
        sns.set_palette(professional_colors)
        
        # Helper function to set reasonable y-axis limits based on percentiles
        def set_perplexity_ylimits(ax, data, percentile=95):
            """Set y-axis limits to exclude extreme outliers"""
            valid_data = data.dropna()
            if len(valid_data) > 0:
                upper_limit = np.percentile(valid_data, percentile)
                lower_limit = max(0, np.percentile(valid_data, 5))  # Don't go below 0
                
                # Add some padding
                y_range = upper_limit - lower_limit
                padding = y_range * 0.1
                ax.set_ylim(lower_limit - padding, upper_limit + padding)
                
                # Add note about outliers
                outliers_count = np.sum(valid_data > upper_limit)
                outliers_pct = (outliers_count / len(valid_data)) * 100
                if outliers_count > 0:
                    ax.text(0.02, 0.98, f'Note: {outliers_count} outliers ({outliers_pct:.1f}%) excluded from view', 
                           transform=ax.transAxes, verticalalignment='top', fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 1. Perplexity Distribution by Text Type
        plt.figure(figsize=(12, 8))
        
        if not self.perplexity_results.empty:
            # Filter for main text types (not continuations)
            main_perp = self.perplexity_results[
                ~self.perplexity_results['text_type'].str.contains('continuation', na=False)
            ]
            
            if not main_perp.empty:
                # Print perplexity statistics for debugging
                print(f"Perplexity statistics:")
                for text_type in main_perp['text_type'].unique():
                    subset = main_perp[main_perp['text_type'] == text_type]['perplexity'].dropna()
                    if len(subset) > 0:
                        print(f"  {text_type}: n={len(subset)}, mean={np.mean(subset):.1f}, "
                              f"median={np.median(subset):.1f}, max={np.max(subset):.1f}, "
                              f"95th percentile={np.percentile(subset, 95):.1f}")
                
                ax = sns.boxplot(data=main_perp, x='text_type', y='perplexity', showfliers=True)
                
                # Set reasonable y-axis limits to focus on the main distribution
                set_perplexity_ylimits(ax, main_perp['perplexity'])
                
                plt.title('Perplexity Distribution by Text Type\n(Lower perplexity = Higher fluency)', 
                         fontsize=16, fontweight='bold')
                plt.xlabel('Text Type', fontsize=12)
                plt.ylabel('Perplexity Score', fontsize=12)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                perp_dist_file = os.path.join(self.output_dir, "perplexity_distribution.png")
                plt.savefig(perp_dist_file, dpi=300, bbox_inches='tight')
                plt.close()
        
        # 2. Model-specific Continuation Fluency
        if not self.perplexity_results.empty:
            continuation_data = self.perplexity_results[
                self.perplexity_results['text_type'].str.contains('continuation', na=False)
            ]
            
            if not continuation_data.empty:
                # Print continuation perplexity statistics
                print(f"Continuation perplexity statistics:")
                for model in continuation_data['model'].unique():
                    subset = continuation_data[continuation_data['model'] == model]['perplexity'].dropna()
                    if len(subset) > 0:
                        print(f"  {model}: n={len(subset)}, mean={np.mean(subset):.1f}, "
                              f"median={np.median(subset):.1f}, max={np.max(subset):.1f}, "
                              f"95th percentile={np.percentile(subset, 95):.1f}")
                
                # Create subplot for each model
                models_available = continuation_data['model'].unique()
                n_models = len(models_available)
                
                if n_models > 0:
                    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 8), sharey=True)
                    if n_models == 1:
                        axes = [axes]
                    
                    # Calculate global y-limits for consistent scaling
                    all_perp_values = continuation_data['perplexity'].dropna()
                    
                    for i, model in enumerate(models_available):
                        model_data = continuation_data[continuation_data['model'] == model]
                        
                        sns.boxplot(data=model_data, x='prompt_type', y='perplexity', ax=axes[i], showfliers=True)
                        
                        # Set consistent y-axis limits across all subplots
                        set_perplexity_ylimits(axes[i], all_perp_values)
                        
                        axes[i].set_title(f'{model.upper()} Continuations', fontweight='bold')
                        axes[i].set_xlabel('Prompt Type')
                        if i == 0:
                            axes[i].set_ylabel('Perplexity Score')
                        axes[i].tick_params(axis='x', rotation=45)
                    
                    plt.suptitle('Model-Specific Continuation Fluency Analysis', 
                               fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    
                    model_fluency_file = os.path.join(self.output_dir, "model_continuation_fluency.png")
                    plt.savefig(model_fluency_file, dpi=300, bbox_inches='tight')
                    plt.close()
        
        # 3. Create a log-scale version for full range visualization
        if not self.perplexity_results.empty:
            main_perp = self.perplexity_results[
                ~self.perplexity_results['text_type'].str.contains('continuation', na=False)
            ]
            
            if not main_perp.empty:
                plt.figure(figsize=(12, 8))
                
                # Filter out invalid values for log scale
                main_perp_log = main_perp[main_perp['perplexity'] > 0].copy()
                
                if not main_perp_log.empty:
                    ax = sns.boxplot(data=main_perp_log, x='text_type', y='perplexity', showfliers=True)
                    ax.set_yscale('log')
                    
                    plt.title('Perplexity Distribution by Text Type (Log Scale)\n(Shows full range including outliers)', 
                             fontsize=16, fontweight='bold')
                    plt.xlabel('Text Type', fontsize=12)
                    plt.ylabel('Perplexity Score (Log Scale)', fontsize=12)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    perp_log_file = os.path.join(self.output_dir, "perplexity_distribution_log_scale.png")
                    plt.savefig(perp_log_file, dpi=300, bbox_inches='tight')
                    plt.close()
        
        print("Visualizations created successfully.")

    def save_results(self):
        """Save all analysis results to CSV files"""
        print("Saving analysis results...")
        
        # Save perplexity results
        if not self.perplexity_results.empty:
            perp_file = os.path.join(self.output_dir, "perplexity_results.csv")
            self.perplexity_results.to_csv(perp_file, index=False)
            print(f"Saved perplexity results: {perp_file}")
        
        # Save statistical comparison results
        if not self.statistical_results.empty:
            stats_file = os.path.join(self.output_dir, "statistical_comparison_results.csv")
            self.statistical_results.to_csv(stats_file, index=False)
            print(f"Saved statistical results: {stats_file}")
        
        # Save correlation results
        if not self.correlation_results.empty:
            corr_file = os.path.join(self.output_dir, "correlation_analysis_results.csv")
            self.correlation_results.to_csv(corr_file, index=False)
            print(f"Saved correlation results: {corr_file}")
        
        # Create and save summary report
        self._create_summary_report()

    def _create_summary_report(self):
        """Create a comprehensive summary report"""
        report_file = os.path.join(self.output_dir, "experiment_d_summary_report.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("EXPERIMENT D: FLUENCY AND TOXICITY CORRELATION ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("RESEARCH QUESTION:\n")
            f.write("How does code-switching affect the fluency of LLM-generated text, and how does\n")
            f.write("this fluency correlate with toxicity?\n\n")
            
            f.write("DATASETS ANALYZED:\n")
            f.write(f"- Perspective Analysis Data: {len(self.perspective_df)} samples\n")
            f.write(f"- HINGE Dataset: {len(self.hinge_df)} samples\n\n")
            
            if not self.perplexity_results.empty:
                f.write("PERPLEXITY ANALYSIS SUMMARY:\n")
                f.write("-" * 40 + "\n")
                
                # Group by text type and show statistics
                perp_summary = self.perplexity_results.groupby('text_type')['perplexity'].agg([
                    'count', 'mean', 'std', 'median', 'min', 'max'
                ]).round(3)
                
                f.write("Perplexity Statistics by Text Type:\n")
                f.write(perp_summary.to_string())
                f.write("\n\n")
            
            if not self.statistical_results.empty:
                f.write("STATISTICAL COMPARISON RESULTS:\n")
                f.write("-" * 40 + "\n")
                
                for _, row in self.statistical_results.iterrows():
                    f.write(f"Comparison: {row['comparison']}\n")
                    f.write(f"  Test Used: {row['test_used']}\n")
                    f.write(f"  p-value: {row['p_value']:.6f}\n")
                    f.write(f"  Significant: {'Yes' if row['significant'] else 'No'}\n")
                    if 'effect_size' in row:
                        f.write(f"  Effect Size: {row['effect_size']:.3f}\n")
                    f.write("\n")
            
            if not self.correlation_results.empty:
                f.write("CORRELATION ANALYSIS SUMMARY:\n")
                f.write("-" * 40 + "\n")
                
                # Show significant correlations
                significant_corr = self.correlation_results[
                    (self.correlation_results['pearson_significant']) |
                    (self.correlation_results['spearman_significant'])
                ]
                
                if not significant_corr.empty:
                    f.write("Significant Correlations (p < 0.05):\n\n")
                    
                    for _, row in significant_corr.iterrows():
                        f.write(f"{row['variable_1']} ↔ {row['variable_2']}:\n")
                        f.write(f"  Pearson r = {row['pearson_r']:.3f} (p = {row['pearson_p']:.6f})\n")
                        f.write(f"  Spearman ρ = {row['spearman_r']:.3f} (p = {row['spearman_p']:.6f})\n")
                        f.write(f"  Sample size: {row['n_samples']}\n\n")
                else:
                    f.write("No significant correlations found.\n")
            
            f.write("\nFILES GENERATED:\n")
            f.write("-" * 20 + "\n")
            f.write("- hinge_preprocessed.csv: Preprocessed HINGE dataset\n")
            f.write("- perplexity_results.csv: Perplexity scores for all texts\n")
            f.write("- statistical_comparison_results.csv: Statistical test results\n")
            f.write("- correlation_analysis_results.csv: Correlation analysis results\n")
            f.write("- perplexity_distribution.png: Perplexity distribution (outlier-trimmed view)\n")
            f.write("- perplexity_distribution_log_scale.png: Perplexity distribution (full range, log scale)\n")
            f.write("- model_continuation_fluency.png: Model-specific fluency analysis\n")
            f.write("- experiment_d_summary_report.txt: This summary report\n")
        
        print(f"Summary report saved: {report_file}")

    def run_analysis(self):
        """Run the complete fluency-toxicity correlation analysis"""
        print("="*80)
        print("EXPERIMENT D: FLUENCY AND TOXICITY CORRELATION ANALYSIS")
        print("="*80)
        
        # Setup
        self.login_huggingface()
        self.create_output_directory()
        
        # Data loading and preprocessing
        if not self.load_and_preprocess_data():
            print("Error: Failed to load data")
            return False
        
        # Calculate perplexity scores
        if not self.calculate_perplexity_scores():
            print("Error: Failed to calculate perplexity scores")
            return False
        
        # Statistical analysis
        print("\nPerforming statistical analyses...")
        self.analyze_monolingual_vs_codeswitched_fluency()
        self.analyze_model_specific_fluency()
        self.analyze_correlations()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save results
        self.save_results()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print(f"Results saved to: {self.output_dir}")
        print("="*80)
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Analyze fluency and toxicity correlation in code-switched content')
    parser.add_argument('--perspective_file', type=str, default='final_outputs/perspective_analysis.csv',
                       help='Path to perspective analysis CSV file')
    parser.add_argument('--hinge_file', type=str, default='ezswitch/data/hinge/train.csv',
                       help='Path to HINGE dataset CSV file')
    parser.add_argument('--output_dir', type=str, default='final_outputs',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = FluencyToxicityAnalyzer(
        perspective_file=args.perspective_file,
        hinge_file=args.hinge_file,
        output_dir=args.output_dir
    )
    
    success = analyzer.run_analysis()
    
    if success:
        print("\nAnalysis completed successfully!")
    else:
        print("\nAnalysis failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 