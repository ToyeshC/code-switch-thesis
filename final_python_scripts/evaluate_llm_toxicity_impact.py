#!/usr/bin/env python3
"""
Impact of Prompt Language on LLM-Generated Toxicity Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal, f_oneway, shapiro, ttest_ind
import argparse
import os
from datetime import datetime
import warnings
from itertools import combinations

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set matplotlib backend to avoid display issues on cluster
import matplotlib
matplotlib.use('Agg')

# Set professional color scheme
plt.style.use('default')
sns.set_style("whitegrid")
professional_colors = ['#2E86AB', '#A8DADC', '#457B9D', '#1D3557', '#A2E4B8', '#52B69A']
sns.set_palette(professional_colors)

class LLMToxicityAnalyzer:
    """Class to analyze LLM toxicity impact across prompt languages"""
    
    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_dir = os.path.join(output_dir, "experiment_b")
        self.df = None
        
        # Perspective API toxicity dimensions
        self.toxicity_dimensions = [
            'toxicity', 'severe_toxicity', 'identity_attack', 
            'insult', 'profanity', 'threat'
        ]
        
        # LLM models
        self.models = ['llama3', 'llama31', 'aya']
        
        # Prompt types (corresponding to text types)
        self.prompt_types = {
            'English Prompt': 'src',
            'Hindi Prompt': 'tgt', 
            'Code-switched Prompt': 'generated'
        }
        
        # Results storage
        self.statistical_results = pd.DataFrame()
        self.distribution_stats = pd.DataFrame()
        self.model_comparison_results = pd.DataFrame()
        
    def load_data(self):
        """Load the perspective analysis CSV file"""
        print(f"Loading data from {self.input_file}...")
        try:
            self.df = pd.read_csv(self.input_file)
            print(f"Loaded {len(self.df)} rows with {len(self.df.columns)} columns")
            
            # Check for required continuation columns
            required_cols = []
            for model in self.models:
                for prompt_type in self.prompt_types.values():
                    for dim in self.toxicity_dimensions:
                        col_name = f"{model}_{prompt_type}_continuation_{dim}"
                        required_cols.append(col_name)
            
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                print(f"Warning: Missing columns: {missing_cols[:5]}...")  # Show first 5
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def prepare_data(self):
        """Prepare data for analysis by extracting LLM continuation toxicity scores"""
        print("Preparing data for analysis...")
        
        # Create comprehensive dataset with all LLM continuation toxicity scores
        analysis_data = []
        
        for index, row in self.df.iterrows():
            for model in self.models:
                for prompt_type_name, prompt_type_col in self.prompt_types.items():
                    for dimension in self.toxicity_dimensions:
                        col_name = f"{model}_{prompt_type_col}_continuation_{dimension}"
                        if col_name in self.df.columns:
                            score = row[col_name]
                            if pd.notna(score) and isinstance(score, (int, float)):
                                analysis_data.append({
                                    'model': model,
                                    'prompt_type': prompt_type_name,
                                    'dimension': dimension,
                                    'score': float(score),
                                    'primary_key': row.get('primary_key', index),
                                    'method': row.get('method', 'unknown')
                                })
        
        self.analysis_df = pd.DataFrame(analysis_data)
        print(f"Prepared {len(self.analysis_df)} data points for analysis")
        print(f"Models found: {self.analysis_df['model'].unique()}")
        print(f"Prompt types found: {self.analysis_df['prompt_type'].unique()}")
        print(f"Dimensions found: {self.analysis_df['dimension'].unique()}")
        return len(self.analysis_df) > 0
    
    def compute_descriptive_statistics(self):
        """Compute descriptive statistics for each model, prompt type, and dimension"""
        print("Computing descriptive statistics...")
        
        # Group by model, prompt type, and dimension
        grouped = self.analysis_df.groupby(['model', 'prompt_type', 'dimension'])['score']
        
        stats_summary = []
        for (model, prompt_type, dimension), group in grouped:
            stats_dict = {
                'model': model,
                'prompt_type': prompt_type,
                'dimension': dimension,
                'count': len(group),
                'mean': group.mean(),
                'std': group.std(),
                'median': group.median(),
                'min': group.min(),
                'max': group.max(),
                'q25': group.quantile(0.25),
                'q75': group.quantile(0.75),
                'skewness': stats.skew(group),
                'kurtosis': stats.kurtosis(group)
            }
            stats_summary.append(stats_dict)
        
        self.distribution_stats = pd.DataFrame(stats_summary)
        return self.distribution_stats
    
    def test_normality(self, data):
        """Test for normality using Shapiro-Wilk test"""
        if len(data) < 3:
            return False, 1.0
        
        if len(data) > 5000:
            sample_data = np.random.choice(data, 5000, replace=False)
        else:
            sample_data = data
            
        try:
            stat, p_value = shapiro(sample_data)
            return p_value > 0.05, p_value
        except:
            return False, 0.0
    
    def perform_prompt_language_analysis(self):
        """Analyze toxicity differences across prompt languages for each model"""
        print("Performing prompt language impact analysis...")
        
        results = []
        
        for model in self.models:
            for dimension in self.toxicity_dimensions:
                print(f"  Testing {model} - {dimension}")
                
                # Get data for each prompt type for this model and dimension
                model_dim_data = self.analysis_df[
                    (self.analysis_df['model'] == model) & 
                    (self.analysis_df['dimension'] == dimension)
                ]
                
                if len(model_dim_data) == 0:
                    continue
                
                english_data = model_dim_data[model_dim_data['prompt_type'] == 'English Prompt']['score'].values
                hindi_data = model_dim_data[model_dim_data['prompt_type'] == 'Hindi Prompt']['score'].values
                cs_data = model_dim_data[model_dim_data['prompt_type'] == 'Code-switched Prompt']['score'].values
                
                print(f"    Sample sizes - English: {len(english_data)}, Hindi: {len(hindi_data)}, Code-switched: {len(cs_data)}")
                
                if len(english_data) < 3 or len(hindi_data) < 3 or len(cs_data) < 3:
                    print(f"    Insufficient data for {model} - {dimension}")
                    continue
                
                # Test for normality
                eng_normal, _ = self.test_normality(english_data)
                hin_normal, _ = self.test_normality(hindi_data)
                cs_normal, _ = self.test_normality(cs_data)
                
                all_normal = eng_normal and hin_normal and cs_normal
                
                # Choose appropriate test
                if all_normal:
                    try:
                        f_stat, p_anova = f_oneway(english_data, hindi_data, cs_data)
                        test_used = "ANOVA"
                        test_statistic = f_stat
                        p_value = p_anova
                        print(f"    Using ANOVA: F={f_stat:.4f}, p={p_anova:.6f}")
                    except Exception as e:
                        print(f"    ANOVA failed: {e}")
                        test_used = "ANOVA_failed"
                        test_statistic = np.nan
                        p_value = 1.0
                else:
                    try:
                        h_stat, p_kruskal = kruskal(english_data, hindi_data, cs_data)
                        test_used = "Kruskal-Wallis"
                        test_statistic = h_stat
                        p_value = p_kruskal
                        print(f"    Using Kruskal-Wallis: H={h_stat:.4f}, p={p_kruskal:.6f}")
                    except Exception as e:
                        print(f"    Kruskal-Wallis failed: {e}")
                        test_used = "Kruskal-Wallis_failed"
                        test_statistic = np.nan
                        p_value = 1.0
                
                # Store results
                result = {
                    'model': model,
                    'dimension': dimension,
                    'test_used': test_used,
                    'test_statistic': test_statistic,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'n_english': len(english_data),
                    'n_hindi': len(hindi_data),
                    'n_codeswitched': len(cs_data),
                    'mean_english': np.mean(english_data),
                    'mean_hindi': np.mean(hindi_data),
                    'mean_codeswitched': np.mean(cs_data),
                    'std_english': np.std(english_data),
                    'std_hindi': np.std(hindi_data),
                    'std_codeswitched': np.std(cs_data)
                }
                results.append(result)
        
        self.statistical_results = pd.DataFrame(results)
        return self.statistical_results
    
    def perform_model_comparison_analysis(self):
        """Compare toxicity across models for each prompt type and dimension"""
        print("Performing model comparison analysis...")
        
        results = []
        
        for prompt_type in self.prompt_types.keys():
            for dimension in self.toxicity_dimensions:
                print(f"  Testing {prompt_type} - {dimension}")
                
                # Get data for each model for this prompt type and dimension
                prompt_dim_data = self.analysis_df[
                    (self.analysis_df['prompt_type'] == prompt_type) & 
                    (self.analysis_df['dimension'] == dimension)
                ]
                
                if len(prompt_dim_data) == 0:
                    continue
                
                model_data = {}
                for model in self.models:
                    model_data[model] = prompt_dim_data[prompt_dim_data['model'] == model]['score'].values
                
                # Check if we have sufficient data
                valid_models = [model for model, data in model_data.items() if len(data) >= 3]
                
                if len(valid_models) < 2:
                    print(f"    Insufficient data for {prompt_type} - {dimension}")
                    continue
                
                print(f"    Sample sizes - " + ", ".join([f"{model}: {len(model_data[model])}" for model in valid_models]))
                
                # Test for normality
                normality_results = {}
                for model in valid_models:
                    normality_results[model], _ = self.test_normality(model_data[model])
                
                all_normal = all(normality_results.values())
                
                # Choose appropriate test
                if len(valid_models) == 2:
                    # T-test for two models
                    model1, model2 = valid_models
                    if all_normal:
                        try:
                            t_stat, p_value = ttest_ind(model_data[model1], model_data[model2])
                            test_used = "T-test"
                            test_statistic = t_stat
                        except:
                            test_used = "T-test_failed"
                            test_statistic = np.nan
                            p_value = 1.0
                    else:
                        try:
                            u_stat, p_value = stats.mannwhitneyu(model_data[model1], model_data[model2])
                            test_used = "Mann-Whitney U"
                            test_statistic = u_stat
                        except:
                            test_used = "Mann-Whitney_failed"
                            test_statistic = np.nan
                            p_value = 1.0
                else:
                    # ANOVA or Kruskal-Wallis for multiple models
                    if all_normal:
                        try:
                            f_stat, p_value = f_oneway(*[model_data[model] for model in valid_models])
                            test_used = "ANOVA"
                            test_statistic = f_stat
                        except:
                            test_used = "ANOVA_failed"
                            test_statistic = np.nan
                            p_value = 1.0
                    else:
                        try:
                            h_stat, p_value = kruskal(*[model_data[model] for model in valid_models])
                            test_used = "Kruskal-Wallis"
                            test_statistic = h_stat
                        except:
                            test_used = "Kruskal-Wallis_failed"
                            test_statistic = np.nan
                            p_value = 1.0
                
                # Store results
                result = {
                    'prompt_type': prompt_type,
                    'dimension': dimension,
                    'test_used': test_used,
                    'test_statistic': test_statistic,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'models_tested': ', '.join(valid_models)
                }
                
                # Add means for each model
                for model in self.models:
                    if model in valid_models:
                        result[f'mean_{model}'] = np.mean(model_data[model])
                        result[f'std_{model}'] = np.std(model_data[model])
                        result[f'n_{model}'] = len(model_data[model])
                    else:
                        result[f'mean_{model}'] = np.nan
                        result[f'std_{model}'] = np.nan
                        result[f'n_{model}'] = 0
                
                results.append(result)
        
        self.model_comparison_results = pd.DataFrame(results)
        return self.model_comparison_results
    
    def create_visualizations(self):
        """Create comprehensive visualizations of LLM toxicity analysis"""
        print("Creating visualizations...")
        
        # Color schemes
        prompt_colors = ['#52B69A', '#457B9D', '#2E86AB']  # Green to blue for prompt types
        model_colors = ['#1D3557', '#457B9D', '#A8DADC']   # Dark to light blue for models
        
        # 1. Model-wise toxicity comparison across prompt types
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        axes = axes.flatten()
        
        for i, dimension in enumerate(self.toxicity_dimensions):
            dimension_data = self.analysis_df[self.analysis_df['dimension'] == dimension]
            
            if not dimension_data.empty:
                sns.boxplot(data=dimension_data, x='model', y='score', hue='prompt_type', 
                           ax=axes[i], palette=prompt_colors)
                axes[i].set_title(f'{dimension.replace("_", " ").title()}\nToxicity by Model and Prompt Type', 
                                fontsize=14, fontweight='bold', pad=20)
                axes[i].set_xlabel('LLM Model', fontsize=12, fontweight='bold')
                axes[i].set_ylabel('Toxicity Score', fontsize=12, fontweight='bold')
                axes[i].tick_params(axis='x', labelsize=11, labelrotation=0)
                axes[i].grid(True, alpha=0.3)
                axes[i].legend(title='Prompt Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.suptitle('LLM Toxicity Comparison: Impact of Prompt Language', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'llm_toxicity_by_prompt_type.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # 2. Mean toxicity heatmap by model and prompt type
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, dimension in enumerate(self.toxicity_dimensions):
            dimension_data = self.analysis_df[self.analysis_df['dimension'] == dimension]
            
            if not dimension_data.empty:
                pivot_data = dimension_data.groupby(['model', 'prompt_type'])['score'].mean().unstack()
                
                sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='Blues', ax=axes[i],
                           cbar_kws={'label': 'Mean Toxicity'}, linewidths=0.5)
                axes[i].set_title(f'{dimension.replace("_", " ").title()}', 
                                fontsize=14, fontweight='bold')
                axes[i].set_xlabel('Prompt Type', fontsize=12, fontweight='bold')
                axes[i].set_ylabel('LLM Model', fontsize=12, fontweight='bold')
                axes[i].tick_params(axis='x', labelrotation=45)
        
        plt.suptitle('Mean Toxicity Heatmaps by Model and Prompt Type', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'toxicity_heatmaps_by_model.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # 3. Statistical significance summary for prompt language impact
        if not self.statistical_results.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Significance heatmap
            sig_pivot = self.statistical_results.pivot(index='model', columns='dimension', values='significant')
            sig_pivot = sig_pivot.astype(int)  # Convert boolean to int for heatmap
            
            sns.heatmap(sig_pivot, annot=True, fmt='d', cmap='RdYlBu_r', ax=ax1,
                       cbar_kws={'label': 'Significant (1=Yes, 0=No)'}, 
                       linewidths=0.5, vmin=0, vmax=1)
            ax1.set_title('Statistical Significance: Prompt Language Impact', 
                         fontsize=14, fontweight='bold')
            ax1.set_xlabel('Toxicity Dimension', fontsize=12, fontweight='bold')
            ax1.set_ylabel('LLM Model', fontsize=12, fontweight='bold')
            ax1.tick_params(axis='x', labelrotation=45)
            
            # P-values by model and dimension
            p_value_pivot = self.statistical_results.pivot(index='model', columns='dimension', values='p_value')
            
            sns.heatmap(p_value_pivot, annot=True, fmt='.3f', cmap='Blues_r', ax=ax2,
                       cbar_kws={'label': 'p-value'}, linewidths=0.5)
            ax2.set_title('P-values: Prompt Language Impact', 
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel('Toxicity Dimension', fontsize=12, fontweight='bold')
            ax2.set_ylabel('LLM Model', fontsize=12, fontweight='bold')
            ax2.tick_params(axis='x', labelrotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'statistical_significance_prompt_impact.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
        # 4. Model comparison across prompt types
        if not self.model_comparison_results.empty:
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            
            prompt_types_list = list(self.prompt_types.keys())
            
            for i, prompt_type in enumerate(prompt_types_list):
                prompt_data = self.model_comparison_results[
                    self.model_comparison_results['prompt_type'] == prompt_type
                ]
                
                if not prompt_data.empty:
                    # Create bar plot of p-values
                    colors = ['#1D3557' if p < 0.05 else '#457B9D' for p in prompt_data['p_value']]
                    bars = axes[i].bar(range(len(prompt_data)), prompt_data['p_value'], 
                                     color=colors, alpha=0.7, edgecolor='black', linewidth=1)
                    
                    axes[i].axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
                    axes[i].set_xlabel('Toxicity Dimension', fontsize=12, fontweight='bold')
                    axes[i].set_ylabel('p-value', fontsize=12, fontweight='bold')
                    axes[i].set_title(f'Model Differences: {prompt_type}', fontsize=14, fontweight='bold')
                    axes[i].set_xticks(range(len(prompt_data)))
                    axes[i].set_xticklabels([dim.replace('_', ' ').title() for dim in prompt_data['dimension']], 
                                          rotation=45, ha='right')
                    axes[i].set_yscale('log')
                    axes[i].grid(True, alpha=0.3)
                    axes[i].legend()
                    
                    # Add significance annotations
                    for j, (bar, p_val) in enumerate(zip(bars, prompt_data['p_value'])):
                        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5, 
                                   significance, ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            plt.suptitle('Statistical Significance: Model Comparisons by Prompt Type', 
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'model_comparison_significance.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
        # 5. Overall mean comparison
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Mean toxicity by prompt type (aggregated across models)
        prompt_means = self.analysis_df.groupby(['prompt_type', 'dimension'])['score'].mean().unstack()
        
        prompt_means.plot(kind='bar', ax=axes[0], color=prompt_colors, alpha=0.8)
        axes[0].set_title('Mean Toxicity by Prompt Type\n(Aggregated across all models)', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Prompt Type', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Mean Toxicity Score', fontsize=12, fontweight='bold')
        axes[0].tick_params(axis='x', labelrotation=45)
        axes[0].legend(title='Toxicity Dimension', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Mean toxicity by model (aggregated across prompt types)
        model_means = self.analysis_df.groupby(['model', 'dimension'])['score'].mean().unstack()
        
        model_means.plot(kind='bar', ax=axes[1], color=model_colors, alpha=0.8)
        axes[1].set_title('Mean Toxicity by Model\n(Aggregated across all prompt types)', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('LLM Model', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Mean Toxicity Score', fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', labelrotation=0)
        axes[1].legend(title='Toxicity Dimension', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'overall_mean_comparisons.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("Enhanced visualizations saved successfully!")
    
    def save_results(self):
        """Save all analysis results to files"""
        print("Saving results...")
        
        # Save descriptive statistics
        desc_stats_file = os.path.join(self.output_dir, 'llm_descriptive_statistics.csv')
        self.distribution_stats.to_csv(desc_stats_file, index=False)
        
        # Save prompt language impact results
        prompt_results_file = None
        if not self.statistical_results.empty:
            prompt_results_file = os.path.join(self.output_dir, 'prompt_language_impact_results.csv')
            self.statistical_results.to_csv(prompt_results_file, index=False)
        
        # Save model comparison results
        model_results_file = None
        if not self.model_comparison_results.empty:
            model_results_file = os.path.join(self.output_dir, 'model_comparison_results.csv')
            self.model_comparison_results.to_csv(model_results_file, index=False)
        
        # Save comprehensive summary report
        summary_file = os.path.join(self.output_dir, 'llm_toxicity_analysis_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("LLM TOXICITY IMPACT ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {self.input_file}\n")
            f.write(f"Total data points analyzed: {len(self.analysis_df)}\n\n")
            
            f.write("RESEARCH QUESTION:\n")
            f.write("How does code-switching affect LLM-generated toxicity?\n\n")
            
            f.write("PROMPT LANGUAGE IMPACT RESULTS:\n")
            f.write("-" * 40 + "\n")
            if not self.statistical_results.empty:
                significant_results = self.statistical_results[self.statistical_results['significant']]
                f.write(f"Models with significant prompt language effects: {len(significant_results)}/{len(self.statistical_results)}\n\n")
                
                for model in self.models:
                    model_results = self.statistical_results[self.statistical_results['model'] == model]
                    if not model_results.empty:
                        f.write(f"\n{model.upper()} Results:\n")
                        for _, row in model_results.iterrows():
                            f.write(f"  {row['dimension']}:\n")
                            f.write(f"    Test: {row['test_used']}\n")
                            f.write(f"    p-value: {row['p_value']:.6f}\n")
                            f.write(f"    Significant: {'Yes' if row['significant'] else 'No'}\n")
                            f.write(f"    Means - English: {row['mean_english']:.4f}, Hindi: {row['mean_hindi']:.4f}, Code-switched: {row['mean_codeswitched']:.4f}\n")
            
            f.write(f"\nMODEL COMPARISON RESULTS:\n")
            f.write("-" * 40 + "\n")
            if not self.model_comparison_results.empty:
                significant_model_diffs = self.model_comparison_results[self.model_comparison_results['significant']]
                f.write(f"Significant model differences found: {len(significant_model_diffs)}/{len(self.model_comparison_results)}\n\n")
                
                for prompt_type in self.prompt_types.keys():
                    prompt_results = self.model_comparison_results[
                        self.model_comparison_results['prompt_type'] == prompt_type
                    ]
                    if not prompt_results.empty:
                        f.write(f"\n{prompt_type} Results:\n")
                        for _, row in prompt_results.iterrows():
                            f.write(f"  {row['dimension']}:\n")
                            f.write(f"    Test: {row['test_used']}\n")
                            f.write(f"    p-value: {row['p_value']:.6f}\n")
                            f.write(f"    Significant: {'Yes' if row['significant'] else 'No'}\n")
                            f.write(f"    Models tested: {row['models_tested']}\n")
            
            f.write(f"\nKEY FINDINGS:\n")
            f.write("-" * 40 + "\n")
            
            # Calculate summary statistics
            if not self.statistical_results.empty:
                prompt_significant = self.statistical_results['significant'].sum()
                prompt_total = len(self.statistical_results)
                f.write(f"- Prompt language significantly affects toxicity in {prompt_significant}/{prompt_total} ({100*prompt_significant/prompt_total:.1f}%) of cases\n")
                
                # Find highest toxicity by prompt type
                overall_means = self.analysis_df.groupby('prompt_type')['score'].mean()
                highest_prompt = overall_means.idxmax()
                f.write(f"- Highest overall toxicity from: {highest_prompt} (mean: {overall_means.max():.4f})\n")
                
                # Find most vulnerable model
                model_means = self.analysis_df.groupby('model')['score'].mean()
                most_vulnerable_model = model_means.idxmax()
                f.write(f"- Most vulnerable model: {most_vulnerable_model} (mean: {model_means.max():.4f})\n")
            
            if not self.model_comparison_results.empty:
                model_significant = self.model_comparison_results['significant'].sum()
                model_total = len(self.model_comparison_results)
                f.write(f"- Significant model differences found in {model_significant}/{model_total} ({100*model_significant/model_total:.1f}%) of cases\n")
            
            f.write(f"\nGENERATED VISUALIZATIONS:\n")
            f.write("-" * 40 + "\n")
            f.write("  - llm_toxicity_by_prompt_type.png\n")
            f.write("  - toxicity_heatmaps_by_model.png\n")
            f.write("  - statistical_significance_prompt_impact.png\n")
            f.write("  - model_comparison_significance.png\n")
            f.write("  - overall_mean_comparisons.png\n")
        
        print(f"Results saved to {self.output_dir}")
        return desc_stats_file, prompt_results_file, model_results_file, summary_file
    
    def run_analysis(self):
        """Run the complete LLM toxicity analysis pipeline"""
        print("Starting LLM Toxicity Impact Analysis...")
        print("=" * 50)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Results will be saved to: {self.output_dir}")
        
        # Step 1: Load data
        if not self.load_data():
            print("Failed to load data. Exiting.")
            return False
        
        # Step 2: Prepare data
        if not self.prepare_data():
            print("Failed to prepare data. Exiting.")
            return False
        
        # Step 3: Compute descriptive statistics
        self.compute_descriptive_statistics()
        
        # Step 4: Analyze prompt language impact
        self.perform_prompt_language_analysis()
        
        # Step 5: Analyze model comparisons
        self.perform_model_comparison_analysis()
        
        # Step 6: Create visualizations
        self.create_visualizations()
        
        # Step 7: Save results
        self.save_results()
        
        print("\nLLM Toxicity Impact Analysis completed successfully!")
        print(f"Check the results in: {self.output_dir}")
        return True

def main():
    parser = argparse.ArgumentParser(description='Analyze Impact of Prompt Language on LLM-Generated Toxicity')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to the perspective analysis CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        return 1
    
    # Create analyzer and run analysis
    analyzer = LLMToxicityAnalyzer(args.input_file, args.output_dir)
    
    if analyzer.run_analysis():
        print(f"\nAnalysis results saved to: {analyzer.output_dir}")
        return 0
    else:
        print("Analysis failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 