#!/usr/bin/env python3
"""
Perspective API Robustness Analysis on Code-Switched Text
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal, f_oneway, shapiro
import argparse
import os
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set matplotlib backend to avoid display issues on cluster
import matplotlib
matplotlib.use('Agg')

# Set style for better visualizations
plt.style.use('default')
sns.set_style("whitegrid")
# Professional blue-green color palette
professional_colors = ['#2E86AB', '#A8DADC', '#457B9D', '#1D3557', '#A2E4B8', '#52B69A']
sns.set_palette(professional_colors)

class PerspectiveRobustnessAnalyzer:
    """Class to analyze Perspective API robustness across text types"""
    
    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_dir = os.path.join(output_dir, "experiment_a")
        self.df = None
        
        # Perspective API toxicity dimensions
        self.toxicity_dimensions = [
            'toxicity', 'severe_toxicity', 'identity_attack', 
            'insult', 'profanity', 'threat'
        ]
        
        # Text types for comparison
        self.text_types = {
            'English': 'src',
            'Hindi': 'tgt', 
            'Code-switched': 'generated'
        }
        
        # Results storage
        self.statistical_results = pd.DataFrame()
        self.distribution_stats = pd.DataFrame()
        
    def load_data(self):
        """Load the perspective analysis CSV file"""
        print(f"Loading data from {self.input_file}...")
        try:
            self.df = pd.read_csv(self.input_file)
            print(f"Loaded {len(self.df)} rows with {len(self.df.columns)} columns")
            print("Columns:", list(self.df.columns))
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def prepare_data(self):
        """Prepare data for analysis by extracting toxicity scores"""
        print("Preparing data for analysis...")
        
        # Create a comprehensive dataset with all toxicity scores
        analysis_data = []
        
        for index, row in self.df.iterrows():
            for text_type_name, text_type_col in self.text_types.items():
                for dimension in self.toxicity_dimensions:
                    col_name = f"{text_type_col}_{dimension}"
                    if col_name in self.df.columns:
                        score = row[col_name]
                        if pd.notna(score) and isinstance(score, (int, float)):
                            analysis_data.append({
                                'text_type': text_type_name,
                                'dimension': dimension,
                                'score': float(score),
                                'primary_key': row.get('primary_key', index),
                                'model': row.get('model', 'unknown'),
                                'method': row.get('method', 'unknown')
                            })
        
        self.analysis_df = pd.DataFrame(analysis_data)
        print(f"Prepared {len(self.analysis_df)} data points for analysis")
        print(f"Text types found: {self.analysis_df['text_type'].unique()}")
        print(f"Dimensions found: {self.analysis_df['dimension'].unique()}")
        return len(self.analysis_df) > 0
    
    def compute_descriptive_statistics(self):
        """Compute descriptive statistics for each text type and dimension"""
        print("Computing descriptive statistics...")
        
        # Group by text type and dimension
        grouped = self.analysis_df.groupby(['text_type', 'dimension'])['score']
        
        stats_summary = []
        for (text_type, dimension), group in grouped:
            stats_dict = {
                'text_type': text_type,
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
        
        # Shapiro-Wilk test works best for small samples (< 5000)
        if len(data) > 5000:
            sample_data = np.random.choice(data, 5000, replace=False)
        else:
            sample_data = data
            
        try:
            stat, p_value = shapiro(sample_data)
            return p_value > 0.05, p_value
        except:
            return False, 0.0
    
    def perform_statistical_tests(self):
        """Perform statistical tests to compare distributions across text types"""
        print("Performing statistical tests...")
        
        results = []
        
        for dimension in self.toxicity_dimensions:
            print(f"  Testing dimension: {dimension}")
            
            # Get data for each text type
            english_data = self.analysis_df[
                (self.analysis_df['text_type'] == 'English') & 
                (self.analysis_df['dimension'] == dimension)
            ]['score'].values
            
            hindi_data = self.analysis_df[
                (self.analysis_df['text_type'] == 'Hindi') & 
                (self.analysis_df['dimension'] == dimension)
            ]['score'].values
            
            codeswitched_data = self.analysis_df[
                (self.analysis_df['text_type'] == 'Code-switched') & 
                (self.analysis_df['dimension'] == dimension)
            ]['score'].values
            
            print(f"    Sample sizes - English: {len(english_data)}, Hindi: {len(hindi_data)}, Code-switched: {len(codeswitched_data)}")
            
            if len(english_data) < 3 or len(hindi_data) < 3 or len(codeswitched_data) < 3:
                print(f"    Insufficient data for {dimension}")
                continue
            
            # Test for normality
            eng_normal, eng_p_norm = self.test_normality(english_data)
            hin_normal, hin_p_norm = self.test_normality(hindi_data)
            cs_normal, cs_p_norm = self.test_normality(codeswitched_data)
            
            all_normal = eng_normal and hin_normal and cs_normal
            
            # Choose appropriate test based on normality
            if all_normal:
                # Use ANOVA for normal distributions
                try:
                    f_stat, p_anova = f_oneway(english_data, hindi_data, codeswitched_data)
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
                # Use Kruskal-Wallis for non-normal distributions
                try:
                    h_stat, p_kruskal = kruskal(english_data, hindi_data, codeswitched_data)
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
                'dimension': dimension,
                'test_used': test_used,
                'test_statistic': test_statistic,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'n_english': len(english_data),
                'n_hindi': len(hindi_data),
                'n_codeswitched': len(codeswitched_data),
                'mean_english': np.mean(english_data),
                'mean_hindi': np.mean(hindi_data),
                'mean_codeswitched': np.mean(codeswitched_data),
                'std_english': np.std(english_data),
                'std_hindi': np.std(hindi_data),
                'std_codeswitched': np.std(codeswitched_data),
                'english_normal': eng_normal,
                'hindi_normal': hin_normal,
                'codeswitched_normal': cs_normal
            }
            results.append(result)
        
        self.statistical_results = pd.DataFrame(results)
        return self.statistical_results
    
    def create_visualizations(self):
        """Create comprehensive visualizations of the toxicity distributions"""
        print("Creating visualizations...")
        
        # 1. Distribution comparison - Box plots with enhanced styling
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        axes = axes.flatten()
        
        colors = ['#A8DADC', '#457B9D', '#2E86AB']  # Professional light to dark blue
        
        for i, dimension in enumerate(self.toxicity_dimensions):
            dimension_data = self.analysis_df[self.analysis_df['dimension'] == dimension]
            
            if not dimension_data.empty:
                box_plot = sns.boxplot(data=dimension_data, x='text_type', y='score', ax=axes[i], palette=colors)
                axes[i].set_title(f'{dimension.replace("_", " ").title()}\nToxicity Score Distribution', 
                                fontsize=14, fontweight='bold', pad=20)
                axes[i].set_xlabel('Text Type', fontsize=12, fontweight='bold')
                axes[i].set_ylabel('Toxicity Score', fontsize=12, fontweight='bold')
                axes[i].tick_params(axis='x', rotation=0, labelsize=11)
                axes[i].tick_params(axis='y', labelsize=10)
                axes[i].grid(True, alpha=0.3)
                
                # Add sample size annotations
                text_types = dimension_data['text_type'].unique()
                for j, text_type in enumerate(text_types):
                    n = len(dimension_data[dimension_data['text_type'] == text_type])
                    axes[i].text(j, axes[i].get_ylim()[1] * 0.95, f'n={n}', 
                               ha='center', va='top', fontsize=10, fontweight='bold')
        
        plt.suptitle('Perspective API Toxicity Score Distributions by Text Type', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'toxicity_distributions_boxplot.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # 2. Mean comparison - Bar plot with error bars
        plt.figure(figsize=(15, 10))
        
        # Calculate means and standard errors for each combination
        summary_stats = self.analysis_df.groupby(['text_type', 'dimension'])['score'].agg(['mean', 'std', 'count']).reset_index()
        summary_stats['se'] = summary_stats['std'] / np.sqrt(summary_stats['count'])
        
        # Create bar plot
        bar_width = 0.25
        dimensions = self.toxicity_dimensions
        text_types = ['English', 'Hindi', 'Code-switched']
        
        x = np.arange(len(dimensions))
        colors = ['#52B69A', '#457B9D', '#2E86AB']  # Professional green to blue gradient
        
        for i, text_type in enumerate(text_types):
            means = []
            errors = []
            for dim in dimensions:
                row = summary_stats[(summary_stats['text_type'] == text_type) & 
                                  (summary_stats['dimension'] == dim)]
                if not row.empty:
                    means.append(row['mean'].iloc[0])
                    errors.append(row['se'].iloc[0])
                else:
                    means.append(0)
                    errors.append(0)
            
            plt.bar(x + i * bar_width, means, bar_width, label=text_type, 
                   color=colors[i], alpha=0.8, yerr=errors, capsize=5)
        
        plt.xlabel('Toxicity Dimensions', fontsize=14, fontweight='bold')
        plt.ylabel('Mean Toxicity Score', fontsize=14, fontweight='bold')
        plt.title('Mean Toxicity Scores by Text Type and Dimension', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(x + bar_width, [dim.replace('_', ' ').title() for dim in dimensions], rotation=45, ha='right')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'mean_toxicity_comparison.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # 3. Enhanced heatmap
        try:
            pivot_means = self.analysis_df.groupby(['text_type', 'dimension'])['score'].mean().unstack()
            
            plt.figure(figsize=(12, 8))
            mask = pivot_means.isnull()
            sns.heatmap(pivot_means, annot=True, fmt='.4f', cmap='Blues', 
                       cbar_kws={'label': 'Mean Toxicity Score'}, mask=mask,
                       linewidths=0.5, square=True, cbar=True)
            plt.title('Toxicity Score Heatmap: Text Types vs Dimensions', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Toxicity Dimension', fontsize=14, fontweight='bold')
            plt.ylabel('Text Type', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'toxicity_heatmap.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        except Exception as e:
            print(f"Could not create heatmap: {e}")
        
        # 4. Statistical significance visualization
        if not self.statistical_results.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # P-values bar chart
            colors = ['#1D3557' if p < 0.05 else '#457B9D' for p in self.statistical_results['p_value']]  # Dark blue for significant, medium blue for non-significant
            bars = ax1.bar(range(len(self.statistical_results)), self.statistical_results['p_value'], 
                          color=colors, alpha=0.7, edgecolor='black', linewidth=1)
            
            ax1.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
            ax1.set_xlabel('Toxicity Dimension', fontsize=14, fontweight='bold')
            ax1.set_ylabel('p-value', fontsize=14, fontweight='bold')
            ax1.set_title('Statistical Significance by Dimension', fontsize=14, fontweight='bold')
            ax1.set_xticks(range(len(self.statistical_results)))
            ax1.set_xticklabels([dim.replace('_', ' ').title() for dim in self.statistical_results['dimension']], 
                              rotation=45, ha='right')
            ax1.legend(fontsize=12)
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
            
            # Add significance annotations
            for i, (bar, p_val, test) in enumerate(zip(bars, self.statistical_results['p_value'], 
                                                      self.statistical_results['test_used'])):
                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5, 
                        significance, ha='center', va='bottom', fontsize=12, fontweight='bold')
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.1, 
                        test.split('-')[0], ha='center', va='bottom', fontsize=8, rotation=45)
            
            # Effect sizes (mean differences) visualization
            dimensions = self.statistical_results['dimension']
            english_means = self.statistical_results['mean_english']
            hindi_means = self.statistical_results['mean_hindi']
            cs_means = self.statistical_results['mean_codeswitched']
            
            x = np.arange(len(dimensions))
            width = 0.25
            
            ax2.bar(x - width, english_means, width, label='English', color='#52B69A', alpha=0.8)
            ax2.bar(x, hindi_means, width, label='Hindi', color='#457B9D', alpha=0.8)
            ax2.bar(x + width, cs_means, width, label='Code-switched', color='#2E86AB', alpha=0.8)
            
            ax2.set_xlabel('Toxicity Dimension', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Mean Toxicity Score', fontsize=14, fontweight='bold')
            ax2.set_title('Mean Toxicity Scores by Text Type', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels([dim.replace('_', ' ').title() for dim in dimensions], 
                              rotation=45, ha='right')
            ax2.legend(fontsize=12)
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'statistical_analysis_summary.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
        # 5. Distribution shapes comparison (violin plot)
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        axes = axes.flatten()
        
        for i, dimension in enumerate(self.toxicity_dimensions):
            dimension_data = self.analysis_df[self.analysis_df['dimension'] == dimension]
            
            if not dimension_data.empty:
                try:
                    sns.violinplot(data=dimension_data, x='text_type', y='score', ax=axes[i], 
                                 palette=['#52B69A', '#457B9D', '#2E86AB'], inner='box')
                    axes[i].set_title(f'{dimension.replace("_", " ").title()}\nDistribution Shape', 
                                    fontsize=14, fontweight='bold', pad=20)
                    axes[i].set_xlabel('Text Type', fontsize=12, fontweight='bold')
                    axes[i].set_ylabel('Toxicity Score', fontsize=12, fontweight='bold')
                    axes[i].tick_params(axis='x', rotation=0, labelsize=11)
                    axes[i].grid(True, alpha=0.3)
                except:
                    # Fallback to box plot if violin plot fails
                    sns.boxplot(data=dimension_data, x='text_type', y='score', ax=axes[i], palette=['#52B69A', '#457B9D', '#2E86AB'])
                    axes[i].set_title(f'{dimension.replace("_", " ").title()}\nDistribution', 
                                    fontsize=14, fontweight='bold', pad=20)
                    axes[i].set_xlabel('Text Type', fontsize=12, fontweight='bold')
                    axes[i].set_ylabel('Toxicity Score', fontsize=12, fontweight='bold')
                    axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Toxicity Score Distribution Shapes by Text Type', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'distribution_shapes_violin.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("Enhanced visualizations saved successfully!")
    
    def save_results(self):
        """Save all analysis results to CSV files"""
        print("Saving results...")
        
        # Save descriptive statistics (no timestamp)
        desc_stats_file = os.path.join(self.output_dir, 'descriptive_statistics.csv')
        self.distribution_stats.to_csv(desc_stats_file, index=False)
        
        # Save statistical test results (no timestamp)
        stats_file = None
        if not self.statistical_results.empty:
            stats_file = os.path.join(self.output_dir, 'statistical_test_results.csv')
            self.statistical_results.to_csv(stats_file, index=False)
        
        # Save summary report (no timestamp)
        summary_file = os.path.join(self.output_dir, 'analysis_summary_report.txt')
        with open(summary_file, 'w') as f:
            f.write("PERSPECTIVE API ROBUSTNESS ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {self.input_file}\n")
            f.write(f"Total data points analyzed: {len(self.analysis_df)}\n\n")
            
            f.write("STATISTICAL TEST RESULTS:\n")
            f.write("-" * 30 + "\n")
            if not self.statistical_results.empty:
                for _, row in self.statistical_results.iterrows():
                    f.write(f"\nDimension: {row['dimension']}\n")
                    f.write(f"  Test used: {row['test_used']}\n")
                    f.write(f"  Test statistic: {row['test_statistic']:.6f}\n")
                    f.write(f"  p-value: {row['p_value']:.6f}\n")
                    f.write(f"  Significant: {'Yes' if row['significant'] else 'No'}\n")
                    f.write(f"  Sample sizes - English: {row['n_english']}, Hindi: {row['n_hindi']}, Code-switched: {row['n_codeswitched']}\n")
                    f.write(f"  Mean scores - English: {row['mean_english']:.4f}, Hindi: {row['mean_hindi']:.4f}, Code-switched: {row['mean_codeswitched']:.4f}\n")
            
            f.write(f"\nSIGNIFICANT DIFFERENCES FOUND:\n")
            f.write("-" * 30 + "\n")
            if not self.statistical_results.empty:
                significant_dims = self.statistical_results[self.statistical_results['significant']]['dimension'].tolist()
                if significant_dims:
                    for dim in significant_dims:
                        f.write(f"  - {dim}\n")
                else:
                    f.write("  No significant differences found\n")
            else:
                f.write("  No statistical tests performed\n")
            
            # Add visualization files list
            f.write(f"\nGENERATED VISUALIZATIONS:\n")
            f.write("-" * 30 + "\n")
            f.write("  - toxicity_distributions_boxplot.png\n")
            f.write("  - mean_toxicity_comparison.png\n")
            f.write("  - toxicity_heatmap.png\n")
            f.write("  - statistical_analysis_summary.png\n")
            f.write("  - distribution_shapes_violin.png\n")
        
        print(f"Results saved to {self.output_dir}")
        return desc_stats_file, stats_file, summary_file
    
    def run_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting Perspective API Robustness Analysis...")
        print("=" * 50)
        
        # Create output directory if it doesn't exist (including experiment_a subfolder)
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
        
        # Step 4: Perform statistical tests
        self.perform_statistical_tests()
        
        # Step 5: Create visualizations
        self.create_visualizations()
        
        # Step 6: Save results
        self.save_results()
        
        print("\nAnalysis completed successfully!")
        print(f"Check the results in: {self.output_dir}")
        return True

def main():
    parser = argparse.ArgumentParser(description='Evaluate Perspective API Robustness on Code-Switched Text')
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
    analyzer = PerspectiveRobustnessAnalyzer(args.input_file, args.output_dir)
    
    if analyzer.run_analysis():
        print(f"\nAnalysis results saved to: {analyzer.output_dir}")
        return 0
    else:
        print("Analysis failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 