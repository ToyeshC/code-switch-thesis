#!/usr/bin/env python3
"""
Human Evaluation Analysis for Code-Switched Text Naturalness and Toxicity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import argparse
import os
from datetime import datetime
import warnings
import re

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set matplotlib backend to avoid display issues on cluster
import matplotlib
matplotlib.use('Agg')

# Set professional color scheme (same as existing scripts)
plt.style.use('default')
sns.set_style("whitegrid")
professional_colors = ['#2E86AB', '#A8DADC', '#457B9D', '#1D3557', '#A2E4B8', '#52B69A']
sns.set_palette(professional_colors)

class HumanEvaluationAnalyzer:
    """Class to analyze human evaluation data for naturalness and toxicity"""
    
    def __init__(self, form_responses_file, perspective_file, output_dir):
        self.form_responses_file = form_responses_file
        self.perspective_file = perspective_file
        self.output_dir = os.path.join(output_dir, "experiment_g")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.form_data = None
        self.perspective_data = None
        self.human_evaluation_data = None
        
        # Results storage
        self.correlation_results = {}
        self.summary_stats = {}
        
    def load_data(self):
        """Load both the Google Form responses and perspective analysis data"""
        print(f"Loading Google Form responses from {self.form_responses_file}...")
        try:
            self.form_data = pd.read_csv(self.form_responses_file)
            print(f"Loaded {len(self.form_data)} form responses")
        except Exception as e:
            print(f"Error loading form data: {e}")
            return False
            
        print(f"Loading perspective analysis data from {self.perspective_file}...")
        try:
            self.perspective_data = pd.read_csv(self.perspective_file)
            print(f"Loaded {len(self.perspective_data)} perspective analysis entries")
        except Exception as e:
            print(f"Error loading perspective data: {e}")
            return False
            
        return True
    
    def parse_form_responses(self):
        """Parse the Google Form responses and extract ratings for each sentence"""
        print("Parsing Google Form responses...")
        
        # Extract column names - first column is timestamp, then groups of 4 columns for each sentence
        columns = self.form_data.columns.tolist()
        timestamp_col = columns[0]
        
        # Find the pattern of question columns (4 questions repeated 50 times)
        question_cols = []
        for i in range(1, len(columns)):
            if "How natural does the code-switched text sound?" in columns[i]:
                # This is the start of a new sentence group
                if i + 3 < len(columns):
                    question_cols.append({
                        'sentence_id': len(question_cols),
                        'naturalness_col': columns[i],
                        'toxicity_col': columns[i + 1], 
                        'comparison_col': columns[i + 2],
                        'toxic_words_col': columns[i + 3]
                    })
        
        print(f"Found {len(question_cols)} sentence groups")
        
        # Parse responses for each participant
        parsed_responses = []
        
        for idx, row in self.form_data.iterrows():
            participant_id = f"participant_{idx}"
            
            for sentence_group in question_cols:
                sentence_id = sentence_group['sentence_id']
                
                # Extract responses for the generated (code-switched) text
                naturalness = row[sentence_group['naturalness_col']]
                toxicity = row[sentence_group['toxicity_col']]
                comparison = row[sentence_group['comparison_col']]
                
                # Skip if responses are missing
                if pd.isna(naturalness) or pd.isna(toxicity):
                    continue
                
                parsed_responses.append({
                    'participant_id': participant_id,
                    'sentence_id': sentence_id,
                    'naturalness_rating': naturalness,
                    'toxicity_rating': toxicity,
                    'toxicity_comparison': comparison if not pd.isna(comparison) else 'cannot_compare'
                })
        
        self.human_evaluation_data = pd.DataFrame(parsed_responses)
        print(f"Parsed {len(self.human_evaluation_data)} individual evaluations")
        
        return len(self.human_evaluation_data) > 0
    
    def convert_ratings_to_scores(self):
        """Convert human ratings to 0-1 scores similar to Perspective API"""
        print("Converting ratings to 0-1 scores...")
        
        def naturalness_to_score(rating):
            """Convert naturalness rating to 0-1 score
            1-2 = not natural (0-0.33), 3 = slightly natural (0.34-0.66), 4-5 = natural (0.67-1.0)
            """
            if rating in [1, 2]:
                return np.random.uniform(0, 0.33)  # Not natural
            elif rating == 3:
                return np.random.uniform(0.34, 0.66)  # Slightly natural
            elif rating in [4, 5]:
                return np.random.uniform(0.67, 1.0)  # Natural
            else:
                return np.nan
        
        def toxicity_to_score(rating):
            """Convert toxicity rating to 0-1 score
            1-2 = not toxic (0-0.33), 3-4 = toxic (0.34-0.83), 5 = severely toxic (0.84-1.0)
            """
            if rating in [1, 2]:
                return np.random.uniform(0, 0.33)  # Not toxic
            elif rating in [3, 4]:
                return np.random.uniform(0.34, 0.83)  # Toxic
            elif rating == 5:
                return np.random.uniform(0.84, 1.0)  # Severely toxic
            else:
                return np.nan
        
        # Convert ratings to 0-1 scores
        self.human_evaluation_data['naturalness_score'] = self.human_evaluation_data['naturalness_rating'].apply(naturalness_to_score)
        self.human_evaluation_data['toxicity_score'] = self.human_evaluation_data['toxicity_rating'].apply(toxicity_to_score)
        
        # Aggregate by sentence (taking mean across participants)
        sentence_aggregated = self.human_evaluation_data.groupby('sentence_id').agg({
            'naturalness_rating': ['mean', 'std', 'count'],
            'toxicity_rating': ['mean', 'std', 'count'],
            'naturalness_score': ['mean', 'std'],
            'toxicity_score': ['mean', 'std'],
            'toxicity_comparison': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'cannot_compare'
        }).reset_index()
        
        # Flatten column names
        sentence_aggregated.columns = [
            'sentence_id', 'naturalness_mean', 'naturalness_std', 'naturalness_count',
            'toxicity_mean', 'toxicity_std', 'toxicity_count',
            'naturalness_score_mean', 'naturalness_score_std',
            'toxicity_score_mean', 'toxicity_score_std', 'toxicity_comparison_mode'
        ]
        
        self.sentence_level_data = sentence_aggregated
        print(f"Aggregated data for {len(self.sentence_level_data)} sentences")
        
        return True
    
    def merge_with_perspective_data(self):
        """Merge human evaluation data with perspective analysis data"""
        print("Merging human evaluation data with perspective analysis data...")
        
        # Reset index to match sentence_id (assuming perspective_data is in order)
        self.perspective_data['sentence_id'] = range(len(self.perspective_data))
        
        # Keep only the generated column and its toxicity metrics
        perspective_columns = ['sentence_id', 'generated', 'generated_toxicity', 
                             'generated_severe_toxicity', 'generated_identity_attack',
                             'generated_insult', 'generated_profanity', 'generated_threat']
        
        perspective_subset = self.perspective_data[perspective_columns].copy()
        
        # Perspective scores are already in 0-1 format, no conversion needed
        
        # Merge datasets
        self.merged_data = pd.merge(
            self.sentence_level_data,
            perspective_subset,
            on='sentence_id',
            how='inner'
        )
        
        print(f"Merged data contains {len(self.merged_data)} sentences")
        return len(self.merged_data) > 0
    
    def compute_correlations(self):
        """Compute correlations between human judgments and automated metrics"""
        print("Computing correlations between human judgments and automated metrics...")
        
        # Define metrics to correlate
        human_metrics = ['naturalness_score_mean', 'toxicity_score_mean']
        automated_metrics = ['generated_toxicity', 'generated_severe_toxicity', 
                           'generated_identity_attack', 'generated_insult',
                           'generated_profanity', 'generated_threat']
        
        # Compute correlations between human metrics and automated metrics
        for human_metric in human_metrics:
            for auto_metric in automated_metrics:
                # Pearson correlation
                pearson_corr, pearson_p = pearsonr(
                    self.merged_data[human_metric],
                    self.merged_data[auto_metric]
                )
                
                # Spearman correlation
                spearman_corr, spearman_p = spearmanr(
                    self.merged_data[human_metric],
                    self.merged_data[auto_metric]
                )
                
                # Store results
                key = f"{human_metric}_vs_{auto_metric}"
                self.correlation_results[key] = {
                    'pearson_correlation': pearson_corr,
                    'pearson_p_value': pearson_p,
                    'spearman_correlation': spearman_corr,
                    'spearman_p_value': spearman_p
                }
        
        # Compute correlation between human naturalness and human toxicity
        naturalness_toxicity_pearson, naturalness_toxicity_p_pearson = pearsonr(
            self.merged_data['naturalness_score_mean'],
            self.merged_data['toxicity_score_mean']
        )
        
        naturalness_toxicity_spearman, naturalness_toxicity_p_spearman = spearmanr(
            self.merged_data['naturalness_score_mean'],
            self.merged_data['toxicity_score_mean']
        )
        
        # Store human-human correlation
        self.correlation_results['naturalness_score_mean_vs_toxicity_score_mean'] = {
            'pearson_correlation': naturalness_toxicity_pearson,
            'pearson_p_value': naturalness_toxicity_p_pearson,
            'spearman_correlation': naturalness_toxicity_spearman,
            'spearman_p_value': naturalness_toxicity_p_spearman
        }
        
        print("Correlation analysis complete")
        return True
    
    def analyze_comparative_toxicity(self):
        """Analyze comparative toxicity responses"""
        print("Analyzing comparative toxicity patterns...")
        
        comparison_counts = self.human_evaluation_data['toxicity_comparison'].value_counts()
        comparison_percentages = comparison_counts / len(self.human_evaluation_data) * 100
        
        self.comparison_analysis = {
            'counts': comparison_counts.to_dict(),
            'percentages': comparison_percentages.to_dict()
        }
        
        return self.comparison_analysis
    
    def compute_summary_statistics(self):
        """Compute summary statistics for human evaluations"""
        print("Computing summary statistics...")
        
        stats = {}
        
        # Overall statistics
        stats['naturalness'] = {
            'mean_rating': self.human_evaluation_data['naturalness_rating'].mean(),
            'std_rating': self.human_evaluation_data['naturalness_rating'].std(),
            'mean_score': self.human_evaluation_data['naturalness_score'].mean(),
            'std_score': self.human_evaluation_data['naturalness_score'].std(),
            'distribution': self.human_evaluation_data['naturalness_rating'].value_counts().to_dict()
        }
        
        stats['toxicity'] = {
            'mean_rating': self.human_evaluation_data['toxicity_rating'].mean(),
            'std_rating': self.human_evaluation_data['toxicity_rating'].std(),
            'mean_score': self.human_evaluation_data['toxicity_score'].mean(),
            'std_score': self.human_evaluation_data['toxicity_score'].std(),
            'distribution': self.human_evaluation_data['toxicity_rating'].value_counts().to_dict()
        }
        
        # Inter-rater reliability (if multiple raters per sentence)
        sentences_with_multiple_raters = self.human_evaluation_data.groupby('sentence_id').size()
        sentences_with_multiple = sentences_with_multiple_raters[sentences_with_multiple_raters > 1]
        
        if len(sentences_with_multiple) > 0:
            # Compute Krippendorff's alpha approximation using ICC
            reliability_data = []
            for sentence_id in sentences_with_multiple.index:
                sentence_ratings = self.human_evaluation_data[
                    self.human_evaluation_data['sentence_id'] == sentence_id
                ]
                
                if len(sentence_ratings) >= 2:
                    reliability_data.append({
                        'naturalness_var': sentence_ratings['naturalness_rating'].var(),
                        'toxicity_var': sentence_ratings['toxicity_rating'].var(),
                        'n_raters': len(sentence_ratings)
                    })
            
            if reliability_data:
                reliability_df = pd.DataFrame(reliability_data)
                stats['inter_rater_reliability'] = {
                    'sentences_with_multiple_raters': len(sentences_with_multiple),
                    'mean_naturalness_variance': reliability_df['naturalness_var'].mean(),
                    'mean_toxicity_variance': reliability_df['toxicity_var'].mean(),
                    'average_raters_per_sentence': reliability_df['n_raters'].mean()
                }
        
        self.summary_stats = stats
        return stats
    
    def create_visualizations(self):
        """Create visualizations for the analysis results"""
        print("Creating visualizations...")
        
        # Set up the output directory for plots
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Distribution of human ratings
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(data=self.human_evaluation_data, x='naturalness_rating', bins=5)
        plt.title('Distribution of Naturalness Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        sns.histplot(data=self.human_evaluation_data, x='toxicity_rating', bins=5)
        plt.title('Distribution of Toxicity Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'rating_distributions.png'))
        plt.close()
        
        # NEW: Distribution of rating percentages (how many people chose each rating)
        self.create_rating_percentage_distributions(plots_dir)
        
        # NEW: Box plots for perspective scores debugging
        self.create_perspective_score_boxplots(plots_dir)
        
        # 2. Comparison of human and automated toxicity distributions
        plt.figure(figsize=(10, 6))
        
        # Convert human toxicity ratings to percentages for better comparison
        human_toxicity_percentages = self.merged_data['toxicity_score_mean']
        perspective_toxicity = self.merged_data['generated_toxicity']
        
        # Create a DataFrame for plotting
        toxicity_comparison = pd.DataFrame({
            'Human Toxicity Score': human_toxicity_percentages,
            'Perspective API Toxicity': perspective_toxicity
        })
        
        # Melt the DataFrame for easier plotting
        toxicity_comparison_melted = pd.melt(toxicity_comparison, 
                                           var_name='Metric',
                                           value_name='Score')
        
        # Create the plot
        sns.histplot(data=toxicity_comparison_melted, 
                    x='Score', 
                    hue='Metric',
                    bins=20,
                    alpha=0.5,
                    multiple='layer')
        
        plt.title('Distribution of Human vs Automated Toxicity Scores')
        plt.xlabel('Toxicity Score')
        plt.ylabel('Count')
        plt.legend(title='Metric')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'toxicity_distribution_comparison.png'))
        plt.close()
        
        # 3. Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_data = self.merged_data[[
            'naturalness_score_mean', 'toxicity_score_mean',
            'generated_toxicity', 'generated_severe_toxicity',
            'generated_identity_attack', 'generated_insult',
            'generated_profanity', 'generated_threat'
        ]]
        sns.heatmap(correlation_data.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap of Human Ratings and Generated Text Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'))
        plt.close()
        
        # 4. Scatter plots for key correlations
        plt.figure(figsize=(15, 10))
        
        # Naturalness vs Generated Toxicity
        plt.subplot(2, 2, 1)
        sns.scatterplot(data=self.merged_data, 
                       x='naturalness_score_mean', 
                       y='generated_toxicity')
        plt.title('Naturalness vs Generated Toxicity')
        plt.xlabel('Naturalness Score')
        plt.ylabel('Generated Toxicity Score')
        
        # Toxicity vs Generated Toxicity
        plt.subplot(2, 2, 2)
        sns.scatterplot(data=self.merged_data, 
                       x='toxicity_score_mean', 
                       y='generated_toxicity')
        plt.title('Human Toxicity vs Generated Toxicity')
        plt.xlabel('Human Toxicity Score')
        plt.ylabel('Generated Toxicity Score')
        
        # Naturalness vs Generated Profanity
        plt.subplot(2, 2, 3)
        sns.scatterplot(data=self.merged_data, 
                       x='naturalness_score_mean', 
                       y='generated_profanity')
        plt.title('Naturalness vs Generated Profanity')
        plt.xlabel('Naturalness Score')
        plt.ylabel('Generated Profanity Score')
        
        # Toxicity vs Generated Profanity
        plt.subplot(2, 2, 4)
        sns.scatterplot(data=self.merged_data, 
                       x='toxicity_score_mean', 
                       y='generated_profanity')
        plt.title('Human Toxicity vs Generated Profanity')
        plt.xlabel('Human Toxicity Score')
        plt.ylabel('Generated Profanity Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'correlation_scatter_plots.png'))
        plt.close()
        
        # 5. Correlation between Human Naturalness and Human Toxicity
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=self.merged_data, 
                       x='naturalness_score_mean', 
                       y='toxicity_score_mean',
                       alpha=0.7)
        
        # Calculate and display correlation
        naturalness_toxicity_corr, naturalness_toxicity_p = pearsonr(
            self.merged_data['naturalness_score_mean'],
            self.merged_data['toxicity_score_mean']
        )
        
        # Add trendline
        x_values = self.merged_data['naturalness_score_mean']
        y_values = self.merged_data['toxicity_score_mean']
        z = np.polyfit(x_values, y_values, 1)
        p = np.poly1d(z)
        plt.plot(x_values, p(x_values), "r--", alpha=0.8)
        
        plt.title(f'Human Naturalness vs Human Toxicity\n'
                 f'Pearson r = {naturalness_toxicity_corr:.3f}, p = {naturalness_toxicity_p:.3f}')
        plt.xlabel('Human Naturalness Score')
        plt.ylabel('Human Toxicity Score')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'naturalness_vs_human_toxicity.png'))
        plt.close()
        
        # 6. Box plots for toxicity comparison
        plt.figure(figsize=(12, 6))
        toxicity_data = pd.melt(self.merged_data, 
                              value_vars=['toxicity_score_mean', 
                                        'generated_toxicity',
                                        'generated_severe_toxicity',
                                        'generated_profanity'],
                              var_name='Metric',
                              value_name='Score')
        sns.boxplot(data=toxicity_data, x='Metric', y='Score')
        plt.title('Distribution of Toxicity Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'toxicity_distributions.png'))
        plt.close()
        
        print(f"Visualizations saved to {plots_dir}")
        return True
    
    def create_rating_percentage_distributions(self, plots_dir):
        """Create distribution plots showing percentage of people who chose each rating (1-5)"""
        print("Creating rating percentage distribution plots...")
        
        # Calculate percentage distributions for naturalness and toxicity ratings
        naturalness_counts = self.human_evaluation_data['naturalness_rating'].value_counts().sort_index()
        toxicity_counts = self.human_evaluation_data['toxicity_rating'].value_counts().sort_index()
        
        total_responses_naturalness = naturalness_counts.sum()
        total_responses_toxicity = toxicity_counts.sum()
        
        naturalness_percentages = (naturalness_counts / total_responses_naturalness * 100).round(1)
        toxicity_percentages = (toxicity_counts / total_responses_toxicity * 100).round(1)
        
        # Create the plot
        plt.figure(figsize=(14, 6))
        
        # Naturalness rating percentages
        plt.subplot(1, 2, 1)
        bars1 = plt.bar(naturalness_percentages.index, naturalness_percentages.values, 
                       color=professional_colors[0], alpha=0.7)
        plt.title('Distribution of Naturalness Ratings\n(Percentage of Responses)')
        plt.xlabel('Rating')
        plt.ylabel('Percentage of Responses (%)')
        plt.xticks(range(1, 6))
        
        # Add percentage labels on bars
        for bar, percentage in zip(bars1, naturalness_percentages.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{percentage}%', ha='center', va='bottom', fontweight='bold')
        
        plt.ylim(0, max(naturalness_percentages.values) * 1.1)
        
        # Toxicity rating percentages
        plt.subplot(1, 2, 2)
        bars2 = plt.bar(toxicity_percentages.index, toxicity_percentages.values, 
                       color=professional_colors[1], alpha=0.7)
        plt.title('Distribution of Toxicity Ratings\n(Percentage of Responses)')
        plt.xlabel('Rating')
        plt.ylabel('Percentage of Responses (%)')
        plt.xticks(range(1, 6))
        
        # Add percentage labels on bars
        for bar, percentage in zip(bars2, toxicity_percentages.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{percentage}%', ha='center', va='bottom', fontweight='bold')
        
        plt.ylim(0, max(toxicity_percentages.values) * 1.1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'rating_percentage_distributions.png'), dpi=300)
        plt.close()
        
        # Print summary statistics
        print(f"Naturalness rating distribution: {dict(naturalness_percentages)}")
        print(f"Toxicity rating distribution: {dict(toxicity_percentages)}")
        
    def create_perspective_score_boxplots(self, plots_dir):
        """Create box plots for perspective scores to help debug distribution issues"""
        print("Creating perspective score box plots for debugging...")
        
        # Create box plots for all perspective scores
        plt.figure(figsize=(15, 8))
        
        # Prepare data for box plots
        perspective_metrics = ['generated_toxicity', 'generated_severe_toxicity', 
                             'generated_identity_attack', 'generated_insult',
                             'generated_profanity', 'generated_threat']
        
        perspective_data = self.merged_data[perspective_metrics]
        
        # Create box plot
        plt.subplot(1, 2, 1)
        sns.boxplot(data=perspective_data)
        plt.title('Distribution of Perspective API Scores\n(Generated Text)')
        plt.xlabel('Perspective Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45, ha='right')
        
        # Add summary statistics as text
        plt.subplot(1, 2, 2)
        plt.axis('off')
        
        # Create summary statistics table
        summary_text = "Perspective API Score Statistics (Generated Text):\n\n"
        for metric in perspective_metrics:
            values = self.merged_data[metric]
            summary_text += f"{metric}:\n"
            summary_text += f"  Mean: {values.mean():.4f}\n"
            summary_text += f"  Std:  {values.std():.4f}\n"
            summary_text += f"  Min:  {values.min():.4f}\n"
            summary_text += f"  Max:  {values.max():.4f}\n"
            summary_text += f"  Q1:   {values.quantile(0.25):.4f}\n"
            summary_text += f"  Q3:   {values.quantile(0.75):.4f}\n\n"
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'perspective_scores_boxplot_debug.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create individual histograms for each metric
        plt.figure(figsize=(15, 10))
        
        for i, metric in enumerate(perspective_metrics, 1):
            plt.subplot(2, 3, i)
            values = self.merged_data[metric]
            
            plt.hist(values, bins=20, alpha=0.7, color=professional_colors[i % len(professional_colors)])
            plt.title(f'{metric}\n(Mean: {values.mean():.4f}, Std: {values.std():.4f})')
            plt.xlabel('Score')
            plt.ylabel('Count')
            
            # Add vertical lines for quartiles
            plt.axvline(values.quantile(0.25), color='red', linestyle='--', alpha=0.7, label='Q1')
            plt.axvline(values.median(), color='orange', linestyle='--', alpha=0.7, label='Median')
            plt.axvline(values.quantile(0.75), color='green', linestyle='--', alpha=0.7, label='Q3')
            
            if i == 1:  # Add legend only to first subplot
                plt.legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'perspective_scores_histograms_debug.png'), dpi=300)
        plt.close()
        
        print("Perspective score debugging plots created.")
    
    def save_results(self):
        """Save analysis results to files"""
        print("Saving analysis results...")
        
        # 1. Save correlation results
        correlation_file = os.path.join(self.output_dir, "correlation_results.csv")
        correlation_data = []
        
        for key, values in self.correlation_results.items():
            human_metric, auto_metric = key.split('_vs_')
            correlation_data.append({
                'human_metric': human_metric,
                'automated_metric': auto_metric,
                'pearson_correlation': values['pearson_correlation'],
                'pearson_p_value': values['pearson_p_value'],
                'spearman_correlation': values['spearman_correlation'],
                'spearman_p_value': values['spearman_p_value']
            })
        
        pd.DataFrame(correlation_data).to_csv(correlation_file, index=False)
        print(f"Correlation results saved to {correlation_file}")
        
        # 2. Save summary statistics
        summary_file = os.path.join(self.output_dir, "summary_statistics.csv")
        
        # Calculate summary statistics for human ratings
        human_summary = self.human_evaluation_data.agg({
            'naturalness_rating': ['mean', 'std', 'min', 'max'],
            'toxicity_rating': ['mean', 'std', 'min', 'max']
        }).round(3)
        
        # Calculate summary statistics for generated text metrics
        generated_summary = self.merged_data.agg({
            'generated_toxicity': ['mean', 'std', 'min', 'max'],
            'generated_severe_toxicity': ['mean', 'std', 'min', 'max'],
            'generated_identity_attack': ['mean', 'std', 'min', 'max'],
            'generated_insult': ['mean', 'std', 'min', 'max'],
            'generated_profanity': ['mean', 'std', 'min', 'max'],
            'generated_threat': ['mean', 'std', 'min', 'max']
        }).round(3)
        
        # Combine summaries
        summary_stats = pd.concat([human_summary, generated_summary], axis=1)
        summary_stats.to_csv(summary_file)
        print(f"Summary statistics saved to {summary_file}")
        
        # 3. Save detailed results
        detailed_file = os.path.join(self.output_dir, "detailed_results.csv")
        self.merged_data.to_csv(detailed_file, index=False)
        print(f"Detailed results saved to {detailed_file}")
        
        return True
    
    def run_analysis(self):
        """Run the complete human evaluation analysis"""
        print("Starting Human Evaluation Analysis...")
        print("=" * 60)
        
        # Load data
        if not self.load_data():
            print("Failed to load data. Exiting.")
            return False
        
        # Parse form responses
        if not self.parse_form_responses():
            print("Failed to parse form responses. Exiting.")
            return False
        
        # Convert ratings to scores
        if not self.convert_ratings_to_scores():
            print("Failed to convert ratings. Exiting.")
            return False
        
        # Merge with perspective data
        if not self.merge_with_perspective_data():
            print("Failed to merge data. Exiting.")
            return False
        
        # Compute correlations
        self.compute_correlations()
        
        # Analyze comparative toxicity
        self.analyze_comparative_toxicity()
        
        # Compute summary statistics
        self.compute_summary_statistics()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save results
        self.save_results()
        
        print("\n" + "=" * 60)
        print("Human Evaluation Analysis completed successfully!")
        print(f"Results saved to: {self.output_dir}")
        
        return True

def main():
    """Main function to run the analysis"""
    parser = argparse.ArgumentParser(description='Human Evaluation Analysis for Code-Switched Text')
    parser.add_argument('--form_responses_file', 
                       default='final_data/Google Form Responses.csv',
                       help='Path to Google Form responses CSV file')
    parser.add_argument('--perspective_file', 
                       default='temp_scripts/perspective_analysis_form.csv',
                       help='Path to perspective analysis CSV file')
    parser.add_argument('--output_dir', 
                       default='final_outputs',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create analyzer instance
    analyzer = HumanEvaluationAnalyzer(
        form_responses_file=args.form_responses_file,
        perspective_file=args.perspective_file,
        output_dir=args.output_dir
    )
    
    # Run analysis
    success = analyzer.run_analysis()
    
    if success:
        print("Analysis completed successfully!")
        return 0
    else:
        print("Analysis failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 