#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, ttest_rel, f_oneway
from matplotlib.ticker import ScalarFormatter
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import math

def load_data(src_csv, tgt_csv, cs_csv):
    """Load and prepare data from source, target, and code-switched CSV files."""
    # Load data
    src_df = pd.read_csv(src_csv)
    tgt_df = pd.read_csv(tgt_csv)
    cs_df = pd.read_csv(cs_csv)
    
    # Make sure we have matching primary keys
    src_df = src_df.dropna(subset=['primary_key'])
    tgt_df = tgt_df.dropna(subset=['primary_key'])
    cs_df = cs_df.dropna(subset=['primary_key'])
    
    # Merge datasets on primary_key
    merged_df = pd.merge(src_df, tgt_df, on='primary_key', suffixes=('_src', '_tgt'))
    merged_df = pd.merge(merged_df, cs_df, on='primary_key', suffixes=('', '_cs'))
    
    return merged_df

def get_toxicity_metrics(df):
    """Extract toxicity metrics columns from dataframe."""
    # Updated to match the actual column naming patterns
    metrics = []
    prefixes = ['perspective_src_', 'perspective_tgt_', 'perspective_generated_']
    base_metrics = ['toxicity', 'identity_attack', 'insult', 'threat', 'severe_toxicity', 
                    'profanity', 'sexually_explicit']
    
    for prefix in prefixes:
        for metric in base_metrics:
            col = prefix + metric
            if col in df.columns:
                metrics.append(col)
    
    return metrics

def create_correlation_plots(df, output_dir, model_name):
    """Create correlation plots for each toxicity metric."""
    os.makedirs(output_dir, exist_ok=True)
    base_metrics = ['toxicity', 'identity_attack', 'insult', 'threat', 'severe_toxicity', 
                   'profanity', 'sexually_explicit']
    
    for metric in base_metrics:
        # Find columns for this metric with updated naming pattern
        src_col = f"perspective_src_{metric}"
        tgt_col = f"perspective_tgt_{metric}"
        cs_col = f"perspective_generated_{metric}"
        
        if not all(col in df.columns for col in [src_col, tgt_col, cs_col]):
            print(f"Missing columns for {metric}, skipping...")
            continue
        
        # Create figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"{model_name.upper()} - {metric.replace('_', ' ').title()} Correlation Analysis", fontsize=16)
        
        # 1. SRC vs CS correlation
        plot_correlation(df, src_col, cs_col, "Source", "Code-Switched", 
                         axes[0], f"{model_name}_{metric}_src_vs_cs.png", output_dir)
        
        # 2. TGT vs CS correlation
        plot_correlation(df, tgt_col, cs_col, "Target", "Code-Switched", 
                         axes[1], f"{model_name}_{metric}_tgt_vs_cs.png", output_dir)
        
        # 3. SRC vs TGT correlation
        plot_correlation(df, src_col, tgt_col, "Source", "Target", 
                         axes[2], f"{model_name}_{metric}_src_vs_tgt.png", output_dir)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f"{model_name}_{metric}_correlation_plots.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created correlation plots for {metric}")

def plot_correlation(df, x_col, y_col, x_label, y_label, ax, filename, output_dir):
    """Create a single correlation plot with trend line and statistics."""
    # Calculate correlations
    valid_data = df[[x_col, y_col]].dropna()
    if len(valid_data) < 2:
        ax.text(0.5, 0.5, "Insufficient data for correlation", ha='center', va='center')
        return
        
    pearson_r, pearson_p = pearsonr(valid_data[x_col], valid_data[y_col])
    spearman_r, spearman_p = spearmanr(valid_data[x_col], valid_data[y_col])
    
    # Create scatter plot
    sns.scatterplot(x=x_col, y=y_col, data=valid_data, ax=ax, alpha=0.7)
    
    # Add trend line
    sns.regplot(x=x_col, y=y_col, data=valid_data, scatter=False, 
                line_kws={'color': 'red'}, ax=ax)
    
    # Add correlation info
    text = (f"Pearson r = {pearson_r:.2f} (p = {pearson_p:.3f})\n"
            f"Spearman ρ = {spearman_r:.2f} (p = {spearman_p:.3f})\n"
            f"n = {len(valid_data)}")
    
    ax.text(0.05, 0.95, text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='left',
            bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
    
    # Extract metric name from column name for better labels
    x_metric = x_col.split('_')[-1]
    y_metric = y_col.split('_')[-1]
    
    # Set labels and title
    ax.set_xlabel(f"{x_label} {x_metric.title()}")
    ax.set_ylabel(f"{y_label} {y_metric.title()}")
    ax.set_title(f"{x_label} vs {y_label}")
    
    # Use log scale if values span multiple orders of magnitude
    if valid_data[x_col].max() > 10 * valid_data[x_col].min():
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ScalarFormatter())
    
    if valid_data[y_col].max() > 10 * valid_data[y_col].min():
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())

def create_combined_model_plots(models_data, metrics, output_dir):
    """Create plots comparing correlation strengths across models."""
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_types = [
        ('src', 'generated', 'Source vs Code-Switched'),
        ('tgt', 'generated', 'Target vs Code-Switched'),
        ('src', 'tgt', 'Source vs Target')
    ]
    
    for metric in metrics:
        # Create a figure for this metric with all models
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Model Comparison - {metric.replace('_', ' ').title()} Correlation", fontsize=16)
        
        for i, (type1, type2, title) in enumerate(comparison_types):
            correlation_data = []
            
            for model_name, model_data in models_data.items():
                # Find columns with updated naming pattern
                col1 = f"perspective_{type1}_{metric}"
                col2 = f"perspective_{type2}_{metric}"
                
                if col1 not in model_data.columns or col2 not in model_data.columns:
                    print(f"Missing columns for {model_name}, {metric}, {type1} vs {type2}")
                    continue
                
                # Calculate correlation
                valid_data = model_data[[col1, col2]].dropna()
                if len(valid_data) < 2:
                    continue
                    
                pearson_r, pearson_p = pearsonr(valid_data[col1], valid_data[col2])
                spearman_r, spearman_p = spearmanr(valid_data[col1], valid_data[col2])
                
                correlation_data.append({
                    'model': model_name,
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p
                })
            
            if correlation_data:
                corr_df = pd.DataFrame(correlation_data)
                
                # Plot correlation coefficients
                barplot = sns.barplot(x='model', y='pearson_r', data=corr_df, ax=axes[i])
                
                # Add significance markers
                for j, p in enumerate(corr_df['pearson_p']):
                    significance = ''
                    if p < 0.001:
                        significance = '***'
                    elif p < 0.01:
                        significance = '**'
                    elif p < 0.05:
                        significance = '*'
                    
                    if significance:
                        axes[i].text(j, corr_df.iloc[j]['pearson_r'] + 0.02, 
                                    significance, ha='center', va='bottom')
                
                # Set labels
                axes[i].set_xlabel('Model')
                axes[i].set_ylabel('Pearson Correlation (r)')
                axes[i].set_title(title)
                axes[i].set_ylim(-1, 1)
                
                # Add horizontal line at y=0
                axes[i].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # Add text explaining significance
                axes[i].text(0.5, -0.15, "* p<0.05, ** p<0.01, *** p<0.001", 
                            transform=axes[i].transAxes, ha='center')
            else:
                axes[i].text(0.5, 0.5, "Insufficient data", ha='center', va='center')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f"model_comparison_{metric}_correlation.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created model comparison plots for {metric}")

def create_distribution_plots(df, output_dir, model_name):
    """Create distribution plots comparing toxicity distributions across source, target, and code-switched text."""
    os.makedirs(output_dir, exist_ok=True)
    base_metrics = ['toxicity', 'identity_attack', 'insult', 'threat', 'severe_toxicity', 
                   'profanity', 'sexually_explicit']
    
    for metric in base_metrics:
        # Find columns for this metric
        src_col = f"perspective_src_{metric}"
        tgt_col = f"perspective_tgt_{metric}"
        cs_col = f"perspective_generated_{metric}"
        
        if not all(col in df.columns for col in [src_col, tgt_col, cs_col]):
            print(f"Missing columns for {metric}, skipping distribution plot...")
            continue
        
        # Create data for plotting
        plot_data = df[[src_col, tgt_col, cs_col]].copy()
        plot_data.columns = ['Source', 'Target', 'Code-Switched']
        plot_data_melted = pd.melt(plot_data, var_name='Type', value_name=f'{metric.title()} Score')
        
        # Create histogram/density plot
        plt.figure(figsize=(12, 7))
        
        # KDE plot
        ax = sns.kdeplot(data=plot_data_melted, x=f'{metric.title()} Score', hue='Type', fill=True, common_norm=False, alpha=0.5)
        
        # Add median lines
        for i, text_type in enumerate(['Source', 'Target', 'Code-Switched']):
            median_val = plot_data[text_type].median()
            plt.axvline(x=median_val, color=sns.color_palette()[i], linestyle='--', 
                        label=f'{text_type} Median: {median_val:.3f}')
        
        plt.title(f"{model_name.upper()} - {metric.replace('_', ' ').title()} Score Distribution")
        plt.xlabel(f"{metric.replace('_', ' ').title()} Score")
        plt.ylabel("Density")
        plt.legend(title='Text Type')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_{metric}_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create boxplot
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=plot_data_melted, x='Type', y=f'{metric.title()} Score')
        plt.title(f"{model_name.upper()} - {metric.replace('_', ' ').title()} Score Boxplot")
        plt.xlabel("Text Type")
        plt.ylabel(f"{metric.replace('_', ' ').title()} Score")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_{metric}_boxplot.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create violin plot
        plt.figure(figsize=(12, 7))
        sns.violinplot(data=plot_data_melted, x='Type', y=f'{metric.title()} Score', inner='quart')
        plt.title(f"{model_name.upper()} - {metric.replace('_', ' ').title()} Score Violin Plot")
        plt.xlabel("Text Type")
        plt.ylabel(f"{metric.replace('_', ' ').title()} Score")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_{metric}_violinplot.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created distribution plots for {metric}")

def create_language_composition_analysis(df, output_dir, model_name):
    """Analyze the relationship between language composition and toxicity."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have language composition metrics
    language_metrics = ['hindi_percent', 'english_percent', 'total_hindi_percent']
    if not any(metric in df.columns for metric in language_metrics):
        print(f"Language composition metrics not found for {model_name}, skipping...")
        return
    
    # Filter to only the metrics we have
    available_lang_metrics = [metric for metric in language_metrics if metric in df.columns]
    
    # Get toxicity metrics
    base_metrics = ['toxicity', 'identity_attack', 'insult', 'threat', 'severe_toxicity', 
                   'profanity', 'sexually_explicit']
    
    for metric in base_metrics:
        cs_col = f"perspective_generated_{metric}"
        
        if cs_col not in df.columns:
            continue
            
        # Create scatter plot for each language metric vs code-switched toxicity
        fig, axes = plt.subplots(1, len(available_lang_metrics), figsize=(6*len(available_lang_metrics), 6))
        if len(available_lang_metrics) == 1:
            axes = [axes]
        
        fig.suptitle(f"{model_name.upper()} - {metric.title()} by Language Composition", fontsize=16)
        
        for i, lang_metric in enumerate(available_lang_metrics):
            # Calculate correlation
            valid_data = df[[lang_metric, cs_col]].dropna()
            if len(valid_data) < 2:
                axes[i].text(0.5, 0.5, "Insufficient data", ha='center', va='center')
                continue
                
            pearson_r, pearson_p = pearsonr(valid_data[lang_metric], valid_data[cs_col])
            
            # Create scatter plot with trend line
            sns.regplot(x=lang_metric, y=cs_col, data=valid_data, ax=axes[i], 
                       scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
            
            # Add correlation info
            text = f"Pearson r = {pearson_r:.2f}\np = {pearson_p:.3f}"
            axes[i].text(0.05, 0.95, text, transform=axes[i].transAxes, 
                      verticalalignment='top', horizontalalignment='left',
                      bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
            
            # Set labels
            axes[i].set_xlabel(f"{lang_metric.replace('_', ' ').title()} (%)")
            axes[i].set_ylabel(f"Code-Switched {metric.title()}")
            axes[i].set_title(f"{lang_metric.replace('_', ' ').title()} vs {metric.title()}")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f"{model_name}_{metric}_language_composition.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created language composition analysis for {metric}")

def calculate_statistical_tests(df, output_dir, model_name):
    """Perform statistical tests to compare toxicity across source, target, and code-switched text."""
    os.makedirs(output_dir, exist_ok=True)
    
    base_metrics = ['toxicity', 'identity_attack', 'insult', 'threat', 'severe_toxicity', 
                   'profanity', 'sexually_explicit']
    
    # Create dataframe to store test results
    test_results = []
    
    for metric in base_metrics:
        # Find columns for this metric
        src_col = f"perspective_src_{metric}"
        tgt_col = f"perspective_tgt_{metric}"
        cs_col = f"perspective_generated_{metric}"
        
        if not all(col in df.columns for col in [src_col, tgt_col, cs_col]):
            print(f"Missing columns for {metric}, skipping statistical tests...")
            continue
        
        # Create data for tests
        valid_data = df[[src_col, tgt_col, cs_col]].dropna()
        
        if len(valid_data) < 2:
            print(f"Insufficient data for {metric}, skipping statistical tests...")
            continue
        
        # 1. Paired t-tests
        # Source vs Code-switched
        src_cs_ttest = ttest_rel(valid_data[src_col], valid_data[cs_col])
        
        # Target vs Code-switched
        tgt_cs_ttest = ttest_rel(valid_data[tgt_col], valid_data[cs_col])
        
        # Source vs Target
        src_tgt_ttest = ttest_rel(valid_data[src_col], valid_data[tgt_col])
        
        # 2. One-way ANOVA
        anova_data = pd.melt(valid_data, value_vars=[src_col, tgt_col, cs_col], var_name='Type', value_name='Value')
        fvalue, pvalue = f_oneway(valid_data[src_col], valid_data[tgt_col], valid_data[cs_col])
        
        # 3. Effect sizes (Cohen's d)
        def cohens_d(x, y):
            nx, ny = len(x), len(y)
            dof = nx + ny - 2
            return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
        
        src_cs_d = cohens_d(valid_data[src_col], valid_data[cs_col])
        tgt_cs_d = cohens_d(valid_data[tgt_col], valid_data[cs_col])
        src_tgt_d = cohens_d(valid_data[src_col], valid_data[tgt_col])
        
        # 4. Post-hoc Tukey test
        tukey_data = anova_data.copy()
        tukey_data['Type'] = tukey_data['Type'].str.replace(f"perspective_", "")
        tukey_data['Type'] = tukey_data['Type'].str.replace(f"_{metric}", "")
        tukey = pairwise_tukeyhsd(endog=tukey_data['Value'], groups=tukey_data['Type'], alpha=0.05)
        
        # Add results to dataframe
        test_results.append({
            'metric': metric,
            'src_mean': valid_data[src_col].mean(),
            'tgt_mean': valid_data[tgt_col].mean(),
            'cs_mean': valid_data[cs_col].mean(),
            'src_cs_ttest_t': src_cs_ttest.statistic,
            'src_cs_ttest_p': src_cs_ttest.pvalue,
            'tgt_cs_ttest_t': tgt_cs_ttest.statistic,
            'tgt_cs_ttest_p': tgt_cs_ttest.pvalue,
            'src_tgt_ttest_t': src_tgt_ttest.statistic,
            'src_tgt_ttest_p': src_tgt_ttest.pvalue,
            'anova_f': fvalue,
            'anova_p': pvalue,
            'src_cs_cohens_d': src_cs_d,
            'tgt_cs_cohens_d': tgt_cs_d,
            'src_tgt_cohens_d': src_tgt_d,
            'n': len(valid_data)
        })
        
        # Create bar plot of means with error bars
        plt.figure(figsize=(10, 6))
        
        # Calculate means and standard errors
        means = [valid_data[src_col].mean(), valid_data[tgt_col].mean(), valid_data[cs_col].mean()]
        std_errors = [valid_data[src_col].std() / math.sqrt(len(valid_data)), 
                     valid_data[tgt_col].std() / math.sqrt(len(valid_data)),
                     valid_data[cs_col].std() / math.sqrt(len(valid_data))]
        
        # Plot
        bars = plt.bar(['Source', 'Target', 'Code-Switched'], means, yerr=std_errors, capsize=10)
        
        # Add significance markers
        def add_significance(p, x1, x2, y, h):
            if p < 0.001:
                sig = '***'
            elif p < 0.01:
                sig = '**'
            elif p < 0.05:
                sig = '*'
            else:
                sig = 'ns'
                
            plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')
            plt.text((x1+x2)*.5, y+h, sig, ha='center', va='bottom')
        
        # Add markers for each comparison
        max_height = max(means) + max(std_errors) * 2
        add_significance(src_cs_ttest.pvalue, 0, 2, max_height, 0.05 * max_height)
        add_significance(tgt_cs_ttest.pvalue, 1, 2, max_height + 0.1 * max_height, 0.05 * max_height)
        add_significance(src_tgt_ttest.pvalue, 0, 1, max_height + 0.2 * max_height, 0.05 * max_height)
        
        # Add labels and title
        plt.xlabel('Text Type')
        plt.ylabel(f'{metric.replace("_", " ").title()} Score')
        plt.title(f'{model_name.upper()} - Mean {metric.replace("_", " ").title()} Scores')
        
        # Add annotation for significance
        plt.figtext(0.5, 0.01, "* p<0.05, ** p<0.01, *** p<0.001, ns: not significant", 
                   ha='center', fontsize=10)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f"{model_name}_{metric}_mean_comparison.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save test results to CSV
    if test_results:
        test_results_df = pd.DataFrame(test_results)
        test_results_df.to_csv(os.path.join(output_dir, f"{model_name}_statistical_tests.csv"), index=False)
        print(f"Saved statistical test results to {model_name}_statistical_tests.csv")

def create_toxicity_difference_plots(df, output_dir, model_name):
    """Create plots showing the difference in toxicity between source, target, and code-switched text."""
    os.makedirs(output_dir, exist_ok=True)
    base_metrics = ['toxicity', 'identity_attack', 'insult', 'threat', 'severe_toxicity', 
                   'profanity', 'sexually_explicit']
    
    for metric in base_metrics:
        # Find columns for this metric
        src_col = f"perspective_src_{metric}"
        tgt_col = f"perspective_tgt_{metric}"
        cs_col = f"perspective_generated_{metric}"
        
        if not all(col in df.columns for col in [src_col, tgt_col, cs_col]):
            print(f"Missing columns for {metric}, skipping difference plots...")
            continue
        
        # Create data for plotting
        valid_data = df[[src_col, tgt_col, cs_col]].dropna().copy()
        
        if len(valid_data) < 2:
            print(f"Insufficient data for {metric}, skipping difference plots...")
            continue
        
        # Calculate differences
        valid_data['src_cs_diff'] = valid_data[cs_col] - valid_data[src_col]
        valid_data['tgt_cs_diff'] = valid_data[cs_col] - valid_data[tgt_col]
        valid_data['src_tgt_diff'] = valid_data[tgt_col] - valid_data[src_col]
        
        # Create histogram of differences
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"{model_name.upper()} - {metric.replace('_', ' ').title()} Score Differences", fontsize=16)
        
        # 1. Source vs CS difference
        sns.histplot(valid_data['src_cs_diff'], kde=True, ax=axes[0], color='blue')
        axes[0].axvline(x=0, color='red', linestyle='--')
        axes[0].set_title('Code-Switched minus Source')
        axes[0].set_xlabel('Difference')
        
        # Add mean and median
        mean_diff = valid_data['src_cs_diff'].mean()
        median_diff = valid_data['src_cs_diff'].median()
        axes[0].axvline(x=mean_diff, color='green', linestyle='-', label=f'Mean: {mean_diff:.3f}')
        axes[0].axvline(x=median_diff, color='purple', linestyle=':', label=f'Median: {median_diff:.3f}')
        axes[0].legend()
        
        # 2. Target vs CS difference
        sns.histplot(valid_data['tgt_cs_diff'], kde=True, ax=axes[1], color='orange')
        axes[1].axvline(x=0, color='red', linestyle='--')
        axes[1].set_title('Code-Switched minus Target')
        axes[1].set_xlabel('Difference')
        
        # Add mean and median
        mean_diff = valid_data['tgt_cs_diff'].mean()
        median_diff = valid_data['tgt_cs_diff'].median()
        axes[1].axvline(x=mean_diff, color='green', linestyle='-', label=f'Mean: {mean_diff:.3f}')
        axes[1].axvline(x=median_diff, color='purple', linestyle=':', label=f'Median: {median_diff:.3f}')
        axes[1].legend()
        
        # 3. Source vs Target difference
        sns.histplot(valid_data['src_tgt_diff'], kde=True, ax=axes[2], color='green')
        axes[2].axvline(x=0, color='red', linestyle='--')
        axes[2].set_title('Target minus Source')
        axes[2].set_xlabel('Difference')
        
        # Add mean and median
        mean_diff = valid_data['src_tgt_diff'].mean()
        median_diff = valid_data['src_tgt_diff'].median()
        axes[2].axvline(x=mean_diff, color='blue', linestyle='-', label=f'Mean: {mean_diff:.3f}')
        axes[2].axvline(x=median_diff, color='purple', linestyle=':', label=f'Median: {median_diff:.3f}')
        axes[2].legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f"{model_name}_{metric}_difference_plots.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created difference plots for {metric}")

def main():
    parser = argparse.ArgumentParser(description="Generate correlation plots and additional analyses for toxicity scores.")
    parser.add_argument("--src_dir", type=str, required=True, help="Directory containing source continuation files")
    parser.add_argument("--tgt_dir", type=str, required=True, help="Directory containing target continuation files")
    parser.add_argument("--cs_dir", type=str, required=True, help="Directory containing code-switched continuation files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")
    parser.add_argument("--models", type=str, nargs='+', required=True, help="Model names to process")
    parser.add_argument("--analyses", type=str, nargs='+', default=['all'], 
                        choices=['correlation', 'distribution', 'language', 'stats', 'difference', 'all'],
                        help="Types of analyses to perform")
    parser.add_argument("--src_pattern", type=str, 
                       default="{model}_src_continuations.csv",
                       help="Filename pattern for source files")
    parser.add_argument("--tgt_pattern", type=str,
                       default="{model}_tgt_continuations.csv", 
                       help="Filename pattern for target files")
    parser.add_argument("--cs_pattern", type=str,
                       default="{model}_continuations_perspective_local.csv",
                       help="Filename pattern for code-switched files")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which analyses to run
    run_all = 'all' in args.analyses
    run_correlation = run_all or 'correlation' in args.analyses
    run_distribution = run_all or 'distribution' in args.analyses
    run_language = run_all or 'language' in args.analyses
    run_stats = run_all or 'stats' in args.analyses
    run_difference = run_all or 'difference' in args.analyses
    
    # Process each model
    models_data = {}
    all_metrics = set()
    
    for model in args.models:
        print(f"\nProcessing model: {model}")
        
        # Construct file paths
        src_csv = os.path.join(args.src_dir, args.src_pattern.format(model=model))
        tgt_csv = os.path.join(args.tgt_dir, args.tgt_pattern.format(model=model))
        cs_csv = os.path.join(args.cs_dir, args.cs_pattern.format(model=model))
        
        # Check if files exist
        if not all(os.path.exists(f) for f in [src_csv, tgt_csv, cs_csv]):
            print(f"Error: One or more input files for model {model} not found")
            print(f"Checking for: {src_csv}, {tgt_csv}, {cs_csv}")
            continue
        
        # Load and merge data
        try:
            merged_df = load_data(src_csv, tgt_csv, cs_csv)
            models_data[model] = merged_df
            
            # Identify available metrics for this model
            metrics_in_model = []
            for metric in ['toxicity', 'identity_attack', 'insult', 'threat', 'severe_toxicity', 
                          'profanity', 'sexually_explicit']:
                src_col = f"perspective_src_{metric}"
                tgt_col = f"perspective_tgt_{metric}"
                cs_col = f"perspective_generated_{metric}"
                
                if all(col in merged_df.columns for col in [src_col, tgt_col, cs_col]):
                    metrics_in_model.append(metric)
                    all_metrics.add(metric)
            
            # Create model-specific output directory
            model_output_dir = os.path.join(args.output_dir, model)
            
            # Run selected analyses
            if run_correlation:
                print(f"Running correlation analysis for {model}...")
                create_correlation_plots(merged_df, model_output_dir, model)
            
            if run_distribution:
                print(f"Running distribution analysis for {model}...")
                create_distribution_plots(merged_df, model_output_dir, model)
            
            if run_language:
                print(f"Running language composition analysis for {model}...")
                create_language_composition_analysis(merged_df, model_output_dir, model)
            
            if run_stats:
                print(f"Running statistical tests for {model}...")
                calculate_statistical_tests(merged_df, model_output_dir, model)
            
            if run_difference:
                print(f"Creating toxicity difference plots for {model}...")
                create_toxicity_difference_plots(merged_df, model_output_dir, model)
            
        except Exception as e:
            print(f"Error processing model {model}: {e}")
            continue
    
    # Create combined model comparison plots if we have data from multiple models
    if len(models_data) > 1 and all_metrics and run_correlation:
        comparison_output_dir = os.path.join(args.output_dir, "model_comparison")
        create_combined_model_plots(models_data, list(all_metrics), comparison_output_dir)
    
    print(f"\nCompleted generating plots and analyses. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 