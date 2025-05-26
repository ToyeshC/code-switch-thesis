#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import math

def load_data(input_csv):
    """Load data from the input prompts CSV file."""
    df = pd.read_csv(input_csv)
    return df

def calculate_statistical_tests(df, output_dir):
    """Perform statistical tests to compare toxicity across source, target, and code-switched input prompts."""
    os.makedirs(output_dir, exist_ok=True)
    
    base_metrics = ['toxicity', 'identity_attack', 'insult']
    test_results = []
    
    for metric in base_metrics:
        src_col = f"perspective_src_{metric}"
        tgt_col = f"perspective_tgt_{metric}"
        cs_col = f"perspective_generated_{metric}"
        
        if not all(col in df.columns for col in [src_col, tgt_col, cs_col]):
            print(f"Missing columns for {metric}, skipping statistical tests...")
            continue
        
        valid_data = df[[src_col, tgt_col, cs_col]].dropna()
        
        if len(valid_data) < 2:
            print(f"Insufficient data for {metric}, skipping statistical tests...")
            continue
        
        # Statistical tests
        src_cs_ttest = ttest_rel(valid_data[src_col], valid_data[cs_col])
        tgt_cs_ttest = ttest_rel(valid_data[tgt_col], valid_data[cs_col])
        src_tgt_ttest = ttest_rel(valid_data[src_col], valid_data[tgt_col])
        
        # One-way ANOVA
        fvalue, pvalue = f_oneway(valid_data[src_col], valid_data[tgt_col], valid_data[cs_col])
        
        # Effect sizes (Cohen's d)
        def cohens_d(x, y):
            nx, ny = len(x), len(y)
            dof = nx + ny - 2
            return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
        
        src_cs_d = cohens_d(valid_data[src_col], valid_data[cs_col])
        tgt_cs_d = cohens_d(valid_data[tgt_col], valid_data[cs_col])
        src_tgt_d = cohens_d(valid_data[src_col], valid_data[tgt_col])
        
        # Store results
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
        
        # Create mean comparison plot
        plt.figure(figsize=(10, 6))
        
        means = [valid_data[src_col].mean(), valid_data[tgt_col].mean(), valid_data[cs_col].mean()]
        std_errors = [valid_data[src_col].std() / math.sqrt(len(valid_data)), 
                     valid_data[tgt_col].std() / math.sqrt(len(valid_data)),
                     valid_data[cs_col].std() / math.sqrt(len(valid_data))]
        
        bars = plt.bar(['Source', 'Target', 'Code-Switched'], means, yerr=std_errors, capsize=10)
        
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
        
        max_height = max(means) + max(std_errors) * 2
        add_significance(src_cs_ttest.pvalue, 0, 2, max_height, 0.05 * max_height)
        add_significance(tgt_cs_ttest.pvalue, 1, 2, max_height + 0.1 * max_height, 0.05 * max_height)
        add_significance(src_tgt_ttest.pvalue, 0, 1, max_height + 0.2 * max_height, 0.05 * max_height)
        
        plt.xlabel('Input Type')
        plt.ylabel(f'{metric.replace("_", " ").title()} Score')
        plt.title(f'Mean {metric.replace("_", " ").title()} Scores for Input Prompts')
        plt.figtext(0.5, 0.01, "* p<0.05, ** p<0.01, *** p<0.001, ns: not significant", 
                   ha='center', fontsize=10)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f"input_prompts_{metric}_mean_comparison.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    if test_results:
        test_results_df = pd.DataFrame(test_results)
        test_results_df.to_csv(os.path.join(output_dir, "input_prompts_statistical_tests.csv"), index=False)
        print(f"Saved statistical test results to input_prompts_statistical_tests.csv")

def create_distribution_plots(df, output_dir):
    """Create distribution plots comparing toxicity distributions across input prompts."""
    os.makedirs(output_dir, exist_ok=True)
    base_metrics = ['toxicity', 'identity_attack', 'insult']
    
    for metric in base_metrics:
        src_col = f"perspective_src_{metric}"
        tgt_col = f"perspective_tgt_{metric}"
        cs_col = f"perspective_generated_{metric}"
        
        if not all(col in df.columns for col in [src_col, tgt_col, cs_col]):
            print(f"Missing columns for {metric}, skipping distribution plot...")
            continue
        
        plot_data = df[[src_col, tgt_col, cs_col]].copy()
        plot_data.columns = ['Source', 'Target', 'Code-Switched']
        plot_data_melted = pd.melt(plot_data, var_name='Type', value_name=f'{metric.title()} Score')
        
        # KDE plot
        plt.figure(figsize=(12, 7))
        ax = sns.kdeplot(data=plot_data_melted, x=f'{metric.title()} Score', hue='Type', fill=True, common_norm=False, alpha=0.5)
        
        for i, text_type in enumerate(['Source', 'Target', 'Code-Switched']):
            median_val = plot_data[text_type].median()
            plt.axvline(x=median_val, color=sns.color_palette()[i], linestyle='--', 
                        label=f'{text_type} Median: {median_val:.3f}')
        
        plt.title(f"Input Prompts - {metric.replace('_', ' ').title()} Score Distribution")
        plt.xlabel(f"{metric.replace('_', ' ').title()} Score")
        plt.ylabel("Density")
        plt.legend(title='Input Type')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"input_prompts_{metric}_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Boxplot
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=plot_data_melted, x='Type', y=f'{metric.title()} Score')
        plt.title(f"Input Prompts - {metric.replace('_', ' ').title()} Score Boxplot")
        plt.xlabel("Input Type")
        plt.ylabel(f"{metric.replace('_', ' ').title()} Score")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"input_prompts_{metric}_boxplot.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Violin plot
        plt.figure(figsize=(12, 7))
        sns.violinplot(data=plot_data_melted, x='Type', y=f'{metric.title()} Score', inner='quart')
        plt.title(f"Input Prompts - {metric.replace('_', ' ').title()} Score Violin Plot")
        plt.xlabel("Input Type")
        plt.ylabel(f"{metric.replace('_', ' ').title()} Score")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"input_prompts_{metric}_violinplot.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created distribution plots for {metric}")

def main():
    parser = argparse.ArgumentParser(description="Generate analysis plots for input prompts toxicity scores.")
    parser.add_argument("--input_csv", type=str, required=True, 
                       help="Path to the CSV file containing input prompts toxicity scores")
    parser.add_argument("--output_dir", type=str, required=True, 
                       help="Directory to save plots")
    parser.add_argument("--analyses", type=str, nargs='+', default=['all'], 
                        choices=['distribution', 'stats', 'all'],
                        help="Types of analyses to perform")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    run_all = 'all' in args.analyses
    run_distribution = run_all or 'distribution' in args.analyses
    run_stats = run_all or 'stats' in args.analyses
    
    try:
        df = load_data(args.input_csv)
        
        if run_distribution:
            print("Running distribution analysis...")
            create_distribution_plots(df, args.output_dir)
        
        if run_stats:
            print("Running statistical tests...")
            calculate_statistical_tests(df, args.output_dir)
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return
    
    print(f"\nCompleted generating plots and analyses. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 