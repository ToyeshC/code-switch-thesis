import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os
import argparse

def load_combined_data(models):
    """Load combined perspective data for all models."""
    combined_data = {}
    for model in models:
        file_path = f"new_outputs/perspective_combined_full/{model}_combined_perspective.csv"
        try:
            combined_data[model] = pd.read_csv(file_path)
            print(f"Loaded data for {model}: {len(combined_data[model])} rows")
        except FileNotFoundError:
            print(f"Warning: Could not find file for {model}: {file_path}")
    return combined_data

def calculate_mean_toxicity_scores(df):
    """Calculate mean toxicity scores for original texts (src, tgt, generated)."""
    results = {}
    
    # Original text toxicity scores
    toxicity_cols = {
        'src': 'perspective_src_toxicity',
        'tgt': 'perspective_tgt_toxicity', 
        'generated': 'perspective_generated_toxicity'
    }
    
    for text_type, col in toxicity_cols.items():
        if col in df.columns:
            scores = df[col].dropna()
            results[text_type] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'count': len(scores)
            }
        else:
            print(f"Warning: Column {col} not found")
            results[text_type] = {'mean': np.nan, 'std': np.nan, 'count': 0}
    
    return results

def calculate_continuation_toxicity_scores(df, model_name):
    """Calculate mean toxicity scores for continuations by model."""
    results = {}
    
    # Continuation toxicity scores
    continuation_cols = {
        'src': f'perspective_{model_name}_continuation_src_toxicity',
        'tgt': f'perspective_{model_name}_continuation_tgt_toxicity',
        'generated': f'perspective_{model_name}_continuation_generated_toxicity'
    }
    
    for text_type, col in continuation_cols.items():
        if col in df.columns:
            scores = df[col].dropna()
            results[f'{model_name}_continuation_{text_type}'] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'count': len(scores)
            }
        else:
            print(f"Warning: Column {col} not found")
            results[f'{model_name}_continuation_{text_type}'] = {'mean': np.nan, 'std': np.nan, 'count': 0}
    
    return results

def calculate_relative_differences(score_a, score_b, name_a, name_b):
    """Calculate absolute difference and percentage change between two scores."""
    # Handle division by zero for percentage change
    if score_b == 0:
        if score_a == 0:
            percentage_change = 0.0
        else:
            percentage_change = np.inf if score_a > 0 else -np.inf
    else:
        percentage_change = ((score_a - score_b) / score_b) * 100
    
    absolute_diff = score_a - score_b
    
    return {
        'comparison': f'{name_a} vs {name_b}',
        'score_a': score_a,
        'score_b': score_b,
        'absolute_difference': absolute_diff,
        'percentage_change': percentage_change
    }

def calculate_correlations_with_language_features(df):
    """Calculate correlations between toxicity and language composition features."""
    # Toxicity score
    toxicity_col = 'perspective_generated_toxicity'
    
    # Language features
    language_features = [
        'total_words',
        'hindi_word_count',
        'english_word_count',
        'romanized_hindi_count',
        'hindi_percent',
        'english_percent',
        'romanized_hindi_percent',
        'total_hindi_percent'
    ]
    
    correlations = {}
    
    if toxicity_col in df.columns:
        toxicity_scores = df[toxicity_col].dropna()
        
        for feature in language_features:
            if feature in df.columns:
                feature_values = df[feature].dropna()
                
                # Align the data
                common_index = toxicity_scores.index.intersection(feature_values.index)
                if len(common_index) > 1:
                    aligned_toxicity = toxicity_scores.loc[common_index]
                    aligned_feature = feature_values.loc[common_index]
                    
                    corr_coef, p_value = pearsonr(aligned_toxicity, aligned_feature)
                    correlations[feature] = {
                        'correlation': corr_coef,
                        'p_value': p_value,
                        'n_samples': len(common_index)
                    }
                else:
                    correlations[feature] = {'correlation': np.nan, 'p_value': np.nan, 'n_samples': 0}
            else:
                print(f"Warning: Feature column {feature} not found")
                correlations[feature] = {'correlation': np.nan, 'p_value': np.nan, 'n_samples': 0}
    
    return correlations

def create_original_vs_continuation_comparison(combined_data, output_dir):
    """Create visualization comparing original texts vs their continuations."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Original vs Continuation Toxicity Comparison', fontsize=16)
    
    # Define text type mappings
    text_type_mapping = {
        'src': 'English',
        'tgt': 'Hindi',
        'generated': 'Code-Switched'
    }
    
    # Define color palette using different shades of blue
    original_color = '#1f77b4'  # Darker blue
    continuation_color = '#7fb3d5'  # Lighter blue
    
    # Calculate overall averages for original texts
    overall_means = {'src': [], 'tgt': [], 'generated': []}
    for model_name, df in combined_data.items():
        original_scores = calculate_mean_toxicity_scores(df)
        for text_type in ['src', 'tgt', 'generated']:
            if not np.isnan(original_scores[text_type]['mean']):
                overall_means[text_type].append(original_scores[text_type]['mean'])
    
    overall_averages = {
        k: np.mean(v) if v else 0 for k, v in overall_means.items()
    }
    
    comparison_pairs = [
        ('src', 'English'),
        ('tgt', 'Hindi'), 
        ('generated', 'Code-Switched')
    ]
    
    for idx, (text_type, label) in enumerate(comparison_pairs):
        ax = axes[idx]
        
        # Get overall average for original text
        original_mean = overall_averages[text_type]
        
        # Collect continuation data across all models
        continuation_means = []
        model_names = []
        
        for model_name, df in combined_data.items():
            continuation_col = f'perspective_{model_name}_continuation_{text_type}_toxicity'
            
            if continuation_col in df.columns:
                continuation_scores = df[continuation_col].dropna()
                
                if len(continuation_scores) > 0:
                    continuation_means.append(continuation_scores.mean())
                    model_names.append(model_name.upper())
        
        if continuation_means:
            x = np.arange(len(model_names))
            width = 0.35
            
            # Plot original mean as a horizontal line
            ax.axhline(y=original_mean, color=original_color, linestyle='-', 
                      label=f'Original {label}', alpha=0.7)
            
            # Plot continuation means as bars
            bars = ax.bar(x, continuation_means, width, 
                         label=f'{label} Continuation', alpha=0.7, color=continuation_color)
            
            # Add value labels on bars
            for bar, mean in zip(bars, continuation_means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Add original mean label
            ax.text(-0.5, original_mean + 0.005, f'{original_mean:.3f}', 
                   ha='right', va='bottom', fontsize=9, color=original_color)
            
            ax.set_ylabel('Mean Perspective Toxicity Score')
            ax.set_title(f'{label} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Calculate and display percentage changes
            for i, (cont, model) in enumerate(zip(continuation_means, model_names)):
                if original_mean != 0:
                    pct_change = ((cont - original_mean) / original_mean) * 100
                    # Add percentage change annotation
                    ax.text(i, max(cont, original_mean) + 0.02, f'{pct_change:+.1f}%', 
                           ha='center', va='bottom', fontsize=8, weight='bold',
                           color='red' if pct_change > 0 else 'green')
        else:
            ax.text(0.5, 0.5, f'No data available for {label}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{label} Comparison')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/original_vs_continuation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_continuation_difference_heatmap(combined_data, output_dir):
    """Create heatmap showing percentage differences between original and continuation toxicity."""
    models = list(combined_data.keys())
    text_types = ['src', 'tgt', 'generated']
    text_labels = ['English', 'Hindi', 'Code-Switched']
    
    # Calculate overall averages for original texts
    overall_means = {'src': [], 'tgt': [], 'generated': []}
    for model_name, df in combined_data.items():
        original_scores = calculate_mean_toxicity_scores(df)
        for text_type in ['src', 'tgt', 'generated']:
            if not np.isnan(original_scores[text_type]['mean']):
                overall_means[text_type].append(original_scores[text_type]['mean'])
    
    overall_averages = {
        k: np.mean(v) if v else 0 for k, v in overall_means.items()
    }
    
    # Create matrix for percentage differences
    diff_matrix = np.zeros((len(models), len(text_types)))
    
    for i, model_name in enumerate(models):
        df = combined_data[model_name]
        for j, text_type in enumerate(text_types):
            continuation_col = f'perspective_{model_name}_continuation_{text_type}_toxicity'
            
            if continuation_col in df.columns:
                continuation_scores = df[continuation_col].dropna()
                
                if len(continuation_scores) > 0:
                    cont_mean = continuation_scores.mean()
                    orig_mean = overall_averages[text_type]
                    
                    if orig_mean != 0:
                        pct_change = ((cont_mean - orig_mean) / orig_mean) * 100
                        diff_matrix[i, j] = pct_change
                    else:
                        diff_matrix[i, j] = np.nan
                else:
                    diff_matrix[i, j] = np.nan
            else:
                diff_matrix[i, j] = np.nan
    
    # Create heatmap
    plt.figure(figsize=(8, 6))
    
    # Create DataFrame for easier plotting
    diff_df = pd.DataFrame(diff_matrix, 
                          index=[m.upper() for m in models], 
                          columns=text_labels)
    
    # Create heatmap with diverging colormap
    sns.heatmap(diff_df, annot=True, cmap='RdBu_r', center=0, 
               fmt='.1f', cbar_kws={'label': 'Percentage Change (%)'})
    
    plt.title('Toxicity Change: Continuation vs Original\n(Positive = More Toxic, Negative = Less Toxic)')
    plt.ylabel('Model')
    plt.xlabel('Text Type')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/continuation_difference_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_visualizations(combined_data, output_dir):
    """Create comprehensive visualizations for toxicity analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    
    # Define color palette using different shades of blue
    colors = {
        'original': '#1f77b4',  # Darker blue
        'continuation': '#7fb3d5',  # Lighter blue
        'boxplot': '#aec7e8',  # Even lighter blue
        'heatmap': 'RdBu_r',  # Red-Blue diverging colormap
        'model_colors': ['#1f77b4', '#4c9cd4', '#7fb3d5']  # Different shades for models
    }
    
    # Define text type mappings
    text_type_mapping = {
        'src': 'English',
        'tgt': 'Hindi',
        'generated': 'Code-Switched'
    }
    
    # 1. Bar Chart: Mean toxicity scores for original texts
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Collect data across all models
    all_original_means = {'src': [], 'tgt': [], 'generated': []}
    
    for model_name, df in combined_data.items():
        original_scores = calculate_mean_toxicity_scores(df)
        for text_type in ['src', 'tgt', 'generated']:
            if not np.isnan(original_scores[text_type]['mean']):
                all_original_means[text_type].append(original_scores[text_type]['mean'])
    
    # Calculate overall means
    overall_means = {k: np.mean(v) if v else 0 for k, v in all_original_means.items()}
    overall_stds = {k: np.std(v) if len(v) > 1 else 0 for k, v in all_original_means.items()}
    
    text_types = list(overall_means.keys())
    means = list(overall_means.values())
    stds = list(overall_stds.values())
    
    # Use mapped labels for x-axis
    labels = [text_type_mapping[t] for t in text_types]
    
    bars = ax.bar(labels, means, yerr=stds, capsize=5, alpha=0.7, color=colors['original'])
    ax.set_ylabel('Mean Perspective Toxicity Score')
    ax.set_title('Mean Toxicity Scores: Original Texts')
    ax.set_ylim(0, max(means) * 1.2 if means else 1)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/original_texts_toxicity_means.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Grouped Bar Chart: Continuation toxicity scores by model
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = list(combined_data.keys())
    text_types = ['src', 'tgt', 'generated']
    text_labels = [text_type_mapping[t] for t in text_types]
    
    x = np.arange(len(text_types))
    width = 0.25
    
    for i, model in enumerate(models):
        df = combined_data[model]
        continuation_scores = calculate_continuation_toxicity_scores(df, model)
        
        means = [continuation_scores[f'{model}_continuation_{t}']['mean'] for t in text_types]
        means = [m if not np.isnan(m) else 0 for m in means]  # Handle NaN values
        
        bars = ax.bar(x + i * width, means, width, label=f'{model.upper()} continuations', 
                     alpha=0.7, color=colors['model_colors'][i % len(colors['model_colors'])])
        
        # Add value labels
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Mean Perspective Toxicity Score')
    ax.set_title('Mean Toxicity Scores: Model Continuations')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{t} Continuations' for t in text_labels])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/continuation_toxicity_by_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Original vs Continuation Comparison
    create_original_vs_continuation_comparison(combined_data, output_dir)
    
    # 4. Continuation Difference Heatmap
    create_continuation_difference_heatmap(combined_data, output_dir)
    
    # 5. Box plots for original text toxicity distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    toxicity_cols = ['perspective_src_toxicity', 'perspective_tgt_toxicity', 'perspective_generated_toxicity']
    titles = ['English (src)', 'Hindi (tgt)', 'Code-Switched (generated)']
    
    for i, (col, title) in enumerate(zip(toxicity_cols, titles)):
        data_for_plot = []
        labels_for_plot = []
        
        for model_name, df in combined_data.items():
            if col in df.columns:
                scores = df[col].dropna()
                if len(scores) > 0:
                    data_for_plot.append(scores)
                    labels_for_plot.append(model_name.upper())
        
        if data_for_plot:
            box = axes[i].boxplot(data_for_plot, labels=labels_for_plot, patch_artist=True)
            # Set box colors
            for patch in box['boxes']:
                patch.set_facecolor(colors['boxplot'])
        axes[i].set_title(f'{title} Toxicity Distribution')
        axes[i].set_ylabel('Perspective Toxicity Score')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/toxicity_distributions_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Correlation heatmap (using data from first available model)
    if combined_data:
        first_model_df = list(combined_data.values())[0]
        correlations = calculate_correlations_with_language_features(first_model_df)
        
        # Create correlation matrix for heatmap
        features = list(correlations.keys())
        corr_values = [correlations[f]['correlation'] for f in features]
        
        # Create a matrix (1 row for toxicity vs all features)
        corr_matrix = pd.DataFrame([corr_values], columns=features, index=['Toxicity'])
        
        plt.figure(figsize=(12, 3))
        sns.heatmap(corr_matrix, annot=True, cmap=colors['heatmap'], center=0, 
                   fmt='.3f', cbar_kws={'shrink': 0.8})
        plt.title('Correlation: Generated Text Toxicity vs Language Features')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/toxicity_language_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()

def generate_comprehensive_report(combined_data, output_dir):
    """Generate comprehensive analysis report."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE TOXICITY ANALYSIS REPORT")
    print("="*80)
    
    # 1. Mean Toxicity Scores for Original Texts
    print("\n1. MEAN TOXICITY SCORES - ORIGINAL TEXTS")
    print("-" * 50)
    
    all_original_scores = {}
    for model_name, df in combined_data.items():
        original_scores = calculate_mean_toxicity_scores(df)
        all_original_scores[model_name] = original_scores
        
        print(f"\nModel: {model_name.upper()}")
        for text_type, scores in original_scores.items():
            print(f"  {text_type.ljust(10)}: {scores['mean']:.6f} ± {scores['std']:.6f} (n={scores['count']})")
    
    # Calculate overall averages across models
    print(f"\nOVERALL AVERAGES ACROSS MODELS:")
    for text_type in ['src', 'tgt', 'generated']:
        means = [all_original_scores[model][text_type]['mean'] 
                for model in combined_data.keys() 
                if not np.isnan(all_original_scores[model][text_type]['mean'])]
        if means:
            overall_mean = np.mean(means)
            overall_std = np.std(means)
            print(f"  {text_type.ljust(10)}: {overall_mean:.6f} ± {overall_std:.6f}")
    
    # 2. Mean Toxicity Scores for Continuations
    print("\n\n2. MEAN TOXICITY SCORES - MODEL CONTINUATIONS")
    print("-" * 50)
    
    all_continuation_scores = {}
    for model_name, df in combined_data.items():
        continuation_scores = calculate_continuation_toxicity_scores(df, model_name)
        all_continuation_scores[model_name] = continuation_scores
        
        print(f"\nModel: {model_name.upper()}")
        for cont_type, scores in continuation_scores.items():
            print(f"  {cont_type.ljust(25)}: {scores['mean']:.6f} ± {scores['std']:.6f} (n={scores['count']})")
    
    # 3. Relative Differences Analysis
    print("\n\n3. RELATIVE DIFFERENCES ANALYSIS")
    print("-" * 50)
    
    # Primary comparisons: generated vs src, generated vs tgt
    print("\nPRIMARY COMPARISONS (Original Texts):")
    for model_name in combined_data.keys():
        original_scores = all_original_scores[model_name]
        
        print(f"\nModel: {model_name.upper()}")
        
        # generated vs src
        gen_vs_src = calculate_relative_differences(
            original_scores['generated']['mean'],
            original_scores['src']['mean'],
            'generated', 'src'
        )
        print(f"  Generated vs English (src):")
        print(f"    Absolute difference: {gen_vs_src['absolute_difference']:.6f}")
        print(f"    Percentage change: {gen_vs_src['percentage_change']:.2f}%")
        
        # generated vs tgt
        gen_vs_tgt = calculate_relative_differences(
            original_scores['generated']['mean'],
            original_scores['tgt']['mean'],
            'generated', 'tgt'
        )
        print(f"  Generated vs Hindi (tgt):")
        print(f"    Absolute difference: {gen_vs_tgt['absolute_difference']:.6f}")
        print(f"    Percentage change: {gen_vs_tgt['percentage_change']:.2f}%")
    
    # Continuation comparisons
    print(f"\nCONTINUATION COMPARISONS:")
    for model_name in combined_data.keys():
        continuation_scores = all_continuation_scores[model_name]
        
        print(f"\nModel: {model_name.upper()}")
        
        # continuation_generated vs continuation_src
        cont_gen_mean = continuation_scores[f'{model_name}_continuation_generated']['mean']
        cont_src_mean = continuation_scores[f'{model_name}_continuation_src']['mean']
        
        if not (np.isnan(cont_gen_mean) or np.isnan(cont_src_mean)):
            cont_gen_vs_src = calculate_relative_differences(
                cont_gen_mean, cont_src_mean,
                f'{model_name}_cont_generated', f'{model_name}_cont_src'
            )
            print(f"  Continuation Generated vs Continuation English:")
            print(f"    Absolute difference: {cont_gen_vs_src['absolute_difference']:.6f}")
            print(f"    Percentage change: {cont_gen_vs_src['percentage_change']:.2f}%")
        
        # continuation_generated vs continuation_tgt
        cont_tgt_mean = continuation_scores[f'{model_name}_continuation_tgt']['mean']
        
        if not (np.isnan(cont_gen_mean) or np.isnan(cont_tgt_mean)):
            cont_gen_vs_tgt = calculate_relative_differences(
                cont_gen_mean, cont_tgt_mean,
                f'{model_name}_cont_generated', f'{model_name}_cont_tgt'
            )
            print(f"  Continuation Generated vs Continuation Hindi:")
            print(f"    Absolute difference: {cont_gen_vs_tgt['absolute_difference']:.6f}")
            print(f"    Percentage change: {cont_gen_vs_tgt['percentage_change']:.2f}%")
    
    # Original vs Continuation comparisons
    print(f"\nORIGINAL vs CONTINUATION COMPARISONS:")
    for model_name in combined_data.keys():
        original_scores = all_original_scores[model_name]
        continuation_scores = all_continuation_scores[model_name]
        
        print(f"\nModel: {model_name.upper()}")
        
        for text_type in ['src', 'tgt', 'generated']:
            orig_mean = original_scores[text_type]['mean']
            cont_mean = continuation_scores[f'{model_name}_continuation_{text_type}']['mean']
            
            if not (np.isnan(orig_mean) or np.isnan(cont_mean)):
                comparison = calculate_relative_differences(
                    cont_mean, orig_mean,
                    f'{model_name}_cont_{text_type}', f'orig_{text_type}'
                )
                print(f"  {text_type.title()} Continuation vs Original:")
                print(f"    Absolute difference: {comparison['absolute_difference']:.6f}")
                print(f"    Percentage change: {comparison['percentage_change']:.2f}%")
    
    # 4. Correlation Analysis
    print("\n\n4. CORRELATION ANALYSIS - TOXICITY vs LANGUAGE FEATURES")
    print("-" * 50)
    
    # Use first available model for correlation analysis
    first_model_df = list(combined_data.values())[0]
    correlations = calculate_correlations_with_language_features(first_model_df)
    
    print(f"\nCorrelations with Generated Text Toxicity:")
    for feature, corr_data in correlations.items():
        if not np.isnan(corr_data['correlation']):
            significance = " ***" if corr_data['p_value'] < 0.001 else " **" if corr_data['p_value'] < 0.01 else " *" if corr_data['p_value'] < 0.05 else ""
            print(f"  {feature.ljust(25)}: r = {corr_data['correlation']:+.4f}, p = {corr_data['p_value']:.4f}{significance} (n={corr_data['n_samples']})")
    
    # Interpretation of correlations
    print(f"\nCORRELATION INTERPRETATIONS:")
    print(f"  • Positive correlations suggest higher feature values associate with higher toxicity")
    print(f"  • Negative correlations suggest higher feature values associate with lower toxicity")
    print(f"  • Statistical significance: *** p<0.001, ** p<0.01, * p<0.05")
    
    # 5. Key Findings and Interpretations
    print("\n\n5. KEY FINDINGS AND INTERPRETATIONS")
    print("-" * 50)
    
    # Determine if code-switched input is less toxic
    overall_means = {}
    for text_type in ['src', 'tgt', 'generated']:
        means = [all_original_scores[model][text_type]['mean'] 
                for model in combined_data.keys() 
                if not np.isnan(all_original_scores[model][text_type]['mean'])]
        if means:
            overall_means[text_type] = np.mean(means)
    
    if all(t in overall_means for t in ['src', 'tgt', 'generated']):
        src_mean = overall_means['src']
        tgt_mean = overall_means['tgt']
        gen_mean = overall_means['generated']
        
        print(f"\nCODE-SWITCHING TOXICITY ANALYSIS:")
        if gen_mean < src_mean and gen_mean < tgt_mean:
            print(f"  ✓ CONFIRMED: Code-switched inputs show lower toxicity than both English and Hindi")
            print(f"    Generated: {gen_mean:.6f} < English: {src_mean:.6f} < Hindi: {tgt_mean:.6f}")
        elif gen_mean < src_mean:
            print(f"  ⚠ PARTIAL: Code-switched inputs less toxic than English but not Hindi")
            print(f"    Generated: {gen_mean:.6f} < English: {src_mean:.6f}, Hindi: {tgt_mean:.6f}")
        elif gen_mean < tgt_mean:
            print(f"  ⚠ PARTIAL: Code-switched inputs less toxic than Hindi but not English")
            print(f"    Generated: {gen_mean:.6f} < Hindi: {tgt_mean:.6f}, English: {src_mean:.6f}")
        else:
            print(f"  ✗ NOT CONFIRMED: Code-switched inputs do not show consistently lower toxicity")
            print(f"    Generated: {gen_mean:.6f}, English: {src_mean:.6f}, Hindi: {tgt_mean:.6f}")
    
    # 6. Limitations and Considerations
    print("\n\n6. LIMITATIONS AND CONSIDERATIONS")
    print("-" * 50)
    print(f"""
IMPORTANT LIMITATIONS:
• Perspective API Performance: The API was primarily trained on English text and may have 
  reduced accuracy for Hindi and code-switched content. Biases in toxicity detection for 
  non-English content should be considered when interpreting results.

• Code-switching Detection: The API's ability to properly evaluate code-switched text has 
  evolved over time. Current results may not fully capture the nuances of multilingual toxicity.

• Sample Size: Statistical robustness depends on adequate sample sizes. Check 'n=' values above.

• Cultural Context: Toxicity perceptions may vary across cultures and languages, which 
  standardized APIs may not fully capture.

STATISTICAL CONSIDERATIONS:
• For robust analysis, consider performing paired t-tests or Wilcoxon signed-rank tests 
  for paired comparisons (e.g., original vs continuation within same texts).

• ANOVA or Kruskal-Wallis tests could be used for comparing multiple groups 
  (e.g., src vs tgt vs generated).

• P-values and confidence intervals would provide statistical significance testing.

• Effect sizes (Cohen's d) would indicate practical significance beyond statistical significance.
    """)
    
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Comprehensive toxicity analysis for code-switched content")
    parser.add_argument("--models", nargs="+", default=["aya", "llama3", "llama31"],
                      help="List of models to analyze (default: aya llama3 llama31)")
    parser.add_argument("--output-dir", default="new_outputs/toxicity_analysis_full",
                      help="Output directory for results (default: new_outputs/toxicity_analysis_full)")
    
    args = parser.parse_args()
    
    # Load data
    combined_data = load_combined_data(args.models)
    
    if not combined_data:
        print("No data loaded. Please check your file paths.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualizations
    print("Creating visualizations...")
    create_visualizations(combined_data, f"{args.output_dir}/plots")
    
    # Generate comprehensive report
    print("Generating comprehensive analysis...")
    generate_comprehensive_report(combined_data, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")
    print(f"Plots saved to {args.output_dir}/plots/")

if __name__ == "__main__":
    main() 