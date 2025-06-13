import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from scipy import stats
from collections import Counter
import re
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class TweetsEDA:
    def __init__(self, data_dir="tweets_outputs/perspective_small", output_dir="tweets_outputs/eda_analysis"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Core columns to analyze (model-agnostic names)
        self.core_columns = [
            'generated', 'primary_key', 'sentiment',
            'english_word_count', 'total_hindi_count', 'total_words',
            'total_hindi_percent', 'english_percent'
        ]
        
        # Model-specific patterns
        self.models = ['llama3', 'llama31', 'aya']
        
    def load_data(self):
        """Load all CSV files from the perspective_small directory."""
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {self.data_dir}")
            return {}
        
        datasets = {}
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            print(f"Loading {filename}...")
            
            try:
                df = pd.read_csv(file_path)
                # Extract model name from filename
                model_name = None
                for model in self.models:
                    if model in filename.lower():
                        model_name = model
                        break
                
                if model_name:
                    datasets[model_name] = df
                    print(f"  Loaded {len(df)} rows for {model_name}")
                else:
                    print(f"  Could not identify model for {filename}")
                    
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
        
        return datasets
    
    def basic_statistics(self, datasets):
        """Generate basic statistics for all datasets."""
        print("\n" + "="*50)
        print("BASIC DATASET STATISTICS")
        print("="*50)
        
        stats_summary = []
        
        for model_name, df in datasets.items():
            print(f"\n{model_name.upper()} Dataset:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {len(df.columns)}")
            
            # Check for key columns
            missing_cols = [col for col in self.core_columns if col not in df.columns]
            if missing_cols:
                print(f"  Missing core columns: {missing_cols}")
            
            # Model-specific columns
            continuation_col = f"{model_name}_continuation"
            toxicity_col = f"perspective_{model_name}_continuation_toxicity"
            identity_col = f"perspective_{model_name}_continuation_identity_attack"
            insult_col = f"perspective_{model_name}_continuation_insult"
            
            model_cols = [continuation_col, toxicity_col, identity_col, insult_col]
            available_model_cols = [col for col in model_cols if col in df.columns]
            print(f"  Available model-specific columns: {len(available_model_cols)}/{len(model_cols)}")
            
            # Basic stats
            stats_summary.append({
                'model': model_name,
                'n_rows': len(df),
                'n_cols': len(df.columns),
                'missing_core_cols': len(missing_cols),
                'available_model_cols': len(available_model_cols)
            })
        
        # Save summary
        stats_df = pd.DataFrame(stats_summary)
        stats_df.to_csv(os.path.join(self.output_dir, 'dataset_summary.csv'), index=False)
        
        return stats_summary
    
    def analyze_language_composition(self, datasets):
        """Analyze language composition across datasets."""
        print("\n" + "="*50)
        print("LANGUAGE COMPOSITION ANALYSIS")
        print("="*50)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        all_lang_stats = []
        
        for idx, (model_name, df) in enumerate(datasets.items()):
            if idx >= len(axes):
                break
                
            # Language composition statistics
            if all(col in df.columns for col in ['total_hindi_percent', 'english_percent', 'total_words']):
                print(f"\n{model_name.upper()} Language Statistics:")
                
                hindi_stats = df['total_hindi_percent'].describe()
                english_stats = df['english_percent'].describe()
                word_stats = df['total_words'].describe()
                
                print(f"  Hindi percentage - Mean: {hindi_stats['mean']:.1f}%, Std: {hindi_stats['std']:.1f}%")
                print(f"  English percentage - Mean: {english_stats['mean']:.1f}%, Std: {english_stats['std']:.1f}%")
                print(f"  Total words - Mean: {word_stats['mean']:.1f}, Median: {word_stats['50%']:.1f}")
                
                # Store for comparison
                for _, row in df.iterrows():
                    all_lang_stats.append({
                        'model': model_name,
                        'hindi_percent': row.get('total_hindi_percent', 0),
                        'english_percent': row.get('english_percent', 0),
                        'total_words': row.get('total_words', 0)
                    })
                
                # Create distribution plot
                ax = axes[idx]
                ax.hist(df['total_hindi_percent'].dropna(), bins=30, alpha=0.7, label='Hindi %', color='orange')
                ax.hist(df['english_percent'].dropna(), bins=30, alpha=0.7, label='English %', color='blue')
                ax.set_title(f'{model_name.upper()} - Language Distribution')
                ax.set_xlabel('Percentage')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'language_composition_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Combined language analysis
        if all_lang_stats:
            lang_df = pd.DataFrame(all_lang_stats)
            
            # Language composition by model
            plt.figure(figsize=(12, 8))
            
            # Box plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            sns.boxplot(data=lang_df, x='model', y='hindi_percent', ax=ax1)
            ax1.set_title('Hindi Percentage by Model')
            ax1.set_ylabel('Hindi Percentage')
            
            sns.boxplot(data=lang_df, x='model', y='english_percent', ax=ax2)
            ax2.set_title('English Percentage by Model')
            ax2.set_ylabel('English Percentage')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'language_composition_by_model.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save language statistics
            lang_summary = lang_df.groupby('model').agg({
                'hindi_percent': ['mean', 'std', 'median'],
                'english_percent': ['mean', 'std', 'median'],
                'total_words': ['mean', 'std', 'median']
            }).round(2)
            
            lang_summary.to_csv(os.path.join(self.output_dir, 'language_composition_summary.csv'))
    
    def analyze_sentiment_distribution(self, datasets):
        """Analyze sentiment distribution across datasets."""
        print("\n" + "="*50)
        print("SENTIMENT ANALYSIS")
        print("="*50)
        
        fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 6))
        if len(datasets) == 1:
            axes = [axes]
        
        all_sentiment_data = []
        
        for idx, (model_name, df) in enumerate(datasets.items()):
            if 'sentiment' in df.columns:
                sentiment_counts = df['sentiment'].value_counts()
                print(f"\n{model_name.upper()} Sentiment Distribution:")
                for sentiment, count in sentiment_counts.items():
                    percentage = (count / len(df)) * 100
                    print(f"  {sentiment}: {count} ({percentage:.1f}%)")
                
                # Store for combined analysis
                for sentiment in df['sentiment'].dropna():
                    all_sentiment_data.append({'model': model_name, 'sentiment': sentiment})
                
                # Create pie chart
                ax = axes[idx] if len(datasets) > 1 else axes[0]
                colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
                wedges, texts, autotexts = ax.pie(sentiment_counts.values, labels=sentiment_counts.index, 
                                                 autopct='%1.1f%%', colors=colors[:len(sentiment_counts)])
                ax.set_title(f'{model_name.upper()} - Sentiment Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sentiment_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Combined sentiment analysis
        if all_sentiment_data:
            sentiment_df = pd.DataFrame(all_sentiment_data)
            
            # Cross-tabulation
            sentiment_crosstab = pd.crosstab(sentiment_df['model'], sentiment_df['sentiment'], normalize='index') * 100
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(sentiment_crosstab, annot=True, fmt='.1f', cmap='Blues', cbar_kws={'label': 'Percentage'})
            plt.title('Sentiment Distribution Across Models (%)')
            plt.ylabel('Model')
            plt.xlabel('Sentiment')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'sentiment_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save sentiment summary
            sentiment_crosstab.to_csv(os.path.join(self.output_dir, 'sentiment_distribution_summary.csv'))
    
    def analyze_toxicity_metrics(self, datasets):
        """Analyze toxicity metrics across models."""
        print("\n" + "="*50)
        print("TOXICITY ANALYSIS")
        print("="*50)
        
        toxicity_data = []
        
        for model_name, df in datasets.items():
            toxicity_col = f"perspective_{model_name}_continuation_toxicity"
            identity_col = f"perspective_{model_name}_continuation_identity_attack"
            insult_col = f"perspective_{model_name}_continuation_insult"
            
            available_tox_cols = [col for col in [toxicity_col, identity_col, insult_col] if col in df.columns]
            
            if available_tox_cols:
                print(f"\n{model_name.upper()} Toxicity Metrics:")
                
                for col in available_tox_cols:
                    metric_name = col.split('_')[-1]
                    values = df[col].dropna()
                    
                    if len(values) > 0:
                        print(f"  {metric_name.title()}:")
                        print(f"    Mean: {values.mean():.3f}")
                        print(f"    Median: {values.median():.3f}")
                        print(f"    Std: {values.std():.3f}")
                        print(f"    High toxicity (>0.5): {(values > 0.5).sum()} ({(values > 0.5).mean()*100:.1f}%)")
                        
                        # Store for comparison
                        for value in values:
                            toxicity_data.append({
                                'model': model_name,
                                'metric': metric_name,
                                'value': value
                            })
        
        if toxicity_data:
            tox_df = pd.DataFrame(toxicity_data)
            
            # Create toxicity distribution plots
            metrics = tox_df['metric'].unique()
            fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 6))
            if len(metrics) == 1:
                axes = [axes]
            
            for idx, metric in enumerate(metrics):
                metric_data = tox_df[tox_df['metric'] == metric]
                ax = axes[idx] if len(metrics) > 1 else axes[0]
                
                sns.boxplot(data=metric_data, x='model', y='value', ax=ax)
                ax.set_title(f'{metric.title()} Distribution by Model')
                ax.set_ylabel(f'{metric.title()} Score')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'toxicity_distributions.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Toxicity correlation matrix
            tox_pivot = tox_df.pivot_table(index=tox_df.index, columns=['model', 'metric'], values='value')
            tox_corr = tox_pivot.corr()
            
            if not tox_corr.empty:
                plt.figure(figsize=(12, 10))
                sns.heatmap(tox_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
                plt.title('Toxicity Metrics Correlation Matrix')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'toxicity_correlation_matrix.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            # Save toxicity summary
            tox_summary = tox_df.groupby(['model', 'metric'])['value'].agg(['mean', 'median', 'std', 'min', 'max']).round(3)
            tox_summary.to_csv(os.path.join(self.output_dir, 'toxicity_summary.csv'))
    
    def analyze_text_characteristics(self, datasets):
        """Analyze characteristics of generated text."""
        print("\n" + "="*50)
        print("TEXT CHARACTERISTICS ANALYSIS")
        print("="*50)
        
        text_stats = []
        
        for model_name, df in datasets.items():
            continuation_col = f"{model_name}_continuation"
            
            if continuation_col in df.columns:
                print(f"\n{model_name.upper()} Text Characteristics:")
                
                # Text length analysis
                text_lengths = df[continuation_col].dropna().astype(str).apply(len)
                word_counts = df[continuation_col].dropna().astype(str).apply(lambda x: len(x.split()))
                
                print(f"  Character length - Mean: {text_lengths.mean():.1f}, Median: {text_lengths.median():.1f}")
                print(f"  Word count - Mean: {word_counts.mean():.1f}, Median: {word_counts.median():.1f}")
                
                # Store for comparison
                for i, text in enumerate(df[continuation_col].dropna().astype(str)):
                    text_stats.append({
                        'model': model_name,
                        'char_length': len(text),
                        'word_count': len(text.split()),
                        'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0
                    })
        
        if text_stats:
            text_df = pd.DataFrame(text_stats)
            
            # Text characteristics comparison
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Character length
            sns.boxplot(data=text_df, x='model', y='char_length', ax=axes[0])
            axes[0].set_title('Character Length Distribution')
            axes[0].set_ylabel('Character Length')
            
            # Word count
            sns.boxplot(data=text_df, x='model', y='word_count', ax=axes[1])
            axes[1].set_title('Word Count Distribution')
            axes[1].set_ylabel('Word Count')
            
            # Average word length
            sns.boxplot(data=text_df, x='model', y='avg_word_length', ax=axes[2])
            axes[2].set_title('Average Word Length Distribution')
            axes[2].set_ylabel('Average Word Length')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'text_characteristics.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save text characteristics summary
            text_summary = text_df.groupby('model').agg({
                'char_length': ['mean', 'median', 'std'],
                'word_count': ['mean', 'median', 'std'],
                'avg_word_length': ['mean', 'median', 'std']
            }).round(2)
            
            text_summary.to_csv(os.path.join(self.output_dir, 'text_characteristics_summary.csv'))
    
    def analyze_generated_text_language_distribution(self, datasets):
        """Analyze language distribution in generated text (continuations)."""
        print("\n" + "="*50)
        print("GENERATED TEXT LANGUAGE DISTRIBUTION ANALYSIS")
        print("="*50)
        
        # Simple language detection for generated text
        def count_language_words(text):
            """Simple heuristic to count Hindi vs English words in generated text."""
            if pd.isna(text) or not isinstance(text, str):
                return {'hindi_words': 0, 'english_words': 0, 'total_words': 0}
            
            words = text.split()
            hindi_words = 0
            english_words = 0
            
            for word in words:
                # Remove punctuation for analysis
                clean_word = re.sub(r'[^\w\s]', '', word.lower())
                if not clean_word:
                    continue
                    
                # Simple heuristic: if word contains Devanagari or common Hindi romanized patterns
                if any('\u0900' <= char <= '\u097F' for char in clean_word):
                    hindi_words += 1
                elif any(pattern in clean_word for pattern in ['hai', 'hain', 'kar', 'kya', 'aur', 'mein', 'se', 'ko', 'ki', 'ka', 'ke']):
                    hindi_words += 1
                else:
                    english_words += 1
            
            return {
                'hindi_words': hindi_words,
                'english_words': english_words,
                'total_words': len(words)
            }
        
        all_generated_lang_data = []
        
        for model_name, df in datasets.items():
            continuation_col = f"{model_name}_continuation"
            
            if continuation_col in df.columns:
                print(f"\n{model_name.upper()} Generated Text Language Analysis:")
                
                # Analyze language distribution in generated text
                lang_stats = df[continuation_col].apply(count_language_words)
                
                # Extract statistics
                hindi_counts = [stat['hindi_words'] for stat in lang_stats]
                english_counts = [stat['english_words'] for stat in lang_stats]
                total_counts = [stat['total_words'] for stat in lang_stats]
                
                # Calculate percentages
                hindi_percentages = [h/t*100 if t > 0 else 0 for h, t in zip(hindi_counts, total_counts)]
                english_percentages = [e/t*100 if t > 0 else 0 for e, t in zip(english_counts, total_counts)]
                
                print(f"  Average Hindi words per generation: {np.mean(hindi_counts):.1f}")
                print(f"  Average English words per generation: {np.mean(english_counts):.1f}")
                print(f"  Average total words per generation: {np.mean(total_counts):.1f}")
                print(f"  Average Hindi percentage: {np.mean(hindi_percentages):.1f}%")
                print(f"  Average English percentage: {np.mean(english_percentages):.1f}%")
                
                # Store for visualization
                for i, (h_count, e_count, t_count, h_pct, e_pct) in enumerate(zip(
                    hindi_counts, english_counts, total_counts, hindi_percentages, english_percentages)):
                    all_generated_lang_data.append({
                        'model': model_name,
                        'hindi_words': h_count,
                        'english_words': e_count,
                        'total_words': t_count,
                        'hindi_percentage': h_pct,
                        'english_percentage': e_pct
                    })
        
        if all_generated_lang_data:
            gen_lang_df = pd.DataFrame(all_generated_lang_data)
            
            # Create visualizations
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Hindi word count distribution
            sns.boxplot(data=gen_lang_df, x='model', y='hindi_words', ax=axes[0,0])
            axes[0,0].set_title('Hindi Words in Generated Text')
            axes[0,0].set_ylabel('Hindi Word Count')
            
            # English word count distribution
            sns.boxplot(data=gen_lang_df, x='model', y='english_words', ax=axes[0,1])
            axes[0,1].set_title('English Words in Generated Text')
            axes[0,1].set_ylabel('English Word Count')
            
            # Hindi percentage distribution
            sns.boxplot(data=gen_lang_df, x='model', y='hindi_percentage', ax=axes[1,0])
            axes[1,0].set_title('Hindi Percentage in Generated Text')
            axes[1,0].set_ylabel('Hindi Percentage')
            
            # English percentage distribution
            sns.boxplot(data=gen_lang_df, x='model', y='english_percentage', ax=axes[1,1])
            axes[1,1].set_title('English Percentage in Generated Text')
            axes[1,1].set_ylabel('English Percentage')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'generated_text_language_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Histogram of language mixing patterns
            fig, axes = plt.subplots(1, len(gen_lang_df['model'].unique()), figsize=(5*len(gen_lang_df['model'].unique()), 6))
            if len(gen_lang_df['model'].unique()) == 1:
                axes = [axes]
            
            for idx, model in enumerate(gen_lang_df['model'].unique()):
                model_data = gen_lang_df[gen_lang_df['model'] == model]
                ax = axes[idx] if len(gen_lang_df['model'].unique()) > 1 else axes[0]
                
                ax.hist(model_data['hindi_percentage'], bins=20, alpha=0.7, label='Hindi %', color='orange')
                ax.hist(model_data['english_percentage'], bins=20, alpha=0.7, label='English %', color='blue')
                ax.set_title(f'{model.upper()} - Generated Text Language Mix')
                ax.set_xlabel('Percentage')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'generated_text_language_histograms.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save summary statistics
            gen_lang_summary = gen_lang_df.groupby('model').agg({
                'hindi_words': ['mean', 'median', 'std'],
                'english_words': ['mean', 'median', 'std'],
                'total_words': ['mean', 'median', 'std'],
                'hindi_percentage': ['mean', 'median', 'std'],
                'english_percentage': ['mean', 'median', 'std']
            }).round(2)
            
            gen_lang_summary.to_csv(os.path.join(self.output_dir, 'generated_text_language_summary.csv'))
    
    def analyze_perspective_score_distributions(self, datasets):
        """Analyze the distribution of Perspective API scores in detail."""
        print("\n" + "="*50)
        print("PERSPECTIVE API SCORE DISTRIBUTIONS")
        print("="*50)
        
        all_perspective_data = []
        
        for model_name, df in datasets.items():
            toxicity_col = f"perspective_{model_name}_continuation_toxicity"
            identity_col = f"perspective_{model_name}_continuation_identity_attack"
            insult_col = f"perspective_{model_name}_continuation_insult"
            
            perspective_cols = [toxicity_col, identity_col, insult_col]
            available_cols = [col for col in perspective_cols if col in df.columns]
            
            if available_cols:
                print(f"\n{model_name.upper()} Perspective Score Analysis:")
                
                for col in available_cols:
                    metric_name = col.split('_')[-1]
                    values = df[col].dropna()
                    
                    if len(values) > 0:
                        print(f"  {metric_name.title()} Score Distribution:")
                        print(f"    Min: {values.min():.3f}")
                        print(f"    25th percentile: {values.quantile(0.25):.3f}")
                        print(f"    Median: {values.median():.3f}")
                        print(f"    75th percentile: {values.quantile(0.75):.3f}")
                        print(f"    Max: {values.max():.3f}")
                        print(f"    Mean: {values.mean():.3f}")
                        print(f"    Std: {values.std():.3f}")
                        
                        # Score ranges
                        low_scores = (values <= 0.2).sum()
                        medium_scores = ((values > 0.2) & (values <= 0.5)).sum()
                        high_scores = (values > 0.5).sum()
                        
                        print(f"    Low scores (≤0.2): {low_scores} ({low_scores/len(values)*100:.1f}%)")
                        print(f"    Medium scores (0.2-0.5): {medium_scores} ({medium_scores/len(values)*100:.1f}%)")
                        print(f"    High scores (>0.5): {high_scores} ({high_scores/len(values)*100:.1f}%)")
                        
                        # Store for visualization
                        for value in values:
                            all_perspective_data.append({
                                'model': model_name,
                                'metric': metric_name,
                                'score': value,
                                'score_category': 'Low (≤0.2)' if value <= 0.2 else 'Medium (0.2-0.5)' if value <= 0.5 else 'High (>0.5)'
                            })
        
        if all_perspective_data:
            perspective_df = pd.DataFrame(all_perspective_data)
            
            # Create detailed distribution plots
            metrics = perspective_df['metric'].unique()
            
            # Histogram distributions
            fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 6*len(metrics)))
            if len(metrics) == 1:
                axes = [axes]
            
            for idx, metric in enumerate(metrics):
                metric_data = perspective_df[perspective_df['metric'] == metric]
                ax = axes[idx] if len(metrics) > 1 else axes[0]
                
                for model in metric_data['model'].unique():
                    model_metric_data = metric_data[metric_data['model'] == model]
                    ax.hist(model_metric_data['score'], bins=30, alpha=0.7, label=model, density=True)
                
                ax.set_title(f'{metric.title()} Score Distribution')
                ax.set_xlabel('Score')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.axvline(x=0.2, color='orange', linestyle='--', alpha=0.7, label='Low/Medium threshold')
                ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Medium/High threshold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'perspective_score_distributions.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Score category analysis
            category_summary = perspective_df.groupby(['model', 'metric', 'score_category']).size().unstack(fill_value=0)
            category_percentages = category_summary.div(category_summary.sum(axis=1), axis=0) * 100
            
            # Stacked bar chart for score categories
            fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 6*len(metrics)))
            if len(metrics) == 1:
                axes = [axes]
            
            colors = ['green', 'orange', 'red']
            
            for idx, metric in enumerate(metrics):
                ax = axes[idx] if len(metrics) > 1 else axes[0]
                
                metric_percentages = category_percentages.loc[category_percentages.index.get_level_values('metric') == metric]
                metric_percentages.index = metric_percentages.index.droplevel('metric')
                
                if not metric_percentages.empty:
                    metric_percentages.plot(kind='bar', stacked=True, ax=ax, color=colors[:len(metric_percentages.columns)])
                    ax.set_title(f'{metric.title()} Score Categories by Model')
                    ax.set_ylabel('Percentage')
                    ax.set_xlabel('Model')
                    ax.legend(title='Score Category')
                    ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'perspective_score_categories.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Box plots for detailed comparison
            fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 6))
            if len(metrics) == 1:
                axes = [axes]
            
            for idx, metric in enumerate(metrics):
                metric_data = perspective_df[perspective_df['metric'] == metric]
                ax = axes[idx] if len(metrics) > 1 else axes[0]
                
                sns.boxplot(data=metric_data, x='model', y='score', ax=ax)
                ax.set_title(f'{metric.title()} Score Distribution')
                ax.set_ylabel('Score')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7)
                ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'perspective_score_boxplots.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save detailed summary
            perspective_summary = perspective_df.groupby(['model', 'metric'])['score'].agg([
                'count', 'mean', 'median', 'std', 'min', 'max',
                lambda x: (x <= 0.2).sum(),  # low count
                lambda x: ((x > 0.2) & (x <= 0.5)).sum(),  # medium count
                lambda x: (x > 0.5).sum()  # high count
            ]).round(3)
            
            perspective_summary.columns = ['count', 'mean', 'median', 'std', 'min', 'max', 'low_count', 'medium_count', 'high_count']
            perspective_summary.to_csv(os.path.join(self.output_dir, 'perspective_detailed_summary.csv'))
    
    def analyze_correlations(self, datasets):
        """Analyze correlations between different metrics."""
        print("\n" + "="*50)
        print("CORRELATION ANALYSIS")
        print("="*50)
        
        for model_name, df in datasets.items():
            print(f"\n{model_name.upper()} Correlations:")
            
            # Identify numeric columns for correlation
            numeric_cols = []
            
            # Language composition
            lang_cols = ['total_hindi_percent', 'english_percent', 'total_words']
            numeric_cols.extend([col for col in lang_cols if col in df.columns])
            
            # Toxicity metrics
            tox_cols = [f"perspective_{model_name}_continuation_toxicity",
                       f"perspective_{model_name}_continuation_identity_attack",
                       f"perspective_{model_name}_continuation_insult"]
            numeric_cols.extend([col for col in tox_cols if col in df.columns])
            
            if len(numeric_cols) > 1:
                # Compute correlation matrix
                corr_matrix = df[numeric_cols].corr()
                
                # Create heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                           square=True, cbar_kws={'label': 'Correlation Coefficient'})
                plt.title(f'{model_name.upper()} - Correlation Matrix')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'{model_name}_correlation_matrix.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # Save correlation matrix
                corr_matrix.to_csv(os.path.join(self.output_dir, f'{model_name}_correlations.csv'))
                
                # Print significant correlations
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.3:  # Threshold for "significant" correlation
                            col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                            print(f"  {col1} <-> {col2}: {corr_val:.3f}")
    
    def generate_summary_report(self, datasets):
        """Generate a comprehensive summary report."""
        print("\n" + "="*50)
        print("GENERATING SUMMARY REPORT")
        print("="*50)
        
        report_lines = []
        report_lines.append("TWEETS DATASET - EXPLORATORY DATA ANALYSIS REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Number of datasets analyzed: {len(datasets)}")
        report_lines.append("")
        
        # Dataset overview
        report_lines.append("DATASET OVERVIEW:")
        report_lines.append("-" * 20)
        for model_name, df in datasets.items():
            report_lines.append(f"{model_name.upper()}:")
            report_lines.append(f"  - Rows: {len(df):,}")
            report_lines.append(f"  - Columns: {len(df.columns)}")
            
            # Check data completeness
            if 'sentiment' in df.columns:
                sentiment_completeness = (1 - df['sentiment'].isna().mean()) * 100
                report_lines.append(f"  - Sentiment data completeness: {sentiment_completeness:.1f}%")
            
            # Language composition
            if 'total_hindi_percent' in df.columns:
                avg_hindi = df['total_hindi_percent'].mean()
                avg_english = df['english_percent'].mean()
                report_lines.append(f"  - Average Hindi content: {avg_hindi:.1f}%")
                report_lines.append(f"  - Average English content: {avg_english:.1f}%")
            
            # Toxicity overview
            tox_col = f"perspective_{model_name}_continuation_toxicity"
            if tox_col in df.columns:
                high_tox_rate = (df[tox_col] > 0.5).mean() * 100
                avg_tox = df[tox_col].mean()
                report_lines.append(f"  - Average toxicity score: {avg_tox:.3f}")
                report_lines.append(f"  - High toxicity rate (>0.5): {high_tox_rate:.1f}%")
            
            report_lines.append("")
        
        # Key findings
        report_lines.append("KEY FINDINGS:")
        report_lines.append("-" * 15)
        report_lines.append("1. Language composition varies across models")
        report_lines.append("2. Toxicity levels differ between model-generated continuations")
        report_lines.append("3. Sentiment distribution shows model-specific patterns")
        report_lines.append("4. Text characteristics (length, complexity) vary by model")
        report_lines.append("")
        
        # Files generated
        report_lines.append("GENERATED FILES:")
        report_lines.append("-" * 16)
        output_files = [
            "dataset_summary.csv",
            "language_composition_distributions.png",
            "language_composition_by_model.png",
            "language_composition_summary.csv",
            "sentiment_distribution.png",
            "sentiment_heatmap.png",
            "sentiment_distribution_summary.csv",
            "toxicity_distributions.png",
            "toxicity_correlation_matrix.png",
            "toxicity_summary.csv",
            "text_characteristics.png",
            "text_characteristics_summary.csv",
            "generated_text_language_distribution.png",
            "generated_text_language_histograms.png",
            "generated_text_language_summary.csv",
            "perspective_score_distributions.png",
            "perspective_score_categories.png",
            "perspective_score_boxplots.png",
            "perspective_detailed_summary.csv"
        ]
        
        for model_name in datasets.keys():
            output_files.extend([
                f"{model_name}_correlation_matrix.png",
                f"{model_name}_correlations.csv"
            ])
        
        for file in output_files:
            if os.path.exists(os.path.join(self.output_dir, file)):
                report_lines.append(f"  ✓ {file}")
        
        # Save report
        report_path = os.path.join(self.output_dir, 'eda_summary_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Summary report saved to: {report_path}")
    
    def run_full_analysis(self):
        """Run the complete exploratory data analysis."""
        print("Starting Tweets Dataset Exploratory Data Analysis...")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Load data
        datasets = self.load_data()
        
        if not datasets:
            print("No datasets loaded. Exiting.")
            return
        
        # Run all analyses
        self.basic_statistics(datasets)
        self.analyze_language_composition(datasets)
        self.analyze_sentiment_distribution(datasets)
        self.analyze_toxicity_metrics(datasets)
        self.analyze_text_characteristics(datasets)
        self.analyze_generated_text_language_distribution(datasets)
        self.analyze_perspective_score_distributions(datasets)
        self.analyze_correlations(datasets)
        self.generate_summary_report(datasets)
        
        print(f"\n{'='*50}")
        print("EXPLORATORY DATA ANALYSIS COMPLETE!")
        print(f"All results saved to: {self.output_dir}")
        print(f"{'='*50}")

def main():
    """Main function to run the EDA."""
    eda = TweetsEDA()
    eda.run_full_analysis()

if __name__ == "__main__":
    main() 