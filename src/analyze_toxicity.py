import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def analyze_toxicity(input_file, output_dir):
    """
    Analyze toxicity metrics from Perspective API results.
    
    Args:
        input_file (str): Path to the CSV file with toxicity analysis results
        output_dir (str): Directory to save visualization outputs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the data
    print(f"Reading toxicity data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing values detected:")
        print(missing_values[missing_values > 0])
        # Fill missing values with 0
        df.fillna(0, inplace=True)
        print("Missing values filled with 0.")
    
    # Extract numeric columns (all except 'comment')
    numeric_cols = [col for col in df.columns if col != 'comment']
    
    # Compute descriptive statistics
    stats = df[numeric_cols].describe()
    
    # Count high and low toxicity comments
    high_toxicity = (df['toxicity'] > 0.7).sum()
    low_toxicity = (df['toxicity'] < 0.2).sum()
    
    # Find category with highest mean score
    highest_category = stats.loc['mean'].idxmax()
    highest_mean = stats.loc['mean', highest_category]
    
    # Generate correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Toxicity Metrics')
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, 'toxicity_correlation_heatmap.png')
    plt.savefig(heatmap_path)
    print(f"Correlation heatmap saved to {heatmap_path}")
    
    # Generate histograms for each toxicity type
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(2, 3, i)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col.replace("_", " ").title()}')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    histogram_path = os.path.join(output_dir, 'toxicity_distributions.png')
    plt.savefig(histogram_path)
    print(f"Toxicity distributions saved to {histogram_path}")
    
    # Print summary report
    print("\n===== Toxicity Analysis Summary =====")
    print(f"Total comments analyzed: {len(df)}")
    print(f"High toxicity comments (>0.7): {high_toxicity} ({high_toxicity/len(df)*100:.1f}%)")
    print(f"Low toxicity comments (<0.2): {low_toxicity} ({low_toxicity/len(df)*100:.1f}%)")
    print(f"Highest mean score category: {highest_category} ({highest_mean:.4f})")
    
    print("\nDescriptive Statistics:")
    print(stats.round(4))
    
    return {
        'total_comments': len(df),
        'high_toxicity': high_toxicity,
        'low_toxicity': low_toxicity,
        'highest_category': highest_category,
        'highest_mean': highest_mean,
        'stats': stats
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze toxicity metrics from Perspective API results')
    parser.add_argument('--input', '-i', required=True, help='Path to the CSV file with toxicity analysis')
    parser.add_argument('--output_dir', '-o', default='data/output', help='Directory to save visualization outputs')
    
    args = parser.parse_args()
    analyze_toxicity(args.input, args.output_dir)

if __name__ == "__main__":
    main() 