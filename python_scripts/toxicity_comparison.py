import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import os

def load_data(file_path):
    try:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('//'):  # Skip empty lines and comments
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line in {file_path}: {str(e)}")
                        continue
        return data
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        raise

def extract_toxicity_scores(data):
    toxicity_scores = []
    prompts = []
    for item in data:
        if isinstance(item, dict) and 'PromptAnnotations' in item and 'Toxicity' in item['PromptAnnotations']:
            toxicity_scores.append(item['PromptAnnotations']['Toxicity'])
            if 'Prompt' in item:
                prompts.append(item['Prompt'])
    return np.array(toxicity_scores), prompts

def create_visualizations(en_toxicity, hi_toxicity, output_dir):
    # Calculate statistics
    en_mean = np.mean(en_toxicity)
    hi_mean = np.mean(hi_toxicity)
    en_std = np.std(en_toxicity)
    hi_std = np.std(hi_toxicity)
    t_stat, p_value = stats.ttest_ind(en_toxicity, hi_toxicity)
    toxicity_diff = hi_toxicity - en_toxicity

    # Format p-value for display
    if p_value < 0.0001:
        p_value_display = f"{p_value:.2e}"  # Scientific notation
    else:
        p_value_display = f"{p_value:.4f}"

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), height_ratios=[2, 1])
    
    # 1. Main comparison plot (top)
    # Create box plot
    box_plot = ax1.boxplot([en_toxicity, hi_toxicity], 
                          labels=['English', 'Hindi'],
                          patch_artist=True,
                          widths=0.6)
    
    # Customize box plot colors
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    # Add individual points with jitter
    x1 = np.random.normal(1, 0.04, size=len(en_toxicity))
    x2 = np.random.normal(2, 0.04, size=len(hi_toxicity))
    ax1.scatter(x1, en_toxicity, alpha=0.2, color='blue', s=20)
    ax1.scatter(x2, hi_toxicity, alpha=0.2, color='red', s=20)
    
    # Add mean points
    ax1.scatter([1, 2], [en_mean, hi_mean], color='black', s=100, marker='*', label='Mean')
    
    # Add statistical significance indicator
    y_max = max(np.max(en_toxicity), np.max(hi_toxicity))
    y_min = min(np.min(en_toxicity), np.min(hi_toxicity))
    y_range = y_max - y_min
    
    # Draw significance bar
    ax1.plot([1, 2], [y_max + 0.1*y_range, y_max + 0.1*y_range], 'k-', lw=1.5)
    ax1.text(1.5, y_max + 0.15*y_range, 
             f'p-value: {p_value_display}\nt-statistic: {t_stat:.2f}',
             ha='center', va='bottom',
             bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
    
    # Customize the main plot
    ax1.set_title('Toxicity Score Comparison: English vs Hindi', pad=20)
    ax1.set_ylabel('Toxicity Score')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Difference distribution plot (bottom)
    ax2.hist(toxicity_diff, bins=30, color='gray', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='No Difference')
    ax2.axvline(x=np.mean(toxicity_diff), color='blue', linestyle='-', 
                label=f'Mean Difference: {np.mean(toxicity_diff):.2f}')
    
    # Add difference statistics
    diff_stats = (f'Mean Difference: {np.mean(toxicity_diff):.2f}\n'
                 f'Std Dev: {np.std(toxicity_diff):.2f}\n'
                 f'HI > EN: {np.sum(toxicity_diff > 0)}\n'
                 f'EN > HI: {np.sum(toxicity_diff < 0)}')
    ax2.text(0.95, 0.95, diff_stats,
             transform=ax2.transAxes,
             ha='right', va='top',
             bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
    
    # Customize the difference plot
    ax2.set_title('Distribution of Toxicity Differences (Hindi - English)')
    ax2.set_xlabel('Toxicity Difference')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'toxicity_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create additional visualizations
    # 1. Violin Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=[en_toxicity, hi_toxicity], palette=['lightblue', 'lightcoral'])
    plt.xticks([0, 1], ['English', 'Hindi'])
    plt.title('Toxicity Score Density Distribution')
    plt.ylabel('Toxicity Score')
    plt.savefig(os.path.join(output_dir, 'toxicity_violin.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Scatter Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(en_toxicity, hi_toxicity, alpha=0.3)
    plt.plot([0, 5], [0, 5], 'r--', alpha=0.5, label='Equal Toxicity')
    plt.xlabel('English Toxicity')
    plt.ylabel('Hindi Toxicity')
    plt.title('English vs Hindi Toxicity Scores')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'toxicity_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'en_mean': en_mean,
        'hi_mean': hi_mean,
        'en_std': en_std,
        'hi_std': hi_std,
        't_stat': t_stat,
        'p_value': p_value,
        'toxicity_diff_mean': np.mean(toxicity_diff),
        'toxicity_diff_std': np.std(toxicity_diff),
        'hi_more_toxic': np.sum(toxicity_diff > 0),
        'en_more_toxic': np.sum(toxicity_diff < 0),
        'equal_toxicity': np.sum(toxicity_diff == 0)
    }

def main():
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(output_dir, exist_ok=True)

        # Load data
        print("Loading English data...")
        en_data = load_data('data/RTP-LX/RTP_LX_EN.json')
        print("Loading Hindi data...")
        hi_data = load_data('data/RTP-LX/RTP_LX_HI.json')

        # Extract toxicity scores and prompts
        en_toxicity, en_prompts = extract_toxicity_scores(en_data)
        hi_toxicity, hi_prompts = extract_toxicity_scores(hi_data)

        print(f"Found {len(en_toxicity)} English toxicity scores")
        print(f"Found {len(hi_toxicity)} Hindi toxicity scores")

        if len(en_toxicity) == 0 or len(hi_toxicity) == 0:
            print("Error: No toxicity scores found in one or both files")
            return

        # Create visualizations and get statistics
        stats = create_visualizations(en_toxicity, hi_toxicity, output_dir)

        # Print comprehensive statistics
        print("\nStatistical Analysis Results:")
        print(f"English Mean Toxicity: {stats['en_mean']:.2f} ± {stats['en_std']:.2f}")
        print(f"Hindi Mean Toxicity: {stats['hi_mean']:.2f} ± {stats['hi_std']:.2f}")
        print(f"t-statistic: {stats['t_stat']:.2f}")
        print(f"p-value: {stats['p_value']:.2e}")  # Changed to scientific notation
        
        print("\nTranslation Effect Analysis:")
        print(f"Average toxicity difference (HI-EN): {stats['toxicity_diff_mean']:.2f}")
        print(f"Standard deviation of difference: {stats['toxicity_diff_std']:.2f}")
        print(f"Number of cases where Hindi is more toxic: {stats['hi_more_toxic']}")
        print(f"Number of cases where English is more toxic: {stats['en_more_toxic']}")
        print(f"Number of cases with equal toxicity: {stats['equal_toxicity']}")

        print("\nVisualizations saved in:", output_dir)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 