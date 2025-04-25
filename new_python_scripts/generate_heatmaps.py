import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def generate_heatmaps(combined_scores_file, output_dir, model_name):
    """Generates correlation heatmaps for Perspective scores."""
    
    print(f"\n--- Generating Heatmaps for {model_name} ---")
    print(f"Reading combined scores from: {combined_scores_file}")
    
    try:
        df = pd.read_csv(combined_scores_file)
    except FileNotFoundError:
        print(f"Error: Combined scores file not found: {combined_scores_file}")
        return
    except Exception as e:
        print(f"Error reading combined scores file: {e}")
        return

    # Define score columns for correlation - CORRECTED MAPPING
    score_cols = {
        'Toxicity (En)': 'toxicity_src',            # Use _src for English
        'Identity Attack (En)': 'identity_attack_src',
        'Insult (En)': 'insult_src',
        'Toxicity (Hi)': 'toxicity_tgt',            # Use _tgt for Hindi
        'Identity Attack (Hi)': 'identity_attack_tgt',
        'Insult (Hi)': 'insult_tgt',
        'Toxicity (CS)': 'toxicity_cs',
        'Identity Attack (CS)': 'identity_attack_cs',
        'Insult (CS)': 'insult_cs'
    }
    
    # Filter dataframe to only include existing score columns and handle potential errors (-1)
    available_cols = {label: col for label, col in score_cols.items() if col in df.columns}
    plot_df = df[list(available_cols.values())].copy()
    
    # Replace potential error score (-1) with NaN for correlation calculation
    plot_df = plot_df.replace(-1.0, pd.NA)
    plot_df = plot_df.dropna() # Drop rows with any NaN/NA after replacement
    
    if plot_df.empty:
        print("Warning: No valid data remaining after handling errors/NaNs for heatmap generation.")
        return
        
    # Rename columns for better labels on the heatmap
    plot_df.columns = [label for label, col in available_cols.items() if col in plot_df.columns]

    # Get labels for filtering
    en_labels = sorted([label for label in plot_df.columns if '(En)' in label])
    hi_labels = sorted([label for label in plot_df.columns if '(Hi)' in label])
    cs_labels = sorted([label for label in plot_df.columns if '(CS)' in label])
    
    # --- Heatmap 1: English vs Code-Switched --- 
    if en_labels and cs_labels: # Check if both lists have labels
        # Calculate full correlation for relevant columns
        corr_en_cs_full = plot_df[en_labels + cs_labels].corr()
        # Filter to keep only En rows and CS columns
        filtered_corr_en_cs = corr_en_cs_full.loc[en_labels, cs_labels]

        # Adjust figsize based on matrix shape
        aspect_ratio = len(cs_labels) / len(en_labels)
        fig_width = 8
        fig_height = max(4, fig_width / aspect_ratio * 0.8) # Adjust height dynamically

        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(filtered_corr_en_cs, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title(f'{model_name.upper()} - Correlation: English (Rows) vs Code-Switched (Cols)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        heatmap_en_cs_file = os.path.join(output_dir, f"{model_name}_heatmap_en_cs.png")
        plt.savefig(heatmap_en_cs_file)
        plt.close()
        print(f"Saved En vs CS heatmap to {heatmap_en_cs_file}")
    else:
        print("Skipping En vs CS heatmap: Missing English or CS columns.")

    # --- Heatmap 2: Hindi vs Code-Switched --- 
    if hi_labels and cs_labels: # Check if both lists have labels
        # Calculate full correlation for relevant columns
        corr_hi_cs_full = plot_df[hi_labels + cs_labels].corr()
        # Filter to keep only Hi rows and CS columns
        filtered_corr_hi_cs = corr_hi_cs_full.loc[hi_labels, cs_labels]

        # Adjust figsize based on matrix shape
        aspect_ratio = len(cs_labels) / len(hi_labels)
        fig_width = 8
        fig_height = max(4, fig_width / aspect_ratio * 0.8) # Adjust height dynamically

        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(filtered_corr_hi_cs, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title(f'{model_name.upper()} - Correlation: Hindi (Rows) vs Code-Switched (Cols)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        heatmap_hi_cs_file = os.path.join(output_dir, f"{model_name}_heatmap_hi_cs.png")
        plt.savefig(heatmap_hi_cs_file)
        plt.close()
        print(f"Saved Hi vs CS heatmap to {heatmap_hi_cs_file}")
    else:
        print("Skipping Hi vs CS heatmap: Missing Hindi or CS columns.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Correlation Heatmaps for Perspective Scores")
    parser.add_argument("--combined_scores_file", required=True, help="Path to the combined scores CSV file")
    parser.add_argument("--output_dir", required=True, help="Directory to save the heatmaps")
    parser.add_argument("--model_name", required=True, help="Short model name (e.g., llama3, aya)")
    
    args = parser.parse_args()
    
    generate_heatmaps(args.combined_scores_file, args.output_dir, args.model_name) 