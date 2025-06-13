import pandas as pd
import os
import argparse

def combine_model_scores(model_name):
    """Combine source and target perspective scores for a specific model."""
    print(f"\nProcessing {model_name}...")
    
    # Define file paths with 'full' directory
    src_file = f"new_outputs/perspective_continuations_full/{model_name}_src_perspective.csv"
    tgt_file = f"new_outputs/perspective_continuations_full/{model_name}_tgt_perspective.csv"
    perspective_file = f"new_outputs/perspective_full/{model_name}_continuations_perspective_local_full.csv"
    output_dir = "new_outputs/perspective_combined_full"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the files
    try:
        src_df = pd.read_csv(src_file)
        tgt_df = pd.read_csv(tgt_file)
        perspective_df = pd.read_csv(perspective_file)
    except FileNotFoundError as e:
        print(f"Error: Could not find one of the required files: {e}")
        return False
    
    # Verify the files have the same number of rows
    if len(src_df) != len(tgt_df):
        print(f"Warning: Source and target files have different numbers of rows ({len(src_df)} vs {len(tgt_df)})")
        print("Using the minimum length to ensure alignment")
        min_length = min(len(src_df), len(tgt_df))
        src_df = src_df.head(min_length)
        tgt_df = tgt_df.head(min_length)
    
    # Rename perspective columns in source dataframe
    src_perspective_cols = {
        f'perspective_{model_name}_continuation_toxicity': f'perspective_{model_name}_continuation_src_toxicity',
        f'perspective_{model_name}_continuation_identity_attack': f'perspective_{model_name}_continuation_src_identity_attack',
        f'perspective_{model_name}_continuation_insult': f'perspective_{model_name}_continuation_src_insult'
    }
    src_df = src_df.rename(columns=src_perspective_cols)
    
    # Rename perspective columns in target dataframe
    tgt_perspective_cols = {
        f'perspective_{model_name}_continuation_toxicity': f'perspective_{model_name}_continuation_tgt_toxicity',
        f'perspective_{model_name}_continuation_identity_attack': f'perspective_{model_name}_continuation_tgt_identity_attack',
        f'perspective_{model_name}_continuation_insult': f'perspective_{model_name}_continuation_tgt_insult'
    }
    tgt_df = tgt_df.rename(columns=tgt_perspective_cols)
    
    # Rename continuation columns
    if f'{model_name}_continuation' in src_df.columns:
        src_df = src_df.rename(columns={f'{model_name}_continuation': f'{model_name}_continuation_src'})
    if f'{model_name}_continuation' in tgt_df.columns:
        tgt_df = tgt_df.rename(columns={f'{model_name}_continuation': f'{model_name}_continuation_tgt'})
    
    # Rename columns from perspective file
    perspective_cols = {
        f'{model_name}_continuation': f'{model_name}_continuation_generated',
        f'perspective_{model_name}_continuation_toxicity': f'perspective_{model_name}_continuation_generated_toxicity',
        f'perspective_{model_name}_continuation_identity_attack': f'perspective_{model_name}_continuation_generated_identity_attack',
        f'perspective_{model_name}_continuation_insult': f'perspective_{model_name}_continuation_generated_insult'
    }
    perspective_df = perspective_df.rename(columns=perspective_cols)
    
    # Define the columns we want in the final output
    final_columns = [
        'src', 'tgt', 'generated', 'method', 'model', 'direction', 'primary_key',
        'hindi_word_count', 'english_word_count', 'romanized_hindi_count',
        'total_hindi_count', 'total_words', 'hindi_percent', 'romanized_hindi_percent',
        'total_hindi_percent', 'english_percent',
        'perspective_generated_toxicity', 'perspective_generated_identity_attack',
        'perspective_generated_insult',
        f'{model_name}_continuation_src', f'{model_name}_continuation_tgt', f'{model_name}_continuation_generated',
        'perspective_src_toxicity', 'perspective_src_identity_attack', 'perspective_src_insult',
        'perspective_tgt_toxicity', 'perspective_tgt_identity_attack', 'perspective_tgt_insult',
        f'perspective_{model_name}_continuation_generated_toxicity',
        f'perspective_{model_name}_continuation_generated_identity_attack',
        f'perspective_{model_name}_continuation_generated_insult',
        f'perspective_{model_name}_continuation_src_toxicity',
        f'perspective_{model_name}_continuation_src_identity_attack',
        f'perspective_{model_name}_continuation_src_insult',
        f'perspective_{model_name}_continuation_tgt_toxicity',
        f'perspective_{model_name}_continuation_tgt_identity_attack',
        f'perspective_{model_name}_continuation_tgt_insult'
    ]
    
    # Combine the dataframes
    combined_df = pd.DataFrame()
    
    # Add base columns from source dataframe
    base_columns = [col for col in final_columns if col in src_df.columns]
    combined_df[base_columns] = src_df[base_columns]
    
    # Add target-specific columns
    tgt_columns = [col for col in final_columns if col in tgt_df.columns and col not in base_columns]
    combined_df[tgt_columns] = tgt_df[tgt_columns]
    
    # Add perspective-specific columns
    perspective_columns = [col for col in final_columns if col in perspective_df.columns and col not in combined_df.columns]
    combined_df[perspective_columns] = perspective_df[perspective_columns]
    
    # Add model-specific continuation columns
    src_cont_col = f'{model_name}_continuation_src'
    tgt_cont_col = f'{model_name}_continuation_tgt'
    
    if src_cont_col in src_df.columns:
        combined_df[src_cont_col] = src_df[src_cont_col]
    if tgt_cont_col in tgt_df.columns:
        combined_df[tgt_cont_col] = tgt_df[tgt_cont_col]
    
    # Ensure all required columns exist (fill with NaN if missing)
    for col in final_columns:
        if col not in combined_df.columns:
            combined_df[col] = pd.NA
    
    # Reorder columns to match the specified order
    combined_df = combined_df[final_columns]
    
    # Save the combined file
    output_file = os.path.join(output_dir, f"{model_name}_combined_perspective.csv")
    combined_df.to_csv(output_file, index=False)
    print(f"Saved combined file to: {output_file}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Combine perspective scores for different models")
    parser.add_argument("--models", nargs="+", default=["aya", "llama3", "llama31"],
                      help="List of models to process (default: aya llama3 llama31)")
    
    args = parser.parse_args()
    
    # Process each model
    for model in args.models:
        success = combine_model_scores(model)
        if success:
            print(f"Successfully combined scores for {model}")
        else:
            print(f"Failed to combine scores for {model}")

if __name__ == "__main__":
    main() 