import pandas as pd
import argparse
import os
import shutil

def main():
    """
    Utility script to prepare existing model outputs for toxicity analysis.
    This script will:
    1. Locate and copy existing model outputs to the analysis directory
    2. Verify and format the files to ensure they have the correct column structure
    """
    parser = argparse.ArgumentParser(description="Prepare existing model outputs for toxicity analysis")
    parser.add_argument("--input", required=True, help="Path to the input CSV file with original prompts")
    parser.add_argument("--llama_output", required=True, help="Path to the existing LLaMA output file or 'placeholder'")
    parser.add_argument("--aya_output", required=True, help="Path to the existing Aya output file or 'placeholder'")
    parser.add_argument("--output_dir", required=True, help="Directory to save prepared files")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define output file paths
    llama_output_file = os.path.join(args.output_dir, 'llama_responses.csv')
    aya_output_file = os.path.join(args.output_dir, 'aya_responses.csv')
    
    # Read the original prompts
    print(f"Reading original prompts from {args.input}")
    prompts_df = pd.read_csv(args.input)
    
    # Ensure prompts have a 'sentence' column (expected by the analysis script)
    if 'sentence' not in prompts_df.columns:
        if 'comment' in prompts_df.columns:
            prompts_df = prompts_df.rename(columns={'comment': 'sentence'})
        else:
            # Try to find a text column
            text_cols = [col for col in prompts_df.columns if prompts_df[col].dtype == 'object']
            if text_cols:
                prompts_df = prompts_df.rename(columns={text_cols[0]: 'sentence'})
            else:
                raise ValueError(f"Could not find a suitable text column in {args.input}")
    
    # Read LLaMA outputs
    if args.llama_output == "placeholder" or not os.path.exists(args.llama_output):
        print("No valid LLaMA output file provided. Creating placeholder responses...")
        # Create a placeholder file with empty responses
        llama_df = pd.DataFrame({
            'prompt': prompts_df['sentence'],
            'response': ["[Placeholder LLaMA response]"] * len(prompts_df),
            'model': 'LLaMA'
        })
        llama_df.to_csv(llama_output_file, index=False)
        print(f"Saved placeholder LLaMA responses to {llama_output_file}")
    else:
        print(f"Reading LLaMA outputs from {args.llama_output}")
        try:
            llama_df = pd.read_csv(args.llama_output)
            
            # Check if the file has the structure we need
            if 'prompt' not in llama_df.columns and 'response' not in llama_df.columns:
                print("LLaMA output file doesn't have expected columns. Attempting to reformat...")
                
                # Try to identify text columns
                text_cols = [col for col in llama_df.columns if llama_df[col].dtype == 'object']
                
                if len(text_cols) >= 1:
                    # If there's only one text column, it's likely the response
                    if len(text_cols) == 1:
                        # Create a new DataFrame with prompts from the original file
                        llama_df = pd.DataFrame({
                            'prompt': prompts_df['sentence'],
                            'response': llama_df[text_cols[0]],
                            'model': 'LLaMA'
                        })
                    # If there are multiple columns, try to determine which is prompt vs response
                    else:
                        # Simple heuristic: first text column might be prompt, second might be response
                        llama_df = pd.DataFrame({
                            'prompt': llama_df[text_cols[0]],
                            'response': llama_df[text_cols[1]],
                            'model': 'LLaMA'
                        })
                else:
                    raise ValueError(f"Could not identify suitable text columns in {args.llama_output}")
            
            # Ensure model column is present
            if 'model' not in llama_df.columns:
                llama_df['model'] = 'LLaMA'
                
            # Save the formatted output
            llama_df.to_csv(llama_output_file, index=False)
            print(f"Saved formatted LLaMA responses to {llama_output_file}")
                
        except Exception as e:
            print(f"Error processing LLaMA output: {e}")
            print("Creating a placeholder LLaMA output file...")
            
            # Create a placeholder file
            llama_df = pd.DataFrame({
                'prompt': prompts_df['sentence'],
                'response': ["[Error: Could not load response]"] * len(prompts_df),
                'model': 'LLaMA'
            })
            llama_df.to_csv(llama_output_file, index=False)
            print(f"Saved placeholder LLaMA responses to {llama_output_file}")
    
    # Read Aya outputs
    if args.aya_output == "placeholder" or not os.path.exists(args.aya_output):
        print("No valid Aya output file provided. Creating placeholder responses...")
        # Create a placeholder file with empty responses
        aya_df = pd.DataFrame({
            'prompt': prompts_df['sentence'],
            'response': ["[Placeholder Aya response]"] * len(prompts_df),
            'model': 'Aya'
        })
        aya_df.to_csv(aya_output_file, index=False)
        print(f"Saved placeholder Aya responses to {aya_output_file}")
    else:
        print(f"Reading Aya outputs from {args.aya_output}")
        try:
            aya_df = pd.read_csv(args.aya_output)
            
            # Check if the file has the structure we need
            if 'prompt' not in aya_df.columns and 'response' not in aya_df.columns:
                print("Aya output file doesn't have expected columns. Attempting to reformat...")
                
                # Try to identify text columns
                text_cols = [col for col in aya_df.columns if aya_df[col].dtype == 'object']
                
                if len(text_cols) >= 1:
                    # If there's only one text column, it's likely the response
                    if len(text_cols) == 1:
                        # Create a new DataFrame with prompts from the original file
                        aya_df = pd.DataFrame({
                            'prompt': prompts_df['sentence'],
                            'response': aya_df[text_cols[0]],
                            'model': 'Aya'
                        })
                    # If there are multiple columns, try to determine which is prompt vs response
                    else:
                        # Simple heuristic: first text column might be prompt, second might be response
                        aya_df = pd.DataFrame({
                            'prompt': aya_df[text_cols[0]],
                            'response': aya_df[text_cols[1]],
                            'model': 'Aya'
                        })
                else:
                    raise ValueError(f"Could not identify suitable text columns in {args.aya_output}")
            
            # Ensure model column is present
            if 'model' not in aya_df.columns:
                aya_df['model'] = 'Aya'
                
            # Save the formatted output
            aya_df.to_csv(aya_output_file, index=False)
            print(f"Saved formatted Aya responses to {aya_output_file}")
                
        except Exception as e:
            print(f"Error processing Aya output: {e}")
            print("Creating a placeholder Aya output file...")
            
            # Create a placeholder file
            aya_df = pd.DataFrame({
                'prompt': prompts_df['sentence'],
                'response': ["[Error: Could not load response]"] * len(prompts_df),
                'model': 'Aya'
            })
            aya_df.to_csv(aya_output_file, index=False)
            print(f"Saved placeholder Aya responses to {aya_output_file}")
    
    print(f"\nModel outputs prepared for analysis in {args.output_dir}")
    print("You can now run the analyze_model_toxicity.py script with the --skip_generation flag")

if __name__ == "__main__":
    main() 