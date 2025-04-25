import pandas as pd
import argparse
import os
import json
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import torch
from datetime import datetime, timedelta

def load_model(model_name, device="cuda" if torch.cuda.is_available() else "cpu", ignore_mismatched_sizes=False):
    """
    Load a Llama model and tokenizer.
    
    Args:
        model_name (str): Name of the model to load
        device (str): Device to load the model on
        ignore_mismatched_sizes (bool): Whether to ignore mismatched sizes when loading the model
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    
    # Determine if we're loading a Llama model
    is_llama = "llama" in model_name.lower() or "meta-llama" in model_name.lower()
    
    # Try different approaches in sequence
    attempts = []
    
    if is_llama:
        # Approach 1: Use specific Llama classes
        attempts.append(lambda: (
            LlamaTokenizer.from_pretrained(model_name),
            LlamaForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        ))
    
    # Approach 2: Auto classes with ignore_mismatched_sizes
    attempts.append(lambda: (
        AutoTokenizer.from_pretrained(model_name),
        AutoModelForCausalLM.from_pretrained(
            model_name,
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    ))
    
    # Approach 3: Auto classes with low memory settings
    attempts.append(lambda: (
        AutoTokenizer.from_pretrained(model_name),
        AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    ))
    
    # Approach 4: Final attempt with minimal settings
    attempts.append(lambda: (
        AutoTokenizer.from_pretrained(model_name),
        AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto"
        )
    ))
    
    # Try each approach in sequence
    last_error = None
    for i, attempt_func in enumerate(attempts):
        try:
            print(f"Attempt {i+1}/{len(attempts)} to load model...")
            tokenizer, model = attempt_func()
            print(f"Successfully loaded model and tokenizer!")
            return model, tokenizer
        except Exception as e:
            last_error = e
            print(f"Error in attempt {i+1}: {e}")
    
    # If all attempts failed, raise the last error
    raise ValueError(f"Failed to load model after {len(attempts)} attempts. Last error: {last_error}")

def generate_continuation(model, tokenizer, prompt, max_length=100, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Generate a continuation for a prompt using a Llama model.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        prompt (str): The prompt to continue
        max_length (int): Maximum length of the generated text
        device (str): Device to run the model on
        
    Returns:
        str: The generated continuation
    """
    # Tokenize the prompt - no need to move to device with device_map="auto"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate the continuation
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    # Decode the generated text
    continuation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the continuation
    continuation = continuation[len(prompt):].strip()
    
    return continuation

def process_data(input_file, output_file, model_name, max_rows=None, quota_limit=60, quota_window=60, ignore_mismatched_sizes=False):
    """
    Process a CSV file with a Llama model and save the results.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the output CSV file
        model_name (str): Name of the model to use
        max_rows (int): Maximum number of rows to process (None for all rows)
        quota_limit (int): Maximum number of requests per window
        quota_window (int): Time window in seconds for quota limit
        ignore_mismatched_sizes (bool): Whether to ignore mismatched sizes when loading the model
    """
    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Limit the number of rows if specified
    if max_rows is not None and max_rows < len(df):
        print(f"Limiting to first {max_rows} rows out of {len(df)} total rows")
        df = df.head(max_rows)
    
    # Load the model - we're using device_map="auto" in load_model
    model, tokenizer = load_model(model_name, ignore_mismatched_sizes=ignore_mismatched_sizes)
    
    # Add column for the continuation if it doesn't exist
    continuation_col = f"{model_name.split('/')[-1]}_continuation"
    if continuation_col not in df.columns:
        df[continuation_col] = ""
    
    # Initialize quota tracking
    request_times = []
    processed_count = 0
    
    # Process each row
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating continuations with {model_name}"):
        # Skip if already processed
        if df.at[i, continuation_col]:
            continue
        
        # Get the prompt
        prompt = row['generated']
        
        # Check if we need to wait for quota reset
        current_time = datetime.now()
        request_times = [t for t in request_times if current_time - t < timedelta(seconds=quota_window)]
        
        if len(request_times) >= quota_limit:
            wait_time = (request_times[0] + timedelta(seconds=quota_window) - current_time).total_seconds()
            if wait_time > 0:
                print(f"\nQuota limit reached. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                request_times = []
        
        # Generate the continuation - device is handled by device_map="auto"
        continuation = generate_continuation(model, tokenizer, prompt)
        
        # Update the DataFrame
        df.at[i, continuation_col] = continuation
        
        # Track the request
        request_times.append(current_time)
        processed_count += 1
        
        # Save progress every 10 rows
        if processed_count % 10 == 0:
            print(f"\nSaving progress after {processed_count} rows...")
            df.to_csv(output_file, index=False)
        
        # Sleep to avoid rate limiting
        time.sleep(1)
    
    # Save the final results
    print(f"Saving results to: {output_file}")
    df.to_csv(output_file, index=False)
    
    print(f"Processed {len(df)} rows")

def compare_continuations(code_switched_file, output_dir):
    """
    Compare continuations from different models.
    
    Args:
        code_switched_file (str): Path to the code-switched data CSV file with continuations
        output_dir (str): Directory to save the comparison results
    """
    print(f"Reading code-switched file: {code_switched_file}")
    df = pd.read_csv(code_switched_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the continuation columns
    continuation_cols = [col for col in df.columns if col.endswith("_continuation")]
    
    # Create a DataFrame for the comparison
    comparison_data = []
    
    for col in continuation_cols:
        # Get the continuations
        continuations = df[col].values
        
        # Calculate statistics
        mean_length = np.mean([len(c.split()) for c in continuations])
        std_length = np.std([len(c.split()) for c in continuations])
        
        # Add to comparison data
        comparison_data.append({
            "Model": col.replace("_continuation", ""),
            "Mean_Length": mean_length,
            "Std_Length": std_length
        })
    
    # Create a DataFrame for the comparison
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save the comparison
    comparison_file = os.path.join(output_dir, "continuation_comparison.csv")
    comparison_df.to_csv(comparison_file, index=False)
    
    print(f"Saved comparison to: {comparison_file}")
    
    # Create visualizations
    # Create a figure
    plt.figure(figsize=(10, 6))
    
    # Create a bar plot
    plt.bar(comparison_df["Model"], comparison_df["Mean_Length"])
    plt.errorbar(comparison_df["Model"], comparison_df["Mean_Length"], yerr=comparison_df["Std_Length"], fmt="none", color="black")
    plt.title("Mean Continuation Length by Model")
    plt.xlabel("Model")
    plt.ylabel("Mean Length (words)")
    plt.xticks(rotation=45)
    
    # Save the figure
    fig_file = os.path.join(output_dir, "continuation_length_comparison.png")
    plt.savefig(fig_file, bbox_inches="tight")
    plt.close()
    
    print(f"Saved continuation length comparison plot to: {fig_file}")

def main():
    parser = argparse.ArgumentParser(description="Run Llama models on filtered code-switched data and get continuations")
    parser.add_argument("--input_file", required=True, help="Path to the input CSV file")
    parser.add_argument("--output_file", required=True, help="Path to save the output CSV file")
    parser.add_argument("--model_name", required=True, help="Name of the model to use")
    parser.add_argument("--max_rows", type=int, default=None, help="Maximum number of rows to process")
    parser.add_argument("--output_dir", help="Directory to save the comparison results")
    parser.add_argument("--quota_limit", type=int, default=60, help="Maximum number of requests per window")
    parser.add_argument("--quota_window", type=int, default=60, help="Time window in seconds for quota limit")
    parser.add_argument("--ignore_mismatched_sizes", action="store_true", help="Ignore mismatched sizes when loading the model")
    
    args = parser.parse_args()
    
    # Process the data
    process_data(args.input_file, args.output_file, args.model_name, args.max_rows, args.quota_limit, args.quota_window, args.ignore_mismatched_sizes)
    
    # Compare continuations if output directory is provided
    if args.output_dir:
        compare_continuations(args.output_file, args.output_dir)

if __name__ == "__main__":
    main() 