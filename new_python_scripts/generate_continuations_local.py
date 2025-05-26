import pandas as pd
import argparse
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from huggingface_hub import login
import os
import time
from datetime import datetime, timedelta
import json
from cohere import Client

def load_model_and_tokenizer(model_name, use_cohere_tokenizer=False, rope_scaling=None):
    print(f"\nLoading model: {model_name}")
    print("Loading tokenizer...")
    
    if use_cohere_tokenizer:
        from cohere_tokenizer import CohereTokenizer
        tokenizer = CohereTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading model...")
    
    # Configure model loading parameters
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto"
    }
    
    # Add RoPE scaling if specified
    if rope_scaling:
        try:
            if isinstance(rope_scaling, str):
                rope_scaling = json.loads(rope_scaling)
            model_kwargs["rope_scaling"] = rope_scaling
        except json.JSONDecodeError:
            print(f"Warning: Invalid rope_scaling JSON format: {rope_scaling}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        print("Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        return None, None

def generate_continuation(model, tokenizer, text, max_tokens=50, temperature=0.7):
    try:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Generate with the same parameters as the API
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode only the new tokens
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        continuation = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return continuation.strip()
    except Exception as e:
        print(f"Error generating continuation: {str(e)}")
        return f"ERROR: Generation failed - {str(e)}"

def generate_continuations_local(input_file, output_file, model_name, text_column, max_new_tokens=50, temperature=0.7, max_rows=None, token_file=None, use_cohere_tokenizer=False, rope_scaling=None):
    """
    Generate continuations locally using transformers library.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the output CSV file
        model_name (str): Name of the Hugging Face model to use
        text_column (str): Name of the column containing input prompts.
        max_new_tokens (int): Maximum number of new tokens to generate
        temperature (float): Temperature for generation
        max_rows (int): Maximum number of rows to process (None for all)
        token_file (str, optional): Path to Hugging Face token file.
        use_cohere_tokenizer (bool): Whether to use Cohere tokenizer for Aya model
        rope_scaling (str, optional): RoPE scaling configuration as JSON string
    """
    # --- Authentication --- 
    # Handle token file if provided or use environment variable
    token = None
    if token_file:
        try:
            with open(token_file, 'r') as f:
                token = f.read().strip()
                print(f"Loaded token from {token_file}")
        except Exception as e:
            print(f"Warning: Error loading token from file: {e}")
    elif "HUGGING_FACE_HUB_TOKEN" in os.environ:
        token = os.environ["HUGGING_FACE_HUB_TOKEN"]
        print("Using Hugging Face token from environment variable")
    else:
        print("Warning: No Hugging Face token provided (file or env var).")
        print("Login via `huggingface-cli login` recommended if model is gated.")
        
    # Use login() if token is available, otherwise proceed without explicit login
    if token:
        login(token=token)
    
    # --- Model Loading --- 
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        use_cohere_tokenizer=use_cohere_tokenizer,
        rope_scaling=rope_scaling
    )
    
    if model is None or tokenizer is None:
        print(f"Skipping this model.")
        # Create a placeholder dataframe with error
        try: # Try reading input even on model load fail
            df_error = pd.read_csv(input_file)
            if max_rows is not None and max_rows < len(df_error):
                df_error = df_error.head(max_rows)
            model_short = model_name.split('/')[-1].lower().replace("-","")
            continuation_col = f"{model_short}_continuation"
            df_error[continuation_col] = f"ERROR: Model loading failed - {str(e)}"
            print(f"Saving error state to {output_file}")
            # Ensure directory exists before saving error file
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df_error.to_csv(output_file, index=False)
        except Exception as read_e:
             print(f"Could not read input file {input_file} to write error state: {read_e}")
        return 
    
    # --- Data Processing --- 
    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Limit the number of rows if specified
    if max_rows is not None and max_rows < len(df):
        print(f"Limiting to first {max_rows} rows out of {len(df)} total rows")
        df = df.head(max_rows)
    
    # --- Use the specified text_column --- 
    if text_column not in df.columns:
        raise ValueError(f"Specified text_column '{text_column}' not found in the input file {input_file}. Available columns: {list(df.columns)}")
    print(f"Using '{text_column}' as the input text column")
    # -------------------------------------
    
    # Extract model short name for the continuation column
    if "aya" in model_name.lower(): model_short = "aya"
    elif "llama-3.1" in model_name.lower(): model_short = "llama31"
    elif "llama-3" in model_name.lower(): model_short = "llama3"
    else: model_short = model_name.split('/')[-1].lower().replace("-","")
        
    continuation_col = f"{model_short}_continuation"
    if continuation_col not in df.columns:
        df[continuation_col] = ""
    else: # Clear existing content if column exists but we are regenerating
        print(f"Warning: Column '{continuation_col}' already exists. Overwriting.")
        df[continuation_col] = ""
        
    # --- Generation Loop --- 
    processed_count = 0
    print("Generating continuations...")
    for i in tqdm(range(len(df))):
        # Skip if already processed (redundant given clearing above, but safe)
        if pd.notna(df.at[i, continuation_col]) and df.at[i, continuation_col] != "":
            continue
        
        # Get text from the SPECIFIED column
        text = df.loc[i, text_column] 
        
        if pd.isna(text) or not isinstance(text, str) or text.strip() == '':
            df.loc[i, continuation_col] = "<EMPTY_INPUT>"
            continue
        
        # Generate continuation
        try:
            continuation = generate_continuation(
                model, 
                tokenizer, 
                text, 
                max_tokens=max_new_tokens,
                temperature=temperature
            )
            
            df.loc[i, continuation_col] = continuation
            processed_count += 1
            
            # Save progress every 10 rows
            if processed_count % 10 == 0:
                print(f"\nSaving progress after {processed_count} generated continuations...")
                # Ensure directory exists before saving progress
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                df.to_csv(output_file, index=False)
            
        except Exception as e:
            error_msg = f"ERROR: Generation failed - {e}"
            print(f"\nError generating continuation for row {i}: {error_msg}")
            df.loc[i, continuation_col] = error_msg
            
    # --- Final Save --- 
    print(f"Saving final results to {output_file}")
    # Ensure directory exists before final save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    # --- Cleanup --- 
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("Model and tokenizer unloaded, memory cleared.")
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate continuations locally using transformers")
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--output", required=True, help="Path to save the output CSV file")
    parser.add_argument("--model", required=True, help="Name of the Hugging Face model to use")
    parser.add_argument("--text_column", required=True, help="Name of the column containing input text/prompts")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--max_rows", type=int, default=None, help="Maximum number of rows to process")
    parser.add_argument("--token_file", type=str, default=None, help="Path to HF token file (optional)")
    parser.add_argument("--use_cohere_tokenizer", action="store_true", help="Use Cohere tokenizer for Aya model")
    parser.add_argument("--rope_scaling", help="RoPE scaling configuration as JSON string")
    
    args = parser.parse_args()
    
    generate_continuations_local(
        args.input, 
        args.output, 
        args.model, 
        args.text_column,
        args.max_tokens, 
        args.temperature,
        args.max_rows,
        args.token_file,
        args.use_cohere_tokenizer,
        args.rope_scaling
    ) 