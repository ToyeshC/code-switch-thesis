import pandas as pd
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from huggingface_hub import login
import os

def generate_continuations(input_file, output_file, model_name, max_new_tokens=50, temperature=0.7):
    """
    Generate continuations for sentences in a CSV file using either Aya or Llama model.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the output CSV file
        model_name (str): Name of the Hugging Face model to use
        max_new_tokens (int): Maximum number of new tokens to generate
        temperature (float): Temperature for generation (higher = more creative)
    """
    # Handle authentication for gated models
    if "HUGGING_FACE_HUB_TOKEN" in os.environ:
        print("Using Hugging Face token from environment variable")
        login()  # This will use the token from the environment variable
    else:
        print("Warning: No Hugging Face token found in environment variables")
        print("If the model is gated, this may fail. Set HUGGING_FACE_HUB_TOKEN if needed.")
    
    # Model-specific configurations
    model_config = {}
    if "aya" in model_name.lower():
        print(f"Loading Aya model: {model_name}")
        # Aya-specific configurations if needed
    elif "llama" in model_name.lower():
        print(f"Loading Llama model: {model_name}")
        # Llama-specific configurations if needed
    else:
        print(f"Loading generic model: {model_name}")
    
    # Load tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto",
        **model_config
    )
    
    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Ensure 'sentence' column exists
    if 'sentence' not in df.columns:
        # Try to find a column that might contain sentences
        text_columns = [col for col in df.columns if df[col].dtype == 'object']
        if text_columns:
            print(f"'sentence' column not found. Using '{text_columns[0]}' as the sentence column.")
            sentence_column = text_columns[0]
        else:
            raise ValueError("No suitable text column found in the input CSV file.")
    else:
        sentence_column = 'sentence'
    
    # Create a new column for the continuations
    df['continuation'] = ''
    
    print("Generating continuations...")
    for i in tqdm(range(len(df))):
        sentence = df.loc[i, sentence_column]
        
        # Skip empty or NaN sentences
        if pd.isna(sentence) or sentence.strip() == '':
            continue
        
        # Prepare input for the model
        inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
        
        # Generate continuation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the continuation part (remove the original sentence)
        continuation = generated_text[len(sentence):].strip()
        
        # Store the continuation
        df.loc[i, 'continuation'] = continuation
    
    # Create a new DataFrame with the original sentence and its continuation
    result_df = df[[sentence_column, 'continuation']]
    
    # Add the full text (original + continuation) as a new column
    result_df['full_text'] = result_df[sentence_column] + ' ' + result_df['continuation']
    
    # Save the results
    print(f"Saving results to {output_file}")
    result_df.to_csv(output_file, index=False)
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate continuations for sentences using Aya or Llama models")
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--output", required=True, help="Path to save the output CSV file")
    parser.add_argument(
        "--model", 
        required=True,
        help="Name of the Hugging Face model to use (e.g., 'CohereForAI/aya-23-8B' or 'meta-llama/Meta-Llama-3-8B-Instruct')"
    )
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=50, 
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7, 
        help="Temperature for generation (higher = more creative)"
    )
    parser.add_argument(
        "--token_file",
        type=str,
        default=None,
        help="Path to a file containing your Hugging Face token (optional)"
    )
    
    args = parser.parse_args()
    
    # Handle token file if provided
    if args.token_file:
        try:
            with open(args.token_file, 'r') as f:
                token = f.read().strip()
                os.environ["HUGGING_FACE_HUB_TOKEN"] = token
                print(f"Loaded token from {args.token_file}")
        except Exception as e:
            print(f"Error loading token from file: {e}")
    
    generate_continuations(
        args.input, 
        args.output, 
        args.model, 
        args.max_tokens, 
        args.temperature
    )
