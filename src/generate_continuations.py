import pandas as pd
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def generate_continuations(input_file, output_file, model_name, max_new_tokens=50, temperature=0.7):
    """
    Generate continuations for sentences in a CSV file using a Llama model.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the output CSV file
        model_name (str): Name of the Hugging Face model to use
        max_new_tokens (int): Maximum number of new tokens to generate
        temperature (float): Temperature for generation (higher = more creative)
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto"
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
    parser = argparse.ArgumentParser(description="Generate continuations for sentences using a Llama model")
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--output", required=True, help="Path to save the output CSV file")
    parser.add_argument(
        "--model", 
        default="meta-llama/Meta-Llama-3-8B-Instruct", 
        help="Name of the Hugging Face model to use"
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
    
    args = parser.parse_args()
    
    generate_continuations(
        args.input, 
        args.output, 
        args.model, 
        args.max_tokens, 
        args.temperature
    )
