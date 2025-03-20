import pandas as pd
import argparse
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfFolder

def generate_model_responses(input_file, model_name, output_file, max_new_tokens=100, temperature=0.7, huggingface_token=None):
    """
    Generate responses for prompts using a specified model while preserving primary keys.
    
    Args:
        input_file (str): Path to the CSV file containing prompts with primary keys
        model_name (str): Name of the Hugging Face model to use
        output_file (str): Path to save the output CSV file
        max_new_tokens (int): Maximum number of new tokens to generate
        temperature (float): Temperature for generation
        huggingface_token (str): Optional Hugging Face token for accessing gated models
    """
    # Handle authentication for gated models
    if huggingface_token:
        print("Using provided Hugging Face token")
        os.environ["HUGGING_FACE_HUB_TOKEN"] = huggingface_token
        HfFolder.save_token(huggingface_token)
    elif "HUGGING_FACE_HUB_TOKEN" in os.environ:
        print("Using Hugging Face token from environment variable")
        huggingface_token = os.environ["HUGGING_FACE_HUB_TOKEN"]
        HfFolder.save_token(huggingface_token)
    else:
        print("Warning: No Hugging Face token found. Gated models may not be accessible.")
    
    # Load tokenizer and model
    print(f"Loading {model_name} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading {model_name} model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Read input file
    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Ensure the required columns exist
    if 'prompt_id' not in df.columns:
        print("Error: No 'prompt_id' column found in the input file.")
        return None
    
    if 'sentence' not in df.columns:
        print("Error: No 'sentence' column found in the input file.")
        return None
    
    # Create a new column for the model responses
    responses = []
    
    # Generate responses for each prompt
    print(f"Generating responses using {model_name}...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row['sentence']
        
        # Skip empty prompts
        if pd.isna(prompt) or prompt.strip() == '':
            responses.append('')
            continue
        
        # Handle model-specific prompting format
        if "llama" in model_name.lower():
            # Llama-specific prompt format
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        elif "aya" in model_name.lower():
            # Aya-specific prompt format
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # Generic format
            formatted_prompt = prompt
        
        # Tokenize the prompt
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Generate response
        try:
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
            
            # Extract only the response (remove the prompt)
            if "llama" in model_name.lower():
                response = generated_text.split("[/INST]")[-1].strip()
            elif "aya" in model_name.lower():
                response = generated_text.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
            else:
                response = generated_text[len(prompt):].strip()
            
            responses.append(response)
        except Exception as e:
            print(f"Error generating response for prompt {i}: {e}")
            responses.append('')
    
    # Create result DataFrame with primary keys
    result_df = pd.DataFrame({
        'prompt_id': df['prompt_id'],
        'prompt': df['sentence'],
        'response': responses,
        'model': model_name
    })
    
    # Save the results
    print(f"Saving responses to {output_file}")
    result_df.to_csv(output_file, index=False)
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description="Generate model responses while preserving primary keys")
    parser.add_argument("--input", required=True, help="Path to the CSV file containing prompts with primary keys")
    parser.add_argument("--model", required=True, help="Name of the Hugging Face model to use")
    parser.add_argument("--output", required=True, help="Path to save the output CSV file")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--huggingface_token", help="Hugging Face token for accessing gated models")
    
    args = parser.parse_args()
    
    generate_model_responses(
        args.input,
        args.model,
        args.output,
        args.max_tokens,
        args.temperature,
        args.huggingface_token
    )

if __name__ == "__main__":
    main() 