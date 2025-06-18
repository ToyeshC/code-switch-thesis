import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# --- Model and Column Configuration ---
MODELS_TO_PROCESS = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama31": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "aya": "CohereForAI/aya-23-8B"
}
COLUMNS_TO_PROCESS = ["src", "tgt", "generated"]

def setup_model_and_tokenizer(model_name, token):
    """Loads the model and tokenizer from Hugging Face."""
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
        torch_dtype=torch.bfloat16,
                    device_map="auto",
        token=token
    )
    
    # Set pad token and padding side for batched generation
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Left-padding is required for batched generation
    tokenizer.padding_side = "left"
            
    set_seed(42)
        return model, tokenizer
        
def generate_continuations_batch(model, tokenizer, texts, max_tokens, temperature):
    """Generates continuations for a batch of texts using the appropriate chat template."""
    
    # All models used are instruct-tuned, so we apply the chat template universally.
    # The tokenizer handles the model-specific formatting.
    messages_batch = [[{"role": "user", "content": text}] for text in texts]
    
    input_ids = tokenizer.apply_chat_template(
        messages_batch,
        add_generation_prompt=True,
            return_tensors="pt", 
            padding=True
        ).to(model.device)
        
    # Define a list of terminators to stop generation.
    # This handles model-specific end-of-sentence tokens.
    terminators = []
    if tokenizer.eos_token_id is not None:
        terminators.append(tokenizer.eos_token_id)
    if "Meta-Llama-3" in model.config.name_or_path:
        terminators.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))

            outputs = model.generate(
        input_ids,
                max_new_tokens=max_tokens,
        eos_token_id=terminators,
        do_sample=True,
                temperature=temperature,
                top_p=0.9,
        pad_token_id=tokenizer.pad_token_id
            )
        
    # Decode only the newly generated tokens, skipping the prompt
    responses = [tokenizer.decode(out[len(in_ids):], skip_special_tokens=True) for in_ids, out in zip(input_ids, outputs)]
    return responses

def main(args):
    """Main function to generate all continuations and save to a single file."""
    print(f"Reading input file: {args.input}")
    df = pd.read_csv(args.input)

    columns_to_process = args.columns.split(',')
    print(f"Target columns for continuation: {columns_to_process}")

    for model_short_name, model_full_name in MODELS_TO_PROCESS.items():
        model, tokenizer = setup_model_and_tokenizer(model_full_name, args.token)

        for column_name in columns_to_process:
            if column_name not in df.columns:
                print(f"Warning: Column '{column_name}' not found in input file. Skipping.")
            continue
        
            new_column_name = f"{model_short_name}_{column_name}_continuation"
            print(f"\nProcessing: Model '{model_short_name}' on column '{column_name}' -> new column '{new_column_name}'")

            all_continuations = []
            
            # Create batches
            texts_to_process = [str(text) if pd.notna(text) else "" for text in df[column_name]]
            
            with tqdm(total=len(texts_to_process), desc=f"Generating for {new_column_name}") as pbar:
                for i in range(0, len(texts_to_process), args.batch_size):
                    batch_texts = texts_to_process[i:i + args.batch_size]
                    batch_results = [""] * len(batch_texts) # Pre-fill with empty strings

                    # Get indices and values of non-empty texts that are not just whitespace
                    indices_and_texts = [(idx, text) for idx, text in enumerate(batch_texts) if text and text.strip()]
                    
                    if indices_and_texts:
                        # Unzip into separate lists of indices and the actual text prompts
                        original_indices, non_empty_texts = zip(*indices_and_texts)
                        
                        # Generate continuations just for the valid prompts
                        continuations = generate_continuations_batch(model, tokenizer, list(non_empty_texts), args.max_tokens, args.temperature)
                        
                        # Place the results back into the correct positions using the saved indices
                        for idx, continuation in zip(original_indices, continuations):
                            batch_results[idx] = continuation

                    all_continuations.extend(batch_results)
                    pbar.update(len(batch_texts))
            
            df[new_column_name] = all_continuations

        # Clear memory after processing a model
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    print(f"\nSaving all continuations to: {args.output}")
    df.to_csv(args.output, index=False)
    print("Script finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text continuations for specified columns and models using batching.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output", type=str, required=True, help="Path to the consolidated output CSV file.")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face API token for gated models.")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature for sampling.")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of prompts to process in parallel on the GPU.")
    parser.add_argument("--columns", type=str, default="src,tgt,generated", help="Comma-separated list of column names to process.")
    
    args = parser.parse_args()
    main(args) 