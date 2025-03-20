import pandas as pd
import numpy as np
import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from huggingface_hub import login, HfFolder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
import json

# Add the directory containing config.py to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import API keys
try:
    # Print the actual path to help debug
    config_path = os.path.join(project_root, 'src', 'config.py')
    print(f"Looking for config.py at: {config_path}")
    
    # Direct import attempt
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_content = f.read()
            print(f"Config file content:\n{config_content}")
            
            # Extract API key using a simple parsing approach
            import re
            perspective_match = re.search(r'PERSPECTIVE_API_KEY\s*=\s*["\']([^"\']+)["\']', config_content)
            if perspective_match:
                PERSPECTIVE_API_KEY = perspective_match.group(1)
                print(f"Extracted PERSPECTIVE_API_KEY: {PERSPECTIVE_API_KEY[:5]}...{PERSPECTIVE_API_KEY[-5:]}")
            else:
                PERSPECTIVE_API_KEY = None
                print("Could not extract PERSPECTIVE_API_KEY from config.py")
                
            huggingface_match = re.search(r'HUGGING_FACE_HUB_TOKEN\s*=\s*["\']([^"\']+)["\']', config_content)
            if huggingface_match:
                HUGGING_FACE_HUB_TOKEN = huggingface_match.group(1)
                print(f"Extracted HUGGING_FACE_HUB_TOKEN: {HUGGING_FACE_HUB_TOKEN[:5]}...{HUGGING_FACE_HUB_TOKEN[-5:]}" if HUGGING_FACE_HUB_TOKEN else "No Hugging Face token found")
            else:
                HUGGING_FACE_HUB_TOKEN = None
    else:
        print(f"Config file not found at: {config_path}")
        PERSPECTIVE_API_KEY = None
        HUGGING_FACE_HUB_TOKEN = None
        
    # Try regular import as backup
    from config import PERSPECTIVE_API_KEY as CONFIG_PERSPECTIVE_API_KEY
    if PERSPECTIVE_API_KEY is None and CONFIG_PERSPECTIVE_API_KEY is not None:
        PERSPECTIVE_API_KEY = CONFIG_PERSPECTIVE_API_KEY
        print(f"Successfully imported PERSPECTIVE_API_KEY from config.py via import")
except Exception as e:
    print(f"Warning: Could not import API keys from config.py: {e}")
    PERSPECTIVE_API_KEY = None
    HUGGING_FACE_HUB_TOKEN = None

# Import existing analysis functions
try:
    from analyse_comments import analyze_comments
    print("Successfully imported analyze_comments function")
except Exception as e:
    print(f"Error importing analyze_comments: {e}")
    print("Will not be able to run Perspective API analysis")
    def analyze_comments(*args, **kwargs):
        print("analyze_comments function not available")
        return pd.DataFrame()

def generate_model_responses(df, model_name, output_file=None, max_new_tokens=100, temperature=0.7):
    """
    Generate responses for prompts using a specified model.
    
    Args:
        df (pd.DataFrame): DataFrame containing prompts in a 'sentence' column
        model_name (str): Name of the Hugging Face model to use
        output_file (str, optional): Path to save the output CSV file
        max_new_tokens (int): Maximum number of new tokens to generate
        temperature (float): Temperature for generation
        
    Returns:
        pd.DataFrame: DataFrame with original prompts and model responses
    """
    # Handle authentication for gated models (using non-interactive method)
    if HUGGING_FACE_HUB_TOKEN:
        print("Using Hugging Face token from config.py")
        os.environ["HUGGING_FACE_HUB_TOKEN"] = HUGGING_FACE_HUB_TOKEN
        # Set token directly instead of using login() which can prompt interactively
        HfFolder.save_token(HUGGING_FACE_HUB_TOKEN)
    elif "HUGGING_FACE_HUB_TOKEN" in os.environ:
        print("Using Hugging Face token from environment variable")
        hf_token = os.environ["HUGGING_FACE_HUB_TOKEN"]
        # Set token directly instead of using login() which can prompt interactively
        HfFolder.save_token(hf_token)
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
    
    # Ensure we have a 'sentence' column
    if 'sentence' not in df.columns:
        raise ValueError("DataFrame must contain a 'sentence' column")
    
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
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'prompt': df['sentence'],
        'response': responses,
        'model': model_name
    })
    
    # Save the results if output file is provided
    if output_file:
        print(f"Saving responses to {output_file}")
        result_df.to_csv(output_file, index=False)
    
    return result_df

def analyze_toxicity_comparison(prompts_df, llama_df, aya_df, output_dir):
    """
    Perform statistical analysis comparing toxicity scores between prompts and model responses.
    
    Args:
        prompts_df (pd.DataFrame): DataFrame with prompts and their toxicity scores
        llama_df (pd.DataFrame): DataFrame with LLaMA responses and their toxicity scores
        aya_df (pd.DataFrame): DataFrame with Aya responses and their toxicity scores
        output_dir (str): Directory to save analysis results and visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Print column names for debugging
    print("\nPrompts DataFrame columns:", prompts_df.columns.tolist())
    print("LLaMA DataFrame columns:", llama_df.columns.tolist())
    print("Aya DataFrame columns:", aya_df.columns.tolist())
    
    # Ensure we have a common key for merging (the original prompt)
    # Handle both 'comment' and 'prompt' column names
    if 'comment' in prompts_df.columns and 'prompt' not in prompts_df.columns:
        prompts_df = prompts_df.rename(columns={'comment': 'prompt'})
    
    # Check if responses have the right column names
    if 'comment' in llama_df.columns and 'prompt' not in llama_df.columns:
        llama_df = llama_df.rename(columns={'comment': 'prompt'})
    elif 'response' in llama_df.columns and 'prompt' not in llama_df.columns:
        # If we have response column but no prompt, try to find a text column for the prompt
        text_cols = [col for col in llama_df.columns if llama_df[col].dtype == 'object' and col != 'response']
        if text_cols:
            llama_df = llama_df.rename(columns={text_cols[0]: 'prompt'})
    
    if 'comment' in aya_df.columns and 'prompt' not in aya_df.columns:
        aya_df = aya_df.rename(columns={'comment': 'prompt'})
    elif 'response' in aya_df.columns and 'prompt' not in aya_df.columns:
        # If we have response column but no prompt, try to find a text column for the prompt
        text_cols = [col for col in aya_df.columns if aya_df[col].dtype == 'object' and col != 'response']
        if text_cols:
            aya_df = aya_df.rename(columns={text_cols[0]: 'prompt'})
    
    # Print column names after transformations
    print("\nAfter transformation:")
    print("Prompts DataFrame columns:", prompts_df.columns.tolist())
    print("LLaMA DataFrame columns:", llama_df.columns.tolist())
    print("Aya DataFrame columns:", aya_df.columns.tolist())
    
    # Check if 'prompt' exists in all dataframes before merging
    for df_name, df in [("prompts_df", prompts_df), ("llama_df", llama_df), ("aya_df", aya_df)]:
        if 'prompt' not in df.columns:
            raise ValueError(f"'prompt' column not found in {df_name}. Available columns: {df.columns.tolist()}")
    
    # Merge LLaMA data with prompts
    print("\nMerging LLaMA data with prompts...")
    llama_merged = pd.merge(prompts_df, llama_df, on='prompt', suffixes=('_prompt', '_llama'))
    
    # Merge Aya data with prompts
    print("Merging Aya data with prompts...")
    aya_merged = pd.merge(prompts_df, aya_df, on='prompt', suffixes=('_prompt', '_aya'))
    
    # Get common columns for the final merge
    prompt_cols = ['prompt'] + [col for col in prompts_df.columns if col != 'prompt']
    
    # Merge all three datasets
    print("Merging all datasets...")
    all_merged = pd.merge(llama_merged, aya_merged, on=prompt_cols, suffixes=('_llama', '_aya'))
    
    # Save the merged datasets
    llama_merged.to_csv(os.path.join(output_dir, 'llama_prompt_comparison.csv'), index=False)
    aya_merged.to_csv(os.path.join(output_dir, 'aya_prompt_comparison.csv'), index=False)
    all_merged.to_csv(os.path.join(output_dir, 'all_models_comparison.csv'), index=False)
    
    # Statistical analysis results
    analysis_results = {
        'prompt_vs_llama': {},
        'prompt_vs_aya': {},
        'llama_vs_aya': {},
        'correlations': {}
    }
    
    # 1. Comparing Prompt and Output Toxicity
    print("\n===== Comparing Prompt vs. LLaMA Toxicity =====")
    toxicity_metrics = ['toxicity', 'severe_toxicity', 'identity_attack', 'insult', 'profanity', 'threat']
    
    for metric in toxicity_metrics:
        # Prompt vs. LLaMA comparison
        prompt_data = llama_merged[f'{metric}_prompt']
        llama_data = llama_merged[f'{metric}_llama']
        
        # T-test
        t_stat, p_value = stats.ttest_rel(prompt_data, llama_data)
        # Wilcoxon test
        w_stat, w_p_value = stats.wilcoxon(prompt_data, llama_data)
        # Effect size (Cohen's d)
        effect_size = (prompt_data.mean() - llama_data.mean()) / np.sqrt((prompt_data.std()**2 + llama_data.std()**2) / 2)
        
        print(f"\n{metric.capitalize()}:")
        print(f"  Prompt mean: {prompt_data.mean():.4f}, LLaMA mean: {llama_data.mean():.4f}")
        print(f"  Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
        print(f"  Wilcoxon test: W={w_stat:.4f}, p={w_p_value:.4f}")
        print(f"  Effect size (Cohen's d): {effect_size:.4f}")
        
        analysis_results['prompt_vs_llama'][metric] = {
            'prompt_mean': prompt_data.mean(),
            'llama_mean': llama_data.mean(),
            't_test': {'t_stat': t_stat, 'p_value': p_value},
            'wilcoxon': {'w_stat': w_stat, 'p_value': w_p_value},
            'effect_size': effect_size
        }
    
    print("\n===== Comparing Prompt vs. Aya Toxicity =====")
    for metric in toxicity_metrics:
        # Prompt vs. Aya comparison
        prompt_data = aya_merged[f'{metric}_prompt']
        aya_data = aya_merged[f'{metric}_aya']
        
        # T-test
        t_stat, p_value = stats.ttest_rel(prompt_data, aya_data)
        # Wilcoxon test
        w_stat, w_p_value = stats.wilcoxon(prompt_data, aya_data)
        # Effect size (Cohen's d)
        effect_size = (prompt_data.mean() - aya_data.mean()) / np.sqrt((prompt_data.std()**2 + aya_data.std()**2) / 2)
        
        print(f"\n{metric.capitalize()}:")
        print(f"  Prompt mean: {prompt_data.mean():.4f}, Aya mean: {aya_data.mean():.4f}")
        print(f"  Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
        print(f"  Wilcoxon test: W={w_stat:.4f}, p={w_p_value:.4f}")
        print(f"  Effect size (Cohen's d): {effect_size:.4f}")
        
        analysis_results['prompt_vs_aya'][metric] = {
            'prompt_mean': prompt_data.mean(),
            'aya_mean': aya_data.mean(),
            't_test': {'t_stat': t_stat, 'p_value': p_value},
            'wilcoxon': {'w_stat': w_stat, 'p_value': w_p_value},
            'effect_size': effect_size
        }
    
    # 2. Comparing LLaMA vs. Aya Responses
    print("\n===== Comparing LLaMA vs. Aya Toxicity =====")
    for metric in toxicity_metrics:
        # LLaMA vs. Aya comparison
        llama_data = all_merged[f'{metric}_llama']
        aya_data = all_merged[f'{metric}_aya']
        
        # T-test
        t_stat, p_value = stats.ttest_rel(llama_data, aya_data)
        # Wilcoxon test
        w_stat, w_p_value = stats.wilcoxon(llama_data, aya_data)
        # Effect size (Cohen's d)
        effect_size = (llama_data.mean() - aya_data.mean()) / np.sqrt((llama_data.std()**2 + aya_data.std()**2) / 2)
        
        print(f"\n{metric.capitalize()}:")
        print(f"  LLaMA mean: {llama_data.mean():.4f}, Aya mean: {aya_data.mean():.4f}")
        print(f"  Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
        print(f"  Wilcoxon test: W={w_stat:.4f}, p={w_p_value:.4f}")
        print(f"  Effect size (Cohen's d): {effect_size:.4f}")
        
        analysis_results['llama_vs_aya'][metric] = {
            'llama_mean': llama_data.mean(),
            'aya_mean': aya_data.mean(),
            't_test': {'t_stat': t_stat, 'p_value': p_value},
            'wilcoxon': {'w_stat': w_stat, 'p_value': w_p_value},
            'effect_size': effect_size
        }
    
    # 3. Correlation Analysis
    print("\n===== Correlation Analysis =====")
    
    # Correlations within prompt toxicity metrics
    prompt_cols = [f'{m}_prompt' for m in toxicity_metrics]
    prompt_corr = all_merged[prompt_cols].corr()
    
    # Correlations within LLaMA toxicity metrics
    llama_cols = [f'{m}_llama' for m in toxicity_metrics]
    llama_corr = all_merged[llama_cols].corr()
    
    # Correlations within Aya toxicity metrics
    aya_cols = [f'{m}_aya' for m in toxicity_metrics]
    aya_corr = all_merged[aya_cols].corr()
    
    # Correlations between prompt and LLaMA toxicity
    prompt_llama_corr = pd.DataFrame(index=toxicity_metrics, columns=toxicity_metrics)
    for p_metric in toxicity_metrics:
        for l_metric in toxicity_metrics:
            corr = all_merged[f'{p_metric}_prompt'].corr(all_merged[f'{l_metric}_llama'])
            prompt_llama_corr.loc[p_metric, l_metric] = corr
    
    # Correlations between prompt and Aya toxicity
    prompt_aya_corr = pd.DataFrame(index=toxicity_metrics, columns=toxicity_metrics)
    for p_metric in toxicity_metrics:
        for a_metric in toxicity_metrics:
            corr = all_merged[f'{p_metric}_prompt'].corr(all_merged[f'{a_metric}_aya'])
            prompt_aya_corr.loc[p_metric, a_metric] = corr
    
    # Store correlation results
    analysis_results['correlations'] = {
        'prompt_internal': prompt_corr.to_dict(),
        'llama_internal': llama_corr.to_dict(),
        'aya_internal': aya_corr.to_dict(),
        'prompt_vs_llama': prompt_llama_corr.to_dict(),
        'prompt_vs_aya': prompt_aya_corr.to_dict()
    }
    
    # Save analysis results
    with open(os.path.join(output_dir, 'statistical_analysis.json'), 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Create visualizations
    # 1. Box plots comparing toxicity distributions
    for metric in toxicity_metrics:
        plt.figure(figsize=(12, 8))
        data = [
            all_merged[f'{metric}_prompt'],
            all_merged[f'{metric}_llama'],
            all_merged[f'{metric}_aya']
        ]
        plt.boxplot(data, labels=['Code-Switched Prompt', 'LLaMA Response', 'Aya Response'])
        plt.title(f'Comparison of {metric.capitalize()} Scores')
        plt.ylabel('Score')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, f'{metric}_comparison_boxplot.png'))
        plt.close()
    
    # 2. Correlation heatmaps
    plt.figure(figsize=(10, 8))
    sns.heatmap(prompt_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Prompt Toxicity Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prompt_correlation_heatmap.png'))
    plt.close()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(prompt_llama_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Prompt and LLaMA Toxicity Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prompt_llama_correlation_heatmap.png'))
    plt.close()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(prompt_aya_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Prompt and Aya Toxicity Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prompt_aya_correlation_heatmap.png'))
    plt.close()
    
    return analysis_results

def main():
    parser = argparse.ArgumentParser(description="Analyze model responses and toxicity for code-switched prompts")
    parser.add_argument("--input", required=True, help="Path to the input CSV file with code-switched prompts")
    parser.add_argument("--output_dir", required=True, help="Directory to save all output files")
    parser.add_argument("--llama_model", default="meta-llama/Meta-Llama-3-8B-Instruct", 
                        help="Name of the LLaMA model to use")
    parser.add_argument("--aya_model", default="CohereForAI/aya-23-8B", 
                        help="Name of the Aya model to use")
    parser.add_argument("--max_tokens", type=int, default=100, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Temperature for generation")
    parser.add_argument("--api_key", help="Perspective API key (overrides config.py)")
    parser.add_argument("--huggingface_token", help="Hugging Face token (overrides config.py)")
    parser.add_argument("--skip_generation", action="store_true", help="Skip generation step and use existing response files")
    parser.add_argument("--skip_perspective", action="store_true", help="Skip Perspective API analysis and use existing analysis files")
    parser.add_argument("--skip_statistical_analysis", action="store_true", help="Skip the statistical analysis part and exit after toxicity analysis")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Override API keys if provided
    if args.api_key:
        global PERSPECTIVE_API_KEY
        PERSPECTIVE_API_KEY = args.api_key
        print(f"Using Perspective API key from command line argument: {PERSPECTIVE_API_KEY[:5]}...{PERSPECTIVE_API_KEY[-5:]}")
    
    if args.huggingface_token:
        global HUGGING_FACE_HUB_TOKEN
        HUGGING_FACE_HUB_TOKEN = args.huggingface_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = HUGGING_FACE_HUB_TOKEN
    
    # Define file paths
    prompt_file = args.input
    llama_responses_file = os.path.join(args.output_dir, 'llama_responses.csv')
    aya_responses_file = os.path.join(args.output_dir, 'aya_responses.csv')
    prompt_toxicity_file = os.path.join(args.output_dir, 'prompt_toxicity.csv')
    llama_toxicity_file = os.path.join(args.output_dir, 'llama_toxicity.csv')
    aya_toxicity_file = os.path.join(args.output_dir, 'aya_toxicity.csv')
    
    # 1. Read the code-switched prompts
    print(f"Reading code-switched prompts from {prompt_file}")
    prompts_df = pd.read_csv(prompt_file)
    
    # 2. Generate responses from LLaMA and Aya models
    if not args.skip_generation:
        # Generate LLaMA responses
        print(f"\n===== Generating LLaMA Responses =====")
        llama_df = generate_model_responses(
            prompts_df,
            args.llama_model,
            llama_responses_file,
            args.max_tokens,
            args.temperature
        )
        
        # Generate Aya responses
        print(f"\n===== Generating Aya Responses =====")
        aya_df = generate_model_responses(
            prompts_df,
            args.aya_model,
            aya_responses_file,
            args.max_tokens,
            args.temperature
        )
    else:
        print("Skipping generation step, using existing response files")
        if not os.path.exists(llama_responses_file):
            raise FileNotFoundError(f"LLaMA responses file not found: {llama_responses_file}")
        if not os.path.exists(aya_responses_file):
            raise FileNotFoundError(f"Aya responses file not found: {aya_responses_file}")
            
        llama_df = pd.read_csv(llama_responses_file)
        aya_df = pd.read_csv(aya_responses_file)
        
        # Debug info
        print(f"Loaded LLaMA responses from {llama_responses_file}")
        print(f"LLaMA DataFrame shape: {llama_df.shape}")
        print(f"LLaMA columns: {llama_df.columns.tolist()}")
        
        print(f"Loaded Aya responses from {aya_responses_file}")
        print(f"Aya DataFrame shape: {aya_df.shape}")
        print(f"Aya columns: {aya_df.columns.tolist()}")
    
    # 3. Run Perspective API on prompts and model outputs
    if not args.skip_perspective:
        print(f"\n===== Analyzing Prompt Toxicity =====")
        # Make sure we're using the correct API key from config.py
        if PERSPECTIVE_API_KEY is None:
            print("WARNING: Perspective API key not found. Cannot run toxicity analysis.")
            print("Using --skip_perspective to continue with existing files (if available)")
            args.skip_perspective = True
        else:
            print(f"Using Perspective API key: {PERSPECTIVE_API_KEY[:5]}...{PERSPECTIVE_API_KEY[-5:]}")
            
            # Create temporary prompt file with the 'comment' column
            temp_prompt_file = os.path.join(args.output_dir, 'temp_prompt.csv')
            pd.DataFrame({'comment': prompts_df['sentence']}).to_csv(temp_prompt_file, index=False)
            
            prompts_with_toxicity = analyze_comments(temp_prompt_file, prompt_toxicity_file, PERSPECTIVE_API_KEY)
            
            print(f"\n===== Analyzing LLaMA Response Toxicity =====")
            # Create a temporary file with just the responses for analysis
            temp_llama_file = os.path.join(args.output_dir, 'temp_llama_responses.csv')
            if 'response' in llama_df.columns:
                pd.DataFrame({'comment': llama_df['response']}).to_csv(temp_llama_file, index=False)
                llama_with_toxicity = analyze_comments(temp_llama_file, llama_toxicity_file, PERSPECTIVE_API_KEY)
            else:
                llama_with_toxicity = analyze_comments(llama_responses_file, llama_toxicity_file, PERSPECTIVE_API_KEY)
            
            print(f"\n===== Analyzing Aya Response Toxicity =====")
            # Create a temporary file with just the responses for analysis
            temp_aya_file = os.path.join(args.output_dir, 'temp_aya_responses.csv')
            if 'response' in aya_df.columns:
                pd.DataFrame({'comment': aya_df['response']}).to_csv(temp_aya_file, index=False)
                aya_with_toxicity = analyze_comments(temp_aya_file, aya_toxicity_file, PERSPECTIVE_API_KEY)
            else:
                aya_with_toxicity = analyze_comments(aya_responses_file, aya_toxicity_file, PERSPECTIVE_API_KEY)
                
            # Clean up temporary files
            for temp_file in [temp_prompt_file, temp_llama_file, temp_aya_file]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
    if args.skip_perspective:
        print("Skipping Perspective API analysis, using existing analysis files")
        try:
            # Try to load existing toxicity files
            if not os.path.exists(prompt_toxicity_file):
                raise FileNotFoundError(f"Prompt toxicity file not found: {prompt_toxicity_file}")
            if not os.path.exists(llama_toxicity_file):
                raise FileNotFoundError(f"LLaMA toxicity file not found: {llama_toxicity_file}")
            if not os.path.exists(aya_toxicity_file):
                raise FileNotFoundError(f"Aya toxicity file not found: {aya_toxicity_file}")
                
            prompts_with_toxicity = pd.read_csv(prompt_toxicity_file)
            llama_with_toxicity = pd.read_csv(llama_toxicity_file)
            aya_with_toxicity = pd.read_csv(aya_toxicity_file)
            
            # Debug info
            print(f"Loaded toxicity data from {prompt_toxicity_file}, {llama_toxicity_file}, and {aya_toxicity_file}")
            print(f"Columns in prompt toxicity: {prompts_with_toxicity.columns.tolist()}")
            print(f"Columns in LLaMA toxicity: {llama_with_toxicity.columns.tolist()}")
            print(f"Columns in Aya toxicity: {aya_with_toxicity.columns.tolist()}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Some required toxicity files are missing. Please run without --skip_perspective first.")
            return
    
    # Skip statistical analysis if requested
    if args.skip_statistical_analysis:
        print("Skipping statistical analysis as requested")
        return
        
    # 4. Perform statistical analysis
    print(f"\n===== Performing Statistical Analysis =====")
    analysis_dir = os.path.join(args.output_dir, 'analysis')
    analysis_results = analyze_toxicity_comparison(
        prompts_with_toxicity,
        llama_with_toxicity,
        aya_with_toxicity,
        analysis_dir
    )
    
    print(f"\nAnalysis complete! Results saved to {analysis_dir}")

if __name__ == "__main__":
    main() 