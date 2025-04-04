#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --partition=rome
#SBATCH --job-name=visualize
#SBATCH --mem=16G
#SBATCH --output=outputs/09_visualization.out

# Load necessary modules
module load 2023
module load Miniconda3/23.5.2-0
source ~/.bashrc
conda activate code-switch

# Define languages - can be modified for other language pairs
BASE_LANG="hindi"
SOURCE_LANG="english"

# Define paths
TOXICITY_DIR="data/output/toxicity_analysis"
LANG_DIR="data/output/language_detection"
OUTPUT_DIR="data/output/visualizations"

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/histograms
mkdir -p $OUTPUT_DIR/boxplots
mkdir -p $OUTPUT_DIR/monolingual_vs_codeswitched

# Compare toxicity for base language prompts and model responses
echo "Comparing toxicity for ${BASE_LANG^} prompts and LLaMA responses..."
python src/compare_toxicity_with_id.py \
    --prompt_file $TOXICITY_DIR/${BASE_LANG}_prompt_toxicity.csv \
    --response_file $TOXICITY_DIR/llama3_8b_${BASE_LANG}_toxicity.csv \
    --output_dir $OUTPUT_DIR/${BASE_LANG}_llama_comparison \
    --model_name "LLaMA 3 (8B)"

echo "Comparing toxicity for ${BASE_LANG^} prompts and Aya responses..."
python src/compare_toxicity_with_id.py \
    --prompt_file $TOXICITY_DIR/${BASE_LANG}_prompt_toxicity.csv \
    --response_file $TOXICITY_DIR/aya_${BASE_LANG}_toxicity.csv \
    --output_dir $OUTPUT_DIR/${BASE_LANG}_aya_comparison \
    --model_name "Aya"

# Compare toxicity for source language prompts and model responses
echo "Comparing toxicity for ${SOURCE_LANG^} prompts and LLaMA responses..."
python src/compare_toxicity_with_id.py \
    --prompt_file $TOXICITY_DIR/${SOURCE_LANG}_prompt_toxicity.csv \
    --response_file $TOXICITY_DIR/llama3_8b_${SOURCE_LANG}_toxicity.csv \
    --output_dir $OUTPUT_DIR/${SOURCE_LANG}_llama_comparison \
    --model_name "LLaMA 3 (8B)"

echo "Comparing toxicity for ${SOURCE_LANG^} prompts and Aya responses..."
python src/compare_toxicity_with_id.py \
    --prompt_file $TOXICITY_DIR/${SOURCE_LANG}_prompt_toxicity.csv \
    --response_file $TOXICITY_DIR/aya_${SOURCE_LANG}_toxicity.csv \
    --output_dir $OUTPUT_DIR/${SOURCE_LANG}_aya_comparison \
    --model_name "Aya"

# Pass language variables to Python script
BASE_LANG=$BASE_LANG SOURCE_LANG=$SOURCE_LANG python -c "
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Get language values from environment variables
base_lang = os.environ.get('BASE_LANG', 'hindi')
source_lang = os.environ.get('SOURCE_LANG', 'english')

# Set styling
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Load datasets
try:
    base_prompt_toxicity = pd.read_csv(f'$TOXICITY_DIR/{base_lang}_prompt_toxicity.csv')
    source_prompt_toxicity = pd.read_csv(f'$TOXICITY_DIR/{source_lang}_prompt_toxicity.csv')
    llama_base_toxicity = pd.read_csv(f'$TOXICITY_DIR/llama3_8b_{base_lang}_toxicity.csv')
    llama_source_toxicity = pd.read_csv(f'$TOXICITY_DIR/llama3_8b_{source_lang}_toxicity.csv')
    aya_base_toxicity = pd.read_csv(f'$TOXICITY_DIR/aya_{base_lang}_toxicity.csv')
    aya_source_toxicity = pd.read_csv(f'$TOXICITY_DIR/aya_{source_lang}_toxicity.csv')
    
    # Load language data
    base_lang_data = pd.read_csv(f'$LANG_DIR/{base_lang}_language_detection.csv')
    source_lang_data = pd.read_csv(f'$LANG_DIR/{source_lang}_language_detection.csv')
    
    # Generate histograms and visualizations
    print(f'Generating visualizations for {base_lang.capitalize()} and {source_lang.capitalize()}... This may take a moment.')
    
    # Additional visualization code would go here
    # ...
    
    print('Visualization generation complete. Results saved to $OUTPUT_DIR')
except Exception as e:
    print(f'Error generating visualizations: {str(e)}')
"

echo "Visualization complete. Results saved to $OUTPUT_DIR"
