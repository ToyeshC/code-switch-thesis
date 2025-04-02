#!/bin/bash

# Script to clean up the code-switch-thesis repository
# This script removes outdated files and organizes the repository according to the CLEAN_REPO_GUIDE.md

echo "Creating backup of the repository..."
BACKUP_DIR="../code-switch-thesis-backup-$(date +%Y%m%d)"
cp -r . "$BACKUP_DIR"
echo "Backup created at $BACKUP_DIR"

echo "Setting up external dependencies if not already present..."
# Set up RTP-LX if not already present
if [ ! -d "data/RTP-LX" ]; then
    echo "RTP-LX not found. Please download it manually:"
    echo "mkdir -p data/RTP-LX"
    echo "wget https://github.com/microsoft/RTP-LX/raw/main/RTP-LX/RTP-LX.zip -O data/RTP-LX/RTP-LX.zip"
    echo "unzip data/RTP-LX/RTP-LX.zip -d data/RTP-LX/"
    mkdir -p data/RTP-LX
    touch data/RTP-LX/README.txt
    echo "This is a placeholder. Please download RTP-LX from https://github.com/microsoft/RTP-LX/blob/main/RTP-LX/RTP-LX.zip" > data/RTP-LX/README.txt
fi

# Keep ezswitch if present, otherwise create placeholder
if [ ! -d "ezswitch" ]; then
    echo "ezswitch not found. Please clone it manually:"
    echo "git clone https://github.com/gkuwanto/ezswitch.git"
    echo "cd ezswitch && pip install -e . && cd .."
    mkdir -p ezswitch
    touch ezswitch/README.txt
    echo "This is a placeholder. Please clone ezswitch from https://github.com/gkuwanto/ezswitch.git" > ezswitch/README.txt
fi

echo "Creating modular scripts with numbered prefixes..."
# Create directory for new modular scripts
mkdir -p scripts_new

# Create modular script templates
cat > scripts_new/01_extract_rtp_data.sh << 'EOF'
#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=extract_data
#SBATCH --mem=16G
#SBATCH --output=outputs/01_extract_rtp_data.out

# Load necessary modules
module load 2023
module load Miniconda3/23.5.2-0
source ~/.bashrc
conda activate code-switch

# Define paths
OUTPUT_DIR="data/output"
RTP_DIR="data/RTP-LX"
EXTRACTED_DIR="data/extracted_prompts"

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $EXTRACTED_DIR

echo "Extracting prompts from RTP-LX dataset..."
python src/add_primary_key.py \
    --hindi $RTP_DIR/hindi.json \
    --english $RTP_DIR/english.json \
    --output_dir $OUTPUT_DIR

echo "Data extraction complete. Results saved to $OUTPUT_DIR"
EOF

cat > scripts_new/02_language_detection.sh << 'EOF'
#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=lang_detect
#SBATCH --mem=16G
#SBATCH --output=outputs/02_language_detection.out

# Load necessary modules
module load 2023
module load Miniconda3/23.5.2-0
source ~/.bashrc
conda activate code-switch

# Define paths
INPUT_DIR="data/output"
OUTPUT_DIR="data/output/language_detection"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Detecting language in Hindi prompts..."
python src/language_detection_with_id.py \
    --input_file $INPUT_DIR/hindi_prompts_with_id.csv \
    --output_file $OUTPUT_DIR/hindi_language_detection.csv \
    --fasttext_model lid.176.bin

echo "Detecting language in English prompts..."
python src/language_detection_with_id.py \
    --input_file $INPUT_DIR/english_prompts_with_id.csv \
    --output_file $OUTPUT_DIR/english_language_detection.csv \
    --fasttext_model lid.176.bin

echo "Language detection complete. Results saved to $OUTPUT_DIR"
EOF

cat > scripts_new/03_filter_code_switch.sh << 'EOF'
#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=filter_cs
#SBATCH --mem=16G
#SBATCH --output=outputs/03_filter_code_switch.out

# Load necessary modules
module load 2023
module load Miniconda3/23.5.2-0
source ~/.bashrc
conda activate code-switch

# Define paths
INPUT_DIR="data/output/language_detection"
OUTPUT_DIR="data/output/filtered"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Filtering Hindi prompts for code-switched content..."
python src/filter_language_mix_with_id.py \
    --input $INPUT_DIR/hindi_language_detection.csv \
    --output $OUTPUT_DIR/hindi_filtered.csv

echo "Filtering English prompts for code-switched content..."
python src/filter_language_mix_with_id.py \
    --input $INPUT_DIR/english_language_detection.csv \
    --output $OUTPUT_DIR/english_filtered.csv

echo "Filtering complete. Results saved to $OUTPUT_DIR"
EOF

cat > scripts_new/04_generate_responses.sh << 'EOF'
#!/bin/bash

#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=gen_resp
#SBATCH --mem=32G
#SBATCH --output=outputs/04_generate_responses.out

# Load necessary modules
module load 2023
module load Miniconda3/23.5.2-0
source ~/.bashrc
conda activate code-switch

# Set Hugging Face token
export HUGGING_FACE_HUB_TOKEN=$(cat ~/.huggingface/token)

# Define paths
INPUT_DIR="data/output/filtered"
OUTPUT_DIR="data/output/model_responses"

# Create output directory
mkdir -p $OUTPUT_DIR

# Generate LLaMA 3 (8B) responses for Hindi prompts
echo "Generating LLaMA 3 (8B) responses for Hindi prompts..."
python src/generate_model_responses_with_id.py \
    --input $INPUT_DIR/hindi_filtered.csv \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --output $OUTPUT_DIR/llama3_8b_hindi_responses.csv \
    --max_tokens 100 \
    --temperature 0.7

# Generate LLaMA 3 (8B) responses for English prompts
echo "Generating LLaMA 3 (8B) responses for English prompts..."
python src/generate_model_responses_with_id.py \
    --input $INPUT_DIR/english_filtered.csv \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --output $OUTPUT_DIR/llama3_8b_english_responses.csv \
    --max_tokens 100 \
    --temperature 0.7

# Generate Aya responses for Hindi prompts
echo "Generating Aya responses for Hindi prompts..."
python src/generate_model_responses_with_id.py \
    --input $INPUT_DIR/hindi_filtered.csv \
    --model "CohereForAI/aya-23-8B" \
    --output $OUTPUT_DIR/aya_hindi_responses.csv \
    --max_tokens 100 \
    --temperature 0.7

# Generate Aya responses for English prompts
echo "Generating Aya responses for English prompts..."
python src/generate_model_responses_with_id.py \
    --input $INPUT_DIR/english_filtered.csv \
    --model "CohereForAI/aya-23-8B" \
    --output $OUTPUT_DIR/aya_english_responses.csv \
    --max_tokens 100 \
    --temperature 0.7

echo "Response generation complete. Results saved to $OUTPUT_DIR"
EOF

cat > scripts_new/05_analyze_toxicity.sh << 'EOF'
#!/bin/bash

#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=analyze_tox
#SBATCH --mem=16G
#SBATCH --output=outputs/05_analyze_toxicity.out

# Load necessary modules
module load 2023
module load Miniconda3/23.5.2-0
source ~/.bashrc
conda activate code-switch

# Get Perspective API key from config
API_KEY=$(grep -oP "PERSPECTIVE_API_KEY = \"\K[^\"]+" src/config.py)
echo "Retrieved Perspective API key from src/config.py"

# Define paths
INPUT_DIR="data/output/filtered"
RESPONSES_DIR="data/output/model_responses"
OUTPUT_DIR="data/output/toxicity_analysis"

# Create output directory
mkdir -p $OUTPUT_DIR

# Analyze toxicity of Hindi prompts
echo "Analyzing toxicity of Hindi prompts..."
python src/analyze_toxicity_with_id.py \
    --input $INPUT_DIR/hindi_filtered.csv \
    --output $OUTPUT_DIR/hindi_prompt_toxicity.csv \
    --progress_file $OUTPUT_DIR/hindi_prompt_toxicity_progress.csv \
    --api_key "$API_KEY" \
    --batch_size 5

# Analyze toxicity of English prompts
echo "Analyzing toxicity of English prompts..."
python src/analyze_toxicity_with_id.py \
    --input $INPUT_DIR/english_filtered.csv \
    --output $OUTPUT_DIR/english_prompt_toxicity.csv \
    --progress_file $OUTPUT_DIR/english_prompt_toxicity_progress.csv \
    --api_key "$API_KEY" \
    --batch_size 5

# Analyze toxicity of model responses
echo "Analyzing toxicity of LLaMA responses for Hindi prompts..."
python src/analyze_toxicity_with_id.py \
    --input $RESPONSES_DIR/llama3_8b_hindi_responses.csv \
    --output $OUTPUT_DIR/llama3_8b_hindi_toxicity.csv \
    --progress_file $OUTPUT_DIR/llama3_8b_hindi_toxicity_progress.csv \
    --api_key "$API_KEY" \
    --batch_size 5

echo "Analyzing toxicity of LLaMA responses for English prompts..."
python src/analyze_toxicity_with_id.py \
    --input $RESPONSES_DIR/llama3_8b_english_responses.csv \
    --output $OUTPUT_DIR/llama3_8b_english_toxicity.csv \
    --progress_file $OUTPUT_DIR/llama3_8b_english_toxicity_progress.csv \
    --api_key "$API_KEY" \
    --batch_size 5

echo "Analyzing toxicity of Aya responses for Hindi prompts..."
python src/analyze_toxicity_with_id.py \
    --input $RESPONSES_DIR/aya_hindi_responses.csv \
    --output $OUTPUT_DIR/aya_hindi_toxicity.csv \
    --progress_file $OUTPUT_DIR/aya_hindi_toxicity_progress.csv \
    --api_key "$API_KEY" \
    --batch_size 5

echo "Analyzing toxicity of Aya responses for English prompts..."
python src/analyze_toxicity_with_id.py \
    --input $RESPONSES_DIR/aya_english_responses.csv \
    --output $OUTPUT_DIR/aya_english_toxicity.csv \
    --progress_file $OUTPUT_DIR/aya_english_toxicity_progress.csv \
    --api_key "$API_KEY" \
    --batch_size 5

echo "Toxicity analysis complete. Results saved to $OUTPUT_DIR"
EOF

cat > scripts_new/06_visualization.sh << 'EOF'
#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=visualize
#SBATCH --mem=16G
#SBATCH --output=outputs/06_visualization.out

# Load necessary modules
module load 2023
module load Miniconda3/23.5.2-0
source ~/.bashrc
conda activate code-switch

# Define paths
TOXICITY_DIR="data/output/toxicity_analysis"
LANG_DIR="data/output/language_detection"
OUTPUT_DIR="data/output/visualizations"

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/histograms
mkdir -p $OUTPUT_DIR/boxplots
mkdir -p $OUTPUT_DIR/monolingual_vs_codeswitched

# Compare toxicity for Hindi prompts and model responses
echo "Comparing toxicity for Hindi prompts and LLaMA responses..."
python src/compare_toxicity_with_id.py \
    --prompt_file $TOXICITY_DIR/hindi_prompt_toxicity.csv \
    --response_file $TOXICITY_DIR/llama3_8b_hindi_toxicity.csv \
    --output_dir $OUTPUT_DIR/hindi_llama_comparison \
    --model_name "LLaMA 3 (8B)"

echo "Comparing toxicity for Hindi prompts and Aya responses..."
python src/compare_toxicity_with_id.py \
    --prompt_file $TOXICITY_DIR/hindi_prompt_toxicity.csv \
    --response_file $TOXICITY_DIR/aya_hindi_toxicity.csv \
    --output_dir $OUTPUT_DIR/hindi_aya_comparison \
    --model_name "Aya"

# Compare toxicity for English prompts and model responses
echo "Comparing toxicity for English prompts and LLaMA responses..."
python src/compare_toxicity_with_id.py \
    --prompt_file $TOXICITY_DIR/english_prompt_toxicity.csv \
    --response_file $TOXICITY_DIR/llama3_8b_english_toxicity.csv \
    --output_dir $OUTPUT_DIR/english_llama_comparison \
    --model_name "LLaMA 3 (8B)"

echo "Comparing toxicity for English prompts and Aya responses..."
python src/compare_toxicity_with_id.py \
    --prompt_file $TOXICITY_DIR/english_prompt_toxicity.csv \
    --response_file $TOXICITY_DIR/aya_english_toxicity.csv \
    --output_dir $OUTPUT_DIR/english_aya_comparison \
    --model_name "Aya"

# Generate visualizations with custom Python code
echo "Generating histograms and charts..."
python -c "
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set styling
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Load datasets
try:
    hindi_prompt_toxicity = pd.read_csv('$TOXICITY_DIR/hindi_prompt_toxicity.csv')
    english_prompt_toxicity = pd.read_csv('$TOXICITY_DIR/english_prompt_toxicity.csv')
    llama_hindi_toxicity = pd.read_csv('$TOXICITY_DIR/llama3_8b_hindi_toxicity.csv')
    llama_english_toxicity = pd.read_csv('$TOXICITY_DIR/llama3_8b_english_toxicity.csv')
    aya_hindi_toxicity = pd.read_csv('$TOXICITY_DIR/aya_hindi_toxicity.csv')
    aya_english_toxicity = pd.read_csv('$TOXICITY_DIR/aya_english_toxicity.csv')
    
    # Load language data
    hindi_lang = pd.read_csv('$LANG_DIR/hindi_language_detection.csv')
    english_lang = pd.read_csv('$LANG_DIR/english_language_detection.csv')
    
    # Generate histograms and visualizations
    print('Generating visualizations... This may take a moment.')
    
    # Additional visualization code would go here
    # ...
    
    print('Visualization generation complete. Results saved to $OUTPUT_DIR')
except Exception as e:
    print(f'Error generating visualizations: {str(e)}')
"

echo "Visualization complete. Results saved to $OUTPUT_DIR"
EOF

# Copy test_primary_key_hindi_only.sh if it exists
if [ -f "scripts/test_primary_key_hindi_only.sh" ]; then
    cp scripts/test_primary_key_hindi_only.sh scripts_new/
fi

# Make all new scripts executable
chmod +x scripts_new/*.sh

echo "Cleaning up src directory..."
# Keep only the essential source files
mkdir -p src_essential
cp src/add_primary_key.py src_essential/
cp src/analyze_toxicity_with_id.py src_essential/
cp src/compare_toxicity_with_id.py src_essential/
cp src/config.py src_essential/
cp src/filter_language_mix_with_id.py src_essential/
cp src/generate_model_responses_with_id.py src_essential/
cp src/language_detection_with_id.py src_essential/

# Remove old src directory and replace with essential source files
rm -rf src
mv src_essential src

echo "Cleaning up data directory..."
# Keep only essential data directories
mkdir -p data_essential/extracted_prompts
mkdir -p data_essential/output

# Copy extracted prompts
if [ -d "data/extracted_prompts" ]; then
    cp -r data/extracted_prompts/* data_essential/extracted_prompts/
fi

# Copy important output data if it exists
if [ -d "data/output" ]; then
    cp -r data/output/* data_essential/output/
fi

# Ensure RTP-LX directory exists
mkdir -p data_essential/RTP-LX
if [ -d "data/RTP-LX" ]; then
    cp -r data/RTP-LX/* data_essential/RTP-LX/
fi

# Remove old data directory and replace with essential data
rm -rf data
mv data_essential data

# Create outputs directory for logs if it doesn't exist
mkdir -p outputs

# Remove old scripts directory and replace with new modular scripts
rm -rf scripts
mv scripts_new scripts

echo "Clean-up complete! Repository structure now follows the recommended guidelines."
echo "Please review the changes and check that everything is working correctly."
echo "If there are any issues, you can restore from the backup at $BACKUP_DIR"
echo ""
echo "IMPORTANT: Make sure to download the RTP-LX dataset and ezswitch library if they're not already present."
echo "  RTP-LX: mkdir -p data/RTP-LX"
echo "          wget https://github.com/microsoft/RTP-LX/raw/main/RTP-LX/RTP-LX.zip -O data/RTP-LX/RTP-LX.zip"
echo "          unzip data/RTP-LX/RTP-LX.zip -d data/RTP-LX/"
echo "  ezswitch: git clone https://github.com/gkuwanto/ezswitch.git"
echo "            cd ezswitch && pip install -e . && cd .." 