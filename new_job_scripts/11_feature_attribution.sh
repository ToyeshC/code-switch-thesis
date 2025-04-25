#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=11_feat_attr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=job_outputs/11_feat_attr_%j.out

# Activate conda environment
source /home/tchakravorty/.bashrc
conda activate code-switch

# Install required packages if needed
pip install --quiet transformers captum matplotlib seaborn pandas torch ipython

# Set environment variables for better error messages
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TOKENIZERS_PARALLELISM=false

# Define models to process
declare -a MODELS=("aya" "llama3" "llama31")
TEXT_COLUMN="generated"
MAX_SAMPLES=100  # Limit for quicker analysis

# Function to check CSV column names and determine the text column
get_column_name() {
    local file="$1"
    local primary="$2"
    local fallback="$3"
    
    # Check if file exists
    if [ ! -f "$file" ]; then
        echo "not_found"
        return
    fi
    
    # Get header line from CSV
    local header=$(head -n 1 "$file")
    
    # Check if primary column exists
    if [[ "$header" == *"$primary"* ]]; then
        echo "$primary"
    # Check if fallback column exists
    elif [[ "$header" == *"$fallback"* ]]; then
        echo "$fallback"
    # Try to find a reasonable column containing 'text', 'content', or 'continuation'
    else
        if [[ "$header" == *"text"* ]]; then
            echo "text"
        elif [[ "$header" == *"content"* ]]; then
            echo "content"
        elif [[ "$header" == *"continuation"* ]]; then
            # Get the first column with 'continuation' in the name
            echo $(echo "$header" | tr ',' '\n' | grep "continuation" | head -n 1 | tr -d ' ')
        else
            # Return the second column as a last resort
            echo $(echo "$header" | tr ',' '\n' | sed -n '2p' | tr -d ' ')
        fi
    fi
}

# Create main output directory and job outputs
mkdir -p new_outputs/feature_attribution
mkdir -p job_outputs

# Process each LLM model
for MODEL in "${MODELS[@]}"
do
    echo "===================================================="
    echo "Starting feature attribution analysis for ${MODEL} model..."
    echo "===================================================="
    
    # Set variables for current model
    MODEL_DIR="new_outputs/models/${MODEL}_toxicity_classifier"
    OUTPUT_DIR="new_outputs/feature_attribution/${MODEL}"
    CS_INPUT="new_outputs/perspective/${MODEL}_continuations_perspective_local.csv"
    SRC_INPUT="new_outputs/src_results/${MODEL}_src_continuations.csv"
    TGT_INPUT="new_outputs/tgt_results/${MODEL}_tgt_continuations.csv"
    
    # Create output directories for this model
    mkdir -p $OUTPUT_DIR/code_switched
    mkdir -p $OUTPUT_DIR/source_language
    mkdir -p $OUTPUT_DIR/target_language
    
    # Check if model directory exists
    if [ ! -d "$MODEL_DIR" ]; then
        echo "Warning: Model directory not found at $MODEL_DIR"
        echo "Please run job script 10_finetune_classifier.sh first for ${MODEL}."
        continue
    fi
    
    # Check if input files exist
    if [ ! -f "$CS_INPUT" ]; then
        echo "Warning: Code-switched input file $CS_INPUT not found. Skipping ${MODEL} code-switched analysis..."
    else
        # Get column name for code-switched text
        CS_TEXT_COL=$(get_column_name "$CS_INPUT" "$TEXT_COLUMN" "generated")
        echo "Using column '$CS_TEXT_COL' for code-switched text in $CS_INPUT"
        
        # Run feature attribution for code-switched text
        echo "Analyzing ${MODEL} code-switched text with feature attribution..."
        python new_python_scripts/feature_attribution.py \
            --model_dir $MODEL_DIR \
            --input_file $CS_INPUT \
            --output_dir $OUTPUT_DIR/code_switched \
            --text_column $CS_TEXT_COL \
            --max_samples $MAX_SAMPLES
    fi
    
    # Check if source language file exists
    if [ ! -f "$SRC_INPUT" ]; then
        echo "Warning: Source language input file $SRC_INPUT not found. Skipping ${MODEL} source language analysis..."
    else
        # Get column name for source language text
        SRC_TEXT_COL=$(get_column_name "$SRC_INPUT" "src_continuation" "text")
        echo "Using column '$SRC_TEXT_COL' for source language text in $SRC_INPUT"
        
        # Run feature attribution for source language
        echo "Analyzing ${MODEL} source language text with feature attribution..."
        python new_python_scripts/feature_attribution.py \
            --model_dir $MODEL_DIR \
            --input_file $SRC_INPUT \
            --output_dir $OUTPUT_DIR/source_language \
            --text_column $SRC_TEXT_COL \
            --max_samples $MAX_SAMPLES
    fi
    
    # Check if target language file exists
    if [ ! -f "$TGT_INPUT" ]; then
        echo "Warning: Target language input file $TGT_INPUT not found. Skipping ${MODEL} target language analysis..."
    else
        # Get column name for target language text
        TGT_TEXT_COL=$(get_column_name "$TGT_INPUT" "tgt_continuation" "text")
        echo "Using column '$TGT_TEXT_COL' for target language text in $TGT_INPUT"
        
        # Run feature attribution for target language
        echo "Analyzing ${MODEL} target language text with feature attribution..."
        python new_python_scripts/feature_attribution.py \
            --model_dir $MODEL_DIR \
            --input_file $TGT_INPUT \
            --output_dir $OUTPUT_DIR/target_language \
            --text_column $TGT_TEXT_COL \
            --max_samples $MAX_SAMPLES
    fi
    
    echo "Feature attribution analysis complete for ${MODEL}. Results saved to $OUTPUT_DIR"
    echo "----------------------------------------------------"
done

echo "All feature attribution jobs completed." 