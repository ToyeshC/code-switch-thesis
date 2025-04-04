#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --partition=rome
#SBATCH --job-name=05b_language_detection
#SBATCH --mem=16G
#SBATCH --output=/home/tchakravorty/tchakravorty/code-switch-thesis/outputs/05b_language_detection.out

# Load necessary modules
module load 2023
module load Miniconda3/23.5.2-0
source ~/.bashrc
conda activate code-switch

# Define languages - can be modified for other language pairs
BASE_LANG="hindi"
SOURCE_LANG="english"

# Get the project root directory using an absolute path
PROJECT_ROOT="/home/tchakravorty/tchakravorty/code-switch-thesis"
# Ensure we're in the PROJECT_ROOT directory
cd $PROJECT_ROOT
echo "Current working directory: $(pwd)"

# Define Indic NLP library path
INDIC_NLP_LIB_DIR="${PROJECT_ROOT}/indic_nlp_library"

# Check if Indic NLP library exists
if [ ! -d "${INDIC_NLP_LIB_DIR}" ]; then
    echo "Warning: Indic NLP library not found at ${INDIC_NLP_LIB_DIR}"
    echo "The script will fall back to simpler language detection methods."
else
    echo "Indic NLP library found at ${INDIC_NLP_LIB_DIR}"
fi

# Install indic-nlp-library if not already installed
pip install indic-nlp-library --quiet

# Define paths with absolute paths to avoid path resolution issues
INPUT_DIR="${PROJECT_ROOT}/data/output/${BASE_LANG}"
ID_MAP_DIR="${PROJECT_ROOT}/data/id_mappings"
OUTPUT_DIR="${PROJECT_ROOT}/data/output/language_detection"

# Check if Python and the necessary modules can be found
echo "Python version:"
python --version
echo "Checking if language_detection_with_id.py exists:"
ls -la ${PROJECT_ROOT}/src/language_detection_with_id.py
echo "Checking if indic_language_detection.py exists:"
ls -la ${PROJECT_ROOT}/src/indic_language_detection.py
echo "Checking if FastText model exists:"
ls -la ${PROJECT_ROOT}/lid.176.bin
echo "Checking if Indic NLP library exists:"
ls -la ${INDIC_NLP_LIB_DIR} 2>/dev/null || echo "Indic NLP library not found"

# Create output directory
mkdir -p $OUTPUT_DIR
echo "Created output directory: $OUTPUT_DIR"

# Make sure Python can find the modules and Indic NLP library
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT:$INDIC_NLP_LIB_DIR
echo "PYTHONPATH set to: $PYTHONPATH"

# Use the compiled Hindi CSV file if it exists
COMPILED_CSV="${INPUT_DIR}/compile_hindi.csv"

if [ -f "$COMPILED_CSV" ]; then
    echo "Using compiled Hindi CSV file: $COMPILED_CSV"
    
    echo "Running FastText detection on compiled CSV..."
    python ${PROJECT_ROOT}/src/language_detection_with_id.py \
        --input_file $COMPILED_CSV \
        --output_file $OUTPUT_DIR/compiled_${BASE_LANG}_fasttext_detection.csv \
        --fasttext_model ${PROJECT_ROOT}/lid.176.bin \
        --is_compiled_csv
    
    echo "Running Indic LID detection on compiled CSV..."
    python ${PROJECT_ROOT}/src/indic_language_detection.py \
        --input_file $COMPILED_CSV \
        --output_file $OUTPUT_DIR/compiled_${BASE_LANG}_indic_detection.csv \
        --is_compiled_csv \
        --indic_nlp_path ${INDIC_NLP_LIB_DIR}
else
    echo "Compiled CSV file not found. Using regular full_responses.csv."
    echo "Looking for full_responses.csv at: $INPUT_DIR/full_responses.csv"
    
    if [ ! -f "$INPUT_DIR/full_responses.csv" ]; then
        echo "ERROR: full_responses.csv not found at $INPUT_DIR"
        echo "Current directory is: $(pwd)"
        echo "Available files in $INPUT_DIR:"
        ls -la $INPUT_DIR
        
        # Check if there are other CSV files we can use (like full_aya.csv or full_llama.csv)
        if [ -f "$INPUT_DIR/full_aya.csv" ] && [ -f "$INPUT_DIR/full_llama.csv" ]; then
            echo "Found alternative CSV files. Creating a combined full_responses.csv"
            # Combine the CSVs while keeping only one header
            head -n 1 "$INPUT_DIR/full_aya.csv" > "$INPUT_DIR/full_responses.csv"
            tail -n +2 "$INPUT_DIR/full_aya.csv" >> "$INPUT_DIR/full_responses.csv"
            tail -n +2 "$INPUT_DIR/full_llama.csv" >> "$INPUT_DIR/full_responses.csv"
            echo "Created combined file at $INPUT_DIR/full_responses.csv"
        else
            exit 1
        fi
    fi
    
    echo "Running FastText detection on full_responses.csv..."
    python ${PROJECT_ROOT}/src/language_detection_with_id.py \
        --input_file $INPUT_DIR/full_responses.csv \
        --id_map $ID_MAP_DIR/${BASE_LANG}_id_map.json \
        --output_file $OUTPUT_DIR/${BASE_LANG}_fasttext_detection.csv \
        --fasttext_model ${PROJECT_ROOT}/lid.176.bin

    echo "Running Indic LID detection on full_responses.csv..."
    python ${PROJECT_ROOT}/src/indic_language_detection.py \
        --input_file $INPUT_DIR/full_responses.csv \
        --id_map $ID_MAP_DIR/${BASE_LANG}_id_map.json \
        --output_file $OUTPUT_DIR/${BASE_LANG}_indic_detection.csv \
        --indic_nlp_path ${INDIC_NLP_LIB_DIR}
fi

echo "Language detection complete. Results saved to $OUTPUT_DIR"
