#!/bin/bash
#SBATCH --partition=rome # Use CPU partition
#SBATCH --job-name=08_compare_full # Updated job name
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G # Increased memory for potentially larger dataframes
#SBATCH --time=48:00:00 # Increased time for API calls on full dataset
#SBATCH --output=job_outputs/08_compare_full_%j.out # Updated output filename

# Activate conda environment
source /home/tchakravorty/.bashrc
conda activate code-switch

# Install required Python packages for analysis
pip install --quiet pandas matplotlib seaborn requests tqdm scipy # Added scipy for stats

# Define directories (using updated names for full run)
SRC_CONT_DIR="new_outputs/src_results_full"
TGT_CONT_DIR="new_outputs/tgt_results_full"
CS_PERSPECTIVE_DIR="new_outputs/perspective_full" # Directory with CS scores from script 05
OUTPUT_DIR="new_outputs/comparison_src_tgt_cs_results_full" # Comparison output dir for full run
# MAX_ROWS removed

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p job_outputs

# Get Perspective API key
PERSPECTIVE_API_KEY=$(python -c "import config; print(config.PERSPECTIVE_API_KEY)")

# Check API key
if [ -z "$PERSPECTIVE_API_KEY" ]; then
    echo "Error: PERSPECTIVE_API_KEY not found in config.py"
    exit 1
fi

# --- Models to process (using updated filenames for full run) --- 
MODELS=(
    # model_short_name  src_cont_file_basename                  tgt_cont_file_basename                  cs_persp_file_basename
    "llama31,llama31_src_continuations_full.csv,llama31_tgt_continuations_full.csv,llama31_continuations_perspective_local_full.csv"
    "aya,aya_src_continuations_full.csv,aya_tgt_continuations_full.csv,aya_continuations_perspective_local_full.csv"
    "llama3,llama3_src_continuations_full.csv,llama3_tgt_continuations_full.csv,llama3_continuations_perspective_local_full.csv"
)

# --- Loop through models --- 
for model_info in "${MODELS[@]}"; do
    IFS=',' read -r model_short_name src_cont_file tgt_cont_file cs_persp_file <<< "$model_info"
    
    echo "\n========================================================="
    echo "Comparing scores for Model: $model_short_name on FULL data"
    echo "========================================================="

    # Construct full paths
    SRC_CONT_PATH="$SRC_CONT_DIR/$src_cont_file"
    TGT_CONT_PATH="$TGT_CONT_DIR/$tgt_cont_file"
    CS_PERSP_PATH="$CS_PERSPECTIVE_DIR/$cs_persp_file"
    COMBINED_SCORES_PATH="$OUTPUT_DIR/${model_short_name}_combined_scores_full.csv" # Updated output filename

    # Check if required input files exist
    if [ ! -f "$SRC_CONT_PATH" ]; then echo "Error: SRC cont file not found: $SRC_CONT_PATH"; continue; fi
    if [ ! -f "$TGT_CONT_PATH" ]; then echo "Error: TGT cont file not found: $TGT_CONT_PATH"; continue; fi
    if [ ! -f "$CS_PERSP_PATH" ]; then echo "Error: CS perspective file not found: $CS_PERSP_PATH"; continue; fi

    # Run the comparison script (which now saves combined scores)
    # NOTE: compare_src_tgt_cs.py MUST be updated to handle unique_id for merging and no max_rows
    echo "Running comparison script (ensure it handles unique_id)..."
    python new_python_scripts/compare_src_tgt_cs.py \
        --src_cont_file "$SRC_CONT_PATH" \
        --tgt_cont_file "$TGT_CONT_PATH" \
        --cs_persp_file "$CS_PERSP_PATH" \
        --api_key "$PERSPECTIVE_API_KEY" \
        --output_dir "$OUTPUT_DIR" \
        --model_name "$model_short_name"
        # Removed --max_rows
        
    # Check if combined scores file was created before generating heatmap
    if [ -f "$COMBINED_SCORES_PATH" ]; then
        # Generate the heatmaps using the combined scores file
        echo "\n---> Generating heatmaps for $model_short_name..."
        # NOTE: generate_heatmaps.py already expects _full.csv input based on COMBINED_SCORES_PATH
        python new_python_scripts/generate_heatmaps.py \
            --combined_scores_file "$COMBINED_SCORES_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --model_name "$model_short_name"
        
        if [ $? -eq 0 ]; then
            echo "Successfully generated heatmaps for ${model_short_name}. Output in $OUTPUT_DIR"
        else
            echo "ERROR: Failed to generate heatmaps for ${model_short_name}."
        fi
    else
        echo "Warning: Combined scores file not found ($COMBINED_SCORES_PATH), skipping heatmap generation for $model_short_name."
        echo "Ensure compare_src_tgt_cs.py saved the file correctly and uses unique_id for merging."
    fi

done

echo "\nComparison and heatmap generation finished for full dataset. Results are in $OUTPUT_DIR" 