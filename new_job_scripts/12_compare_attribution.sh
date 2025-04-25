#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=12_comp_attr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=job_outputs/12_comp_attr_%j.out

# Activate conda environment
source /home/tchakravorty/.bashrc
conda activate code-switch

# Install required packages if needed
pip install --quiet pandas numpy matplotlib seaborn

# Define models to process
declare -a MODELS=("aya" "llama3" "llama31")
# We now use only the simple_attribution method since we've replaced the complex methods
declare -a METHODS=("simple_attribution")

# Create main output directory and job outputs
mkdir -p new_outputs/feature_attribution/comparison_results
mkdir -p job_outputs

# Process each LLM model
for MODEL in "${MODELS[@]}"
do
    echo "===================================================="
    echo "Starting attribution comparison for ${MODEL} model..."
    echo "===================================================="
    
    # Set variables for current model
    CS_DIR="new_outputs/feature_attribution/${MODEL}/code_switched"
    SRC_DIR="new_outputs/feature_attribution/${MODEL}/source_language"
    TGT_DIR="new_outputs/feature_attribution/${MODEL}/target_language"
    OUTPUT_DIR="new_outputs/feature_attribution/comparison_results/${MODEL}"
    
    # Create output directories for this model
    mkdir -p $OUTPUT_DIR
    
    # Check if input directories exist
    if [ ! -d "$CS_DIR" ] || [ ! -d "$SRC_DIR" ] || [ ! -d "$TGT_DIR" ]; then
        echo "Warning: One or more input directories do not exist for ${MODEL}."
        echo "Please run job script 11_feature_attribution.sh first for ${MODEL}."
        continue
    fi
    
    # Loop through each attribution method and compare results
    for METHOD in "${METHODS[@]}"
    do
        echo "Comparing attribution results for ${MODEL} using method: $METHOD"
        python new_python_scripts/compare_attribution_languages.py \
            --cs_dir $CS_DIR \
            --src_dir $SRC_DIR \
            --tgt_dir $TGT_DIR \
            --output_dir $OUTPUT_DIR/$METHOD \
            --method $METHOD
    done
    
    echo "Attribution comparison complete for ${MODEL}. Results saved to $OUTPUT_DIR"
    
    # Generate a summary HTML report for this model
    echo "Generating summary report for ${MODEL}..."
    cat > $OUTPUT_DIR/summary.html << EOL
<!DOCTYPE html>
<html>
<head>
    <title>Feature Attribution Comparison Summary - ${MODEL}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2 { color: #333; }
        .method { margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
        img { max-width: 100%; height: auto; margin: 10px 0; }
        .images { display: flex; flex-wrap: wrap; gap: 10px; }
        .image-container { flex: 1; min-width: 300px; }
    </style>
</head>
<body>
    <h1>Feature Attribution Comparison Summary - ${MODEL} Model</h1>
    <p>Generated on $(date)</p>
    
EOL

    # Add content for each method
    for METHOD in "${METHODS[@]}"
    do
        if [ -d "$OUTPUT_DIR/$METHOD" ]; then
            cat >> $OUTPUT_DIR/summary.html << EOL
    <div class="method">
        <h2>Token Attribution Analysis</h2>
        <div class="images">
            <div class="image-container">
                <h3>Mean Comparison</h3>
                <img src="$METHOD/${METHOD}_mean_comparison.png" alt="Mean comparison">
            </div>
            <div class="image-container">
                <h3>Distribution Comparison</h3>
                <img src="$METHOD/${METHOD}_distribution_violin.png" alt="Distribution comparison">
            </div>
        </div>
    </div>
EOL
        fi
    done

    # Close the HTML file
    cat >> $OUTPUT_DIR/summary.html << EOL
</body>
</html>
EOL

    echo "Summary report generated for ${MODEL} at $OUTPUT_DIR/summary.html"
    echo "----------------------------------------------------"
done

# Generate a main index HTML that links to all model reports
echo "Generating main index page..."
MAIN_OUTPUT_DIR="new_outputs/feature_attribution/comparison_results"
cat > $MAIN_OUTPUT_DIR/index.html << EOL
<!DOCTYPE html>
<html>
<head>
    <title>Feature Attribution Comparison - All Models</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2 { color: #333; }
        .model-card { margin-bottom: 20px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
        a { color: #0066cc; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>Feature Attribution Comparison - All Models</h1>
    <p>Generated on $(date)</p>
    
EOL

# Add links to each model's summary page
for MODEL in "${MODELS[@]}"
do
    if [ -f "$MAIN_OUTPUT_DIR/${MODEL}/summary.html" ]; then
        cat >> $MAIN_OUTPUT_DIR/index.html << EOL
    <div class="model-card">
        <h2>${MODEL} Model</h2>
        <p><a href="${MODEL}/summary.html">View detailed attribution comparison</a></p>
    </div>
EOL
    fi
done

# Close the main index HTML file
cat >> $MAIN_OUTPUT_DIR/index.html << EOL
</body>
</html>
EOL

echo "Main index page generated at $MAIN_OUTPUT_DIR/index.html"
echo "All attribution comparison jobs completed." 