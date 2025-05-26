#!/bin/bash

# Define variables
METRICS_DIR="new_outputs/classifier_metrics"
OUTPUT_DIR="new_outputs/classifier_comparison"
OUTPUT_FILE="${OUTPUT_DIR}/classifier_metrics_comparison.html"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Run the comparison script
echo "Comparing classifier metrics..."
python new_python_scripts/compare_classifier_metrics.py \
    --metrics_dir ${METRICS_DIR} \
    --output_file ${OUTPUT_FILE}

echo "Comparison completed. Results saved to ${OUTPUT_FILE}" 