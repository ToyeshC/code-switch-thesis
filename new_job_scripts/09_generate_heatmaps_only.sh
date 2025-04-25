#!/bin/bash
#SBATCH --job-name=generate_heatmaps
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=job_outputs/09_generate_heatmaps_%j.out

# Ensure the output directory exists
mkdir -p job_outputs

# Activate conda environment
echo "Activating conda environment..."
source /home/tchakravorty/.bashrc
conda activate code-switch
echo "Conda environment activated."

# --- Configuration ---
MODELS=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "CohereForAI/aya-23-8B"
)
OUTPUT_DIR="new_outputs/comparison_src_tgt_cs_results"
PYTHON_SCRIPT="new_python_scripts/generate_heatmaps.py"

# Ensure the comparison output directory exists (it should if previous steps ran)
mkdir -p ${OUTPUT_DIR}

# --- Generate Heatmaps ---
echo "--- Starting Heatmap Generation ---"

for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_NAME_SAFE=$(echo ${MODEL_NAME} | sed 's/\//-/g') # Replace '/' with '-' for filenames
    COMBINED_SCORES_FILE="${OUTPUT_DIR}/${MODEL_NAME_SAFE}_combined_scores.csv"
    HEATMAP_EN_CS_FILE="${OUTPUT_DIR}/${MODEL_NAME_SAFE}_heatmap_en_cs.png"
    HEATMAP_HI_CS_FILE="${OUTPUT_DIR}/${MODEL_NAME_SAFE}_heatmap_hi_cs.png"

    echo "Processing model: ${MODEL_NAME}"
    echo "Input combined scores file: ${COMBINED_SCORES_FILE}"

    if [ -f "${COMBINED_SCORES_FILE}" ]; then
        echo "Generating heatmaps for ${MODEL_NAME}..."
        python ${PYTHON_SCRIPT} \
            --input_file "${COMBINED_SCORES_FILE}" \
            --output_heatmap_en_cs "${HEATMAP_EN_CS_FILE}" \
            --output_heatmap_hi_cs "${HEATMAP_HI_CS_FILE}"

        if [ $? -eq 0 ]; then
            echo "Successfully generated heatmaps for ${MODEL_NAME}."
        else
            echo "ERROR: Failed to generate heatmaps for ${MODEL_NAME}."
        fi
    else
        echo "ERROR: Combined scores file not found for ${MODEL_NAME}: ${COMBINED_SCORES_FILE}. Skipping heatmap generation."
        echo "Please ensure step 6 of the comparison script (08_compare_src_tgt_cs.sh) completed successfully."
    fi
    echo "----------------------------------------"
done

echo "--- Heatmap Generation Finished ---"

conda deactivate
echo "Conda environment deactivated."
echo "Job finished." 