# Code-Switching Toxicity Analysis

This repository contains a pipeline for analyzing toxicity in code-switched text prompts and model responses, with a focus on Hindi-English code-switching. It compares how language models respond to monolingual vs. code-switched inputs, particularly analyzing differences in toxicity metrics.

## Repository Structure

The clean repository structure is organized as follows:

```
code-switch-thesis/
├── data/
│   ├── extracted_prompts/       # Hindi and English prompt data
│   ├── RTP-LX/                  # RTP-LX datasets (needs to be downloaded)
│   └── output/                  # Analysis outputs and results
├── ezswitch/                    # ezswitch library (needs to be cloned)
├── scripts/                     # Modular execution scripts
│   ├── 01_extract_rtp_data.sh   # Extract data from RTP-LX
│   ├── 02_language_detection.sh # Detect language in prompts
│   ├── 03_filter_code_switch.sh # Filter for code-switched content
│   ├── 04_generate_responses.sh # Generate LLM responses
│   ├── 05_analyze_toxicity.sh   # Analyze toxicity of prompts/responses
│   └── 06_visualization.sh      # Generate visualizations
├── src/                         # Core Python implementation
│   ├── add_primary_key.py       # Add unique identifiers to prompts
│   ├── analyze_toxicity_with_id.py    # Toxicity analysis with ID tracking
│   ├── compare_toxicity_with_id.py    # Compare prompt and response toxicity
│   ├── config.py                # Configuration settings
│   ├── filter_language_mix_with_id.py # Filter for code-switched sentences
│   ├── generate_model_responses_with_id.py # Generate model responses
│   └── language_detection_with_id.py  # Detect language composition
├── lid.176.bin                  # FastText language identification model
├── .gitignore                   # Git ignore file
└── README.md                    # This file
```

## Setup

### Prerequisites

- Python 3.8+
- PyTorch
- HuggingFace Transformers
- FastText language identification model (`lid.176.bin`)
- Perspective API key (for toxicity analysis)

### 1. Download RTP-LX Dataset

The analysis pipeline uses the RTP-LX dataset which needs to be downloaded and placed in the data directory:

```bash
# Create the RTP-LX directory
mkdir -p data/RTP-LX

# Download the RTP-LX zip file from Microsoft's repository
wget https://github.com/microsoft/RTP-LX/raw/main/RTP-LX/RTP-LX.zip -O data/RTP-LX/RTP-LX.zip

# Unzip the file
unzip data/RTP-LX/RTP-LX.zip -d data/RTP-LX/

# Ensure all language JSONs are present in the data/RTP-LX directory
ls data/RTP-LX/*.json
```

The RTP-LX dataset contains pre-defined prompts in multiple languages, which our scripts will extract and process. The extracted prompts will automatically be placed in the `data/extracted_prompts` directory during the first step of the pipeline (`01_extract_rtp_data.sh`).

### 2. Set Up ezswitch Library

The ezswitch library is used for code-switching analysis and needs to be set up separately:

```bash
# Clone ezswitch repository from gkuwanto's fork
git clone https://github.com/gkuwanto/ezswitch.git

# Navigate to ezswitch directory
cd ezswitch

# Install the ezswitch package in development mode
pip install -e .

# Return to the main project directory
cd ..
```

This library helps with analyzing and generating code-switched sentences according to linguistic constraints.

### 3. Configure API Keys

```bash
# Set up your Hugging Face API token
export HUGGING_FACE_HUB_TOKEN=your_token_here

# Add your Perspective API key to src/config.py
echo 'PERSPECTIVE_API_KEY = "your_key_here"' > src/config.py
```

## Core Functionality

The codebase implements a pipeline with the following key components:

1. **Primary Key Assignment**: Adding unique identifiers to prompts to track them through the pipeline
2. **Language Detection**: Analyzing Hindi and English content in each prompt
3. **Language Filtering**: Identifying balanced code-switched sentences
4. **Model Response Generation**: Getting responses from LLMs (LLaMA, Aya) for prompts
5. **Toxicity Analysis**: Measuring toxicity metrics in prompts and responses
6. **Visualization**: Generating histograms, boxplots, and comparison charts

## Execution Pipeline

The analysis is split into modular scripts that should be executed in numerical order:

### 1. Data Extraction (`01_extract_rtp_data.sh`)
Extracts Hindi and English prompts from the RTP-LX dataset and assigns primary keys. This script creates the initial dataset in the `data/extracted_prompts` directory and prepares it with unique identifiers for tracking through the pipeline.

### 2. Language Detection (`02_language_detection.sh`)
Analyzes the language composition of each prompt, identifying Hindi (Devanagari & Romanized) and English words.

### 3. Code-Switch Filtering (`03_filter_code_switch.sh`)
Filters prompts to retain balanced code-switched content based on language composition.

### 4. Response Generation (`04_generate_responses.sh`)
Generates responses from language models (LLaMA, Aya) for the filtered prompts.

### 5. Toxicity Analysis (`05_analyze_toxicity.sh`)
Analyzes toxicity metrics for both prompts and model responses using the Perspective API.

### 6. Visualization (`06_visualization.sh`)
Creates visualizations comparing toxicity between monolingual and code-switched content.

## Running the Analysis

You can run the full pipeline sequentially:

```bash
# Run each script in order
sbatch scripts/01_extract_rtp_data.sh
# Wait for completion, then
sbatch scripts/02_language_detection.sh
# And so on...
```

Alternatively, for testing purposes with Hindi prompts only:

```bash
# For testing with Hindi prompts only
sbatch scripts/test_primary_key_hindi_only.sh
```

Results will be stored in the `data/output/` directory, including:
- Prompt and response CSVs with primary keys
- Language detection and filtering results
- Toxicity analysis for prompts and responses
- Visualization charts in PNG format
- Primary key verification reports

## Visualizations

The analysis generates several types of visualizations:

1. **Histograms**: Distribution of toxicity metrics for prompts and responses
2. **Boxplots**: Comparison of toxicity between prompts and model responses
3. **Comparison Charts**: Differences in toxicity between monolingual and code-switched inputs

All visualizations are saved in PNG format in the `visualizations/` directory within the output folder.

## Primary Key Tracking

The pipeline implements primary key tracking to ensure data consistency throughout the analysis. Each prompt is assigned a unique ID that follows it through:
- Language detection
- Filtering
- Model response generation
- Toxicity analysis
- Result comparison

A verification report is generated to confirm the integrity of this tracking.

## Repository Cleanup Details

This repository has been cleaned up to improve organization and maintainability. Here's why certain files were removed:

1. **Redundant Scripts**: Multiple numbered scripts (e.g., `0_extract_prompts.sh` through `22_model_toxicity_analysis_only.sh`) were consolidated into more descriptive, modular scripts with clear functional purposes. The original scripts often contained redundant or overlapping functionality.

2. **Superseded Testing Scripts**: Files like `test_primary_key_100_points.sh` and `test_primary_key_limited_requests.sh` were superseded by the more comprehensive `test_primary_key_hindi_only.sh`, which includes additional verification and visualization steps.

3. **Non-ID-Tracking Files**: Early versions of processing scripts (e.g., `language_detection.py` instead of `language_detection_with_id.py`) lacked the primary key tracking functionality necessary for proper end-to-end analysis. Only the ID-tracking versions are kept for consistency.

4. **Legacy Analysis Files**: Older scripts with outdated approaches or partial implementations were removed to avoid confusion, as they've been replaced by more sophisticated versions that incorporate best practices.

The current structure focuses only on the essential components needed for the analysis pipeline, making it easier to understand and maintain. 