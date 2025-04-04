# Code-Switching Toxicity Analysis

This repository contains a pipeline for analyzing toxicity in code-switched text prompts and model responses, with a focus on Hindi-English code-switching. It compares how language models respond to monolingual vs. code-switched inputs, particularly analyzing differences in toxicity metrics.

## Repository Structure

The repository structure is organized as follows:

```
code-switch-thesis/
├── data/
│   ├── extracted_prompts/        # Text files for Hindi and English prompts
│   ├── translate_api_outputs/    # Machine translations 
│   ├── alignments/               # Word alignments between languages
│   ├── RTP-LX/                   # RTP-LX datasets (needs to be downloaded)
│   └── output/                   # Analysis outputs and results
├── ezswitch/                     # ezswitch library (needs to be cloned)
├── scripts/                      # Modular execution scripts
│   ├── 00_extract_rtp_data.sh    # Extract data from RTP-LX
│   ├── 01_extract_prompts_to_text.sh # Extract prompts to text files
│   ├── 02_translate_files.sh     # Translate using Google Translate API
│   ├── 03_generate_alignments.sh # Generate word alignments
│   ├── 04_generate_code_switched_outputs.sh # Generate code-switched outputs
│   ├── 05_language_detection.sh  # Detect language in prompts
│   ├── 06_filter_code_switch.sh  # Filter for code-switched content
│   ├── 07_generate_responses.sh  # Generate LLM responses
│   ├── 08_analyze_toxicity.sh    # Analyze toxicity of prompts/responses
│   └── 09_visualization.sh       # Generate visualizations
├── src/                          # Core Python implementation
│   ├── add_primary_key.py        # Add unique identifiers to prompts
│   ├── extract_prompts_to_text.py # Extract prompts to text files
│   ├── translate_file.py         # Translate files using Google Translate
│   ├── analyze_toxicity_with_id.py    # Toxicity analysis with ID tracking
│   ├── compare_toxicity_with_id.py    # Compare prompt and response toxicity
│   ├── config.py                 # Configuration settings
│   ├── filter_language_mix_with_id.py # Filter for code-switched sentences
│   ├── generate_model_responses_with_id.py # Generate model responses
│   └── language_detection_with_id.py  # Detect language composition
├── lid.176.bin                   # FastText language identification model
├── .gitignore                    # Git ignore file
└── README.md                     # This file
```

## Setup

### Prerequisites

- Python 3.8+
- PyTorch
- HuggingFace Transformers
- FastText language identification model (`lid.176.bin`)
- Perspective API key (for toxicity analysis)
- Google Translate Python library (`googletrans`)

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

The RTP-LX dataset contains pre-defined prompts in multiple languages, which our scripts will extract and process.

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
2. **Translation**: Translating prompts between languages using Google Translate
3. **Alignment**: Generating word alignments between languages
4. **Code-Switching**: Generating code-switched outputs using ezswitch
5. **Language Detection**: Analyzing Hindi and English content in each prompt
6. **Language Filtering**: Identifying balanced code-switched sentences
7. **Model Response Generation**: Getting responses from LLMs (LLaMA, Aya) for prompts
8. **Toxicity Analysis**: Measuring toxicity metrics in prompts and responses
9. **Visualization**: Generating histograms, boxplots, and comparison charts

## Execution Pipeline

The analysis is split into modular scripts that should be executed in numerical order:

### 0. Data Extraction (`00_extract_rtp_data.sh`)
Extracts Hindi and English prompts from the RTP-LX dataset and assigns primary keys. This script creates the initial dataset with unique identifiers for tracking through the pipeline.

### 1. Extract Prompts to Text (`01_extract_prompts_to_text.sh`)
Extracts prompts from CSV files to plain text files for translation and alignment.

### 2. Translate Files (`02_translate_files.sh`)
Translates the text files between Hindi and English using Google Translate API.

### 3. Generate Alignments (`03_generate_alignments.sh`)
Generates word alignments between the original texts and their translations.

### 4. Generate Code-Switched Outputs (`04_generate_code_switched_outputs.sh`)
Uses the ezswitch library to generate code-switched outputs based on the alignments.

### 5. Language Detection (`05_language_detection.sh`)
Analyzes the language composition of each prompt, identifying Hindi (Devanagari & Romanized) and English words.

### 6. Code-Switch Filtering (`06_filter_code_switch.sh`)
Filters prompts to retain balanced code-switched content based on language composition.

### 7. Response Generation (`07_generate_responses.sh`)
Generates responses from language models (LLaMA, Aya) for the filtered prompts.

### 8. Toxicity Analysis (`08_analyze_toxicity.sh`)
Analyzes toxicity metrics for both prompts and model responses using the Perspective API.

### 9. Visualization (`09_visualization.sh`)
Creates visualizations comparing toxicity between monolingual and code-switched content.

## Running the Analysis

You can run the full pipeline sequentially:

```bash
# Run each script in order
sbatch scripts/00_extract_rtp_data.sh
# Wait for completion, then
sbatch scripts/01_extract_prompts_to_text.sh
# Wait for completion, then
sbatch scripts/02_translate_files.sh
# And so on...
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

## Language Extensibility

The pipeline is designed to be extensible to other language pairs beyond Hindi-English. To use a different language pair:

1. Update the language variables in each script:
   ```bash
   # Define languages - can be modified for other language pairs
   BASE_LANG="your_base_language"
   SOURCE_LANG="your_source_language"
   
   # Define language codes for file naming/translation
   BASE_LANG_CODE="base_code"  # e.g., "es" for Spanish
   SOURCE_LANG_CODE="source_code"  # e.g., "de" for German
   ```

2. Ensure you have the corresponding RTP-LX dataset files for your languages:
   ```
   RTP_LX_XX.json  # where XX is your language code, e.g., ES for Spanish
   ```

3. The rest of the pipeline will automatically adapt to the new language pair.

Note: For best results, ensure the FastText language identification model supports your target languages. 