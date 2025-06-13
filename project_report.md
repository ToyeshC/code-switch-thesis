# Code-Switching Thesis Project Report

## Overview
This report documents the work done on a code-switching thesis project, focusing on analyzing and processing bilingual (Hindi-English) text data, with a particular emphasis on code-switching patterns, toxicity analysis, and language model evaluations.

## Main Pipeline Components

### 1. Data Extraction and Translation
- **Initial Data Extraction** (`0_extract_prompts.sh`)
  - Extracts prompts from RTP-LX dataset in both Hindi and English
  - Processes files: `RTP_LX_HI.json` and `RTP_LX_EN.json`
  - Outputs: `train_hi.txt` and `train_en.txt`

- **Translation Pipeline** (`1_translate_file.sh`)
  - Translates English prompts to Hindi and vice versa
  - Uses translation API to generate parallel corpora
  - Creates silver-standard translations for comparison

### 2. Alignment and Analysis
- **Alignment Generation** (`2_get_alignment_hi.sh`, `6_en_all.sh`)
  - Uses GIZA++ for word alignment between Hindi and English
  - Generates both gold and silver alignments
  - Creates alignment files for both language directions (en-hi and hi-en)

### 3. Response Generation and Model Evaluation
- **Response Generation** (`3_generate_response_hi.sh`)
  - Uses multiple language models for generation:
    - CohereForAI/aya-23-8B
    - Meta-Llama-3.1-8B-Instruct
    - Meta-Llama-3-8B-Instruct
  - Generates responses in both Hindi and English

### 4. Data Compilation
- **Compilation Scripts** (`4_compile_english.sh`, `4_compile_hindi.sh`)
  - Compiles results from various model outputs
  - Creates consolidated CSV files for analysis

## Advanced Analysis Scripts (new_job_scripts)

### 1. Model Evaluation and Analysis
- **Perplexity Analysis**
  - `15_perplexity.sh`: Basic perplexity calculations
    - Computes perplexity scores for generated text using multiple language models
    - Analyzes perplexity patterns across different language conditions
  - `17_perplexity_check.sh`: Advanced perplexity verification
    - Implements outlier detection using multiple methods (IQR, Z-score, Percentile)
    - Performs statistical significance testing on perplexity scores
  - `19_run_perplexity_on_srctgt_full_gen.sh`: Full generation perplexity analysis
    - Comprehensive perplexity analysis across source and target languages
    - Generates comparative visualizations of perplexity distributions

### 2. Toxicity Analysis
- **Toxicity Evaluation**
  - `16_rtplx_toxicity_compare.sh`: RTP-LX dataset toxicity comparison
    - Compares toxicity scores between original and generated text
    - Analyzes toxicity patterns across different language models
  - `18_tox_perplexity_in_out_compare.sh`: Toxicity-perplexity correlation
    - Performs correlation analysis between toxicity and perplexity scores
    - Generates scatter plots and correlation matrices
    - Implements statistical significance testing (p-value < 0.05)
  - `21_add_perspective_generated.sh`: Perspective API integration
    - Integrates Google's Perspective API for toxicity scoring
    - Analyzes toxicity patterns in code-switched text

### 3. Feature Analysis
- **Feature Attribution**
  - `11_feature_attribution.sh`: Feature importance analysis
    - Uses Captum library for model interpretability
    - Analyzes feature importance across three models:
      - CohereForAI/aya-23-8B
      - Meta-Llama-3.1-8B-Instruct
      - Meta-Llama-3-8B-Instruct
    - Generates attribution scores for code-switched, source, and target language text
  - `12_compare_attribution.sh`: Attribution comparison
    - Compares feature attributions across different models
    - Analyzes patterns in token-level importance
  - `13_correlation_plots.sh`: Correlation visualization
    - Creates comprehensive correlation matrices
    - Generates heatmaps with statistical significance markers
    - Analyzes relationships between:
      - Language usage patterns
      - Perplexity scores
      - Toxicity metrics
    - Implements outlier detection and removal methods:
      - IQR (Interquartile Range)
      - Z-score
      - Percentile-based

### 4. Code-Switching Analysis
- **Code-Switching Studies**
  - `08_compare_src_tgt_cs.sh`: Source-target code-switching comparison
    - Analyzes code-switching patterns between source and target languages
    - Generates comparative statistics and visualizations
  - `07_generate_src_tgt_local.sh`: Local code-switching generation
    - Implements local generation strategies for code-switched text
    - Analyzes generation quality and patterns
  - `05_generate_continuations_local.sh`: Continuation generation
    - Generates continuations for code-switched text
    - Analyzes continuation quality and coherence

## Tweet Analysis Scripts

### 1. Data Processing
- `process_hinglish.py`: Processes Hinglish (Hindi-English mixed) tweets
  - Implements text cleaning and normalization
  - Handles mixed script processing
  - Generates language identification labels
- `exploratory_data_analysis.py`: Comprehensive EDA of tweet data
  - Analyzes language distribution patterns
  - Generates statistical summaries
  - Creates visualizations of language mixing patterns

### 2. Analysis Tools
- `perplexity_toxicity_correlation.py`: Analyzes relationship between perplexity and toxicity
  - Implements multiple correlation analysis methods:
    - Pearson correlation with significance testing
    - Outlier detection and removal
    - Correlation heatmaps with p-value markers
  - Generates visualizations:
    - Scatter plots with correlation coefficients
    - Distribution plots
    - Outlier analysis plots
- `analyze_perplexity_language_toxicity.py`: Language-specific toxicity analysis
  - Analyzes toxicity patterns across different language conditions
  - Implements statistical testing for language-specific effects
  - Generates comparative visualizations

## Technical Infrastructure
- Uses SLURM for job scheduling
- Runs on GPU and CPU partitions
- Utilizes Python with various ML libraries
- Integrates with Hugging Face models and APIs

## Key Features
1. Bilingual processing (Hindi-English)
2. Multiple model evaluations
3. Toxicity analysis
4. Perplexity measurements
5. Code-switching pattern analysis
6. Feature attribution studies
7. Comprehensive visualization tools

## Output Structure
- Results stored in `data/output/` directory
- Separate directories for Hindi and English outputs
- Compiled results in CSV format
- Various analysis plots and visualizations 