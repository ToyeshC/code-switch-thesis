# Guide to Cleaning the Repository

This guide details which files to keep and which to remove to achieve a clean, well-organized repository structure for the code-switching toxicity analysis project.

## Files to Keep

### Top-level Files
- `lid.176.bin` - Essential FastText language identification model
- `.gitignore` - Git configuration
- `README.md` - Project documentation

### Core Source Files (`src/`)
1. **Primary Key Management**
   - `add_primary_key.py` - Adds tracking IDs to prompts

2. **Language Analysis**
   - `language_detection_with_id.py` - Detects language composition with ID tracking
   - `filter_language_mix_with_id.py` - Filters for code-switched content

3. **Model Interaction**
   - `generate_model_responses_with_id.py` - Handles model response generation

4. **Toxicity Analysis**
   - `analyze_toxicity_with_id.py` - Performs toxicity analysis
   - `compare_toxicity_with_id.py` - Compares prompt and response toxicity

5. **Configuration**
   - `config.py` - Contains API keys and configuration settings

### Main Scripts (`scripts/`)
The scripts follow a modular structure with numbered prefixes to indicate execution order:

1. **Data Extraction**
   - `01_extract_rtp_data.sh` - Extracts data from RTP-LX dataset

2. **Language Processing**
   - `02_language_detection.sh` - Detects language in prompts
   - `03_filter_code_switch.sh` - Filters for code-switched content

3. **Model Interaction**
   - `04_generate_responses.sh` - Generates LLM responses for prompts

4. **Analysis**
   - `05_analyze_toxicity.sh` - Analyzes toxicity of prompts and responses
   - `06_visualization.sh` - Generates visualizations and reports

5. **Testing Script**
   - `test_primary_key_hindi_only.sh` - For testing with Hindi prompts only

### Data Directories (`data/`)
1. **Input Data**
   - `data/extracted_prompts/` - Contains Hindi and English prompts
   - `data/RTP-LX/` - Contains RTP-LX datasets (should be downloaded)

2. **Output Structure**
   - `data/output/` - For analysis results

### External Libraries
- `ezswitch/` - Code-switching analysis library (should be cloned)

## Files That Can Be Removed

### Redundant or Superseded Scripts
- All standalone numbered scripts without descriptive prefixes (`0_extract_prompts.sh` through `22_model_toxicity_analysis_only.sh`)
- `test_primary_key_100_points.sh` - Superseded by `test_primary_key_hindi_only.sh`
- `test_primary_key_limited_requests.sh` - Superseded by newer implementations
- `run_original_analysis.sh` - Replaced by modular scripts
- `run_bert_toxicity_analysis.sh` - Not part of the core pipeline

### Older Source Files
- Files without ID tracking (e.g., `language_detection.py` instead of `language_detection_with_id.py`)
- `analyze_toxicity.py` - Replaced by `analyze_toxicity_with_id.py`
- `filter_language_mix.py` - Replaced by `filter_language_mix_with_id.py`
- `generate_continuations.py` - Replaced by `generate_model_responses_with_id.py`

## Suggested Clean-Up Process

1. **Create a Backup**
   ```bash
   cp -r code-switch-thesis code-switch-thesis-backup
   ```

2. **Set Up Required External Dependencies**
   ```bash
   # Download and unzip RTP-LX dataset
   mkdir -p data/RTP-LX
   wget https://github.com/microsoft/RTP-LX/raw/main/RTP-LX/RTP-LX.zip -O data/RTP-LX/RTP-LX.zip
   unzip data/RTP-LX/RTP-LX.zip -d data/RTP-LX/
   
   # Clone ezswitch repository (from gkuwanto's fork)
   git clone https://github.com/gkuwanto/ezswitch.git
   
   # Set up ezswitch
   cd ezswitch
   pip install -e .
   cd ..
   ```

3. **Create Modular Scripts**
   ```bash
   # Create new modular scripts
   mkdir -p scripts_new
   
   # Create script template for each step
   cat > scripts_new/01_extract_rtp_data.sh << 'EOF'
   #!/bin/bash
   # Script to extract data from RTP-LX dataset
   # Logic from existing extract_prompts.py
   EOF
   
   # Make them executable
   chmod +x scripts_new/01_extract_rtp_data.sh
   # Repeat for other scripts
   ```

4. **Remove Unnecessary Files**
   ```bash
   # Remove redundant scripts
   rm scripts/[0-9]*.sh
   rm scripts/test_primary_key_100_points.sh
   rm scripts/test_primary_key_limited_requests.sh
   
   # Remove older source files
   rm src/language_detection.py
   rm src/filter_language_mix.py
   rm src/analyze_toxicity.py
   rm src/generate_continuations.py
   rm src/analyse_comments.py
   rm src/extract_generated_sentences.py
   ```

5. **Organize Output Directory**
   ```bash
   # Make sure output directory exists
   mkdir -p data/output
   ```

6. **Update Documentation**
   - Ensure README.md reflects the cleaned repository structure

## Final Structure

After cleaning, your repository should look like this:

```
code-switch-thesis/
├── data/
│   ├── extracted_prompts/       # Hindi and English prompt data
│   ├── RTP-LX/                  # RTP-LX datasets
│   └── output/                  # Analysis outputs and results
├── ezswitch/                    # ezswitch library
├── scripts/
│   ├── 01_extract_rtp_data.sh   # Extract data from RTP-LX
│   ├── 02_language_detection.sh # Detect language in prompts
│   ├── 03_filter_code_switch.sh # Filter for code-switched content
│   ├── 04_generate_responses.sh # Generate LLM responses
│   ├── 05_analyze_toxicity.sh   # Analyze toxicity of prompts/responses
│   ├── 06_visualization.sh      # Generate visualizations
│   └── test_primary_key_hindi_only.sh # Test script for Hindi prompts
├── src/
│   ├── add_primary_key.py       # Add unique identifiers to prompts
│   ├── analyze_toxicity_with_id.py    # Toxicity analysis with ID tracking
│   ├── compare_toxicity_with_id.py    # Compare prompt and response toxicity
│   ├── config.py                # Configuration settings
│   ├── filter_language_mix_with_id.py # Filter for code-switched sentences
│   ├── generate_model_responses_with_id.py # Generate model responses
│   └── language_detection_with_id.py  # Detect language composition
├── lid.176.bin                  # FastText language identification model
├── .gitignore                   # Git ignore file
├── README.md                    # Project documentation
└── CLEAN_REPO_GUIDE.md          # This guide
```

This structure ensures you maintain all essential functionality while eliminating redundant or outdated code. 