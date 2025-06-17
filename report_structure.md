# Code-Switched Text Toxicity Analysis: Research Report Structure

## Project Overview

### Research Objective
This research investigates the intersection of **code-switching** (mixing languages within a single text) and **toxicity detection** in the context of English-Hindi language pairs. The study aims to understand how large language models (LLMs) handle code-switched content and whether toxicity patterns differ across monolingual versus code-switched text.

### Key Research Questions
1. How do different LLM models generate continuations for code-switched text compared to monolingual text?
2. What is the relationship between language mixing patterns and toxicity scores?
3. Do models assign higher perplexity to code-switched data compared to individual languages?
4. How do perplexity and toxicity correlate across different models and text types?
5. What are the implications for content moderation systems in multilingual contexts?

---

## Current Dataset Overview

### Dataset: `perspective_analysis.csv`
The dataset contains **95 columns** with the following key components:

#### Core Text Data
- **src**: English source sentences
- **tgt**: Hindi target sentences  
- **generated**: Code-switched sentences generated using the EZSwitch framework
- **method**: Generation method (baseline/silver/gold)
- **model**: Model used for generation (llama_3_8B, llama_3_1_8B, aya_23_8B)
- **direction**: Translation direction
- **primary_key**: Unique identifier

#### Language Statistics (for generated text)
- **hindi_word_count, english_word_count**: Language-specific word counts
- **romanized_hindi_count, total_hindi_count**: Different script representations
- **total_words**: Total word count
- **hindi_percent, english_percent**: Language distribution percentages
- **romanized_hindi_percent, total_hindi_percent**: Script distribution

#### Model Continuations
- **{model}_src_continuation**: Continuations when given source text
- **{model}_tgt_continuation**: Continuations when given target text  
- **{model}_generated_continuation**: Continuations when given code-switched text
- Models: llama3, llama31, aya

#### Toxicity Scores (Perspective API)
For each text type (src, tgt, generated, and all continuations):
- **toxicity**: General toxicity score
- **severe_toxicity**: Severe toxicity detection
- **identity_attack**: Identity-based attacks
- **insult**: Insult detection
- **profanity**: Profanity detection
- **threat**: Threat detection

---

## Data Processing Pipeline

### Step 0: Extract Prompts (`0_extract_prompts.sh`)
```bash
# Extracts prompts from RTP-LX dataset
python extract_prompts.py RTP_LX_HI.json train_hi.txt
python extract_prompts.py RTP_LX_EN.json train_en.txt
```

### Step 1: Translation (`1_translate_file.sh`)
```bash
# Bidirectional translation using Google Translate API
python translate_file.py --input train_en.txt --target hi --output train_hi.txt
python translate_file.py --input train_hi.txt --target en --output train_en.txt
```

### Step 2: Code-Switch Generation (`2_run_ezswitch_hi.sh`)
**Key Components:**
1. **Alignment Generation**: Creates word alignments using MGIZA++
   ```bash
   python giza.py --source train_en.txt --target train_hi.txt --alignments en-hi_align_gold.txt
   ```

2. **EZSwitch Generation**: Generates code-switched text using three models
   - Aya-23-8B
   - Llama-3.1-8B-Instruct  
   - Llama-3-8B-Instruct

3. **Methods**:
   - **Gold**: Uses human-translated alignments
   - **Silver**: Uses machine-translated alignments
   - **Baseline**: Standard generation

### Step 3: Post-Processing (`3_processing_ezswitch_generation.sh`)
1. **Add Primary Keys**: Assigns unique identifiers
2. **Language Detection**: Analyzes language composition using IndicLID
3. **Language Mix Filtering**: Filters appropriate code-switching ratios

### Step 4: Generate Continuations (`4_generate_continuations.sh`)
- Generates continuations for src, tgt, and generated text using three models
- Batch processing with configurable parameters
- Temperature: 0.7, Max tokens: 50

### Step 5: Toxicity Analysis (`5_run_perspective_api.sh`)
- Runs Perspective API on all text variants
- Analyzes 6 toxicity dimensions across all text types
- Handles rate limiting and batch processing

### Step 6: Perplexity Calculation (`6_get_perplexity.sh`)
- Calculates perplexity scores for generated text
- Uses multiple model backends for comparison

---

## Future Research Directions

### 1. Grammatical Properties and Perplexity Analysis

**Objective**: Analyze the relationship between grammatical properties and perplexity using external models.

**Implementation Steps**:
1. **Perplexity Calculation with mT5-XL**:
   ```python
   # Use mT5-XL for cross-lingual perplexity assessment
   from transformers import T5ForConditionalGeneration, T5Tokenizer
   
   model = T5ForConditionalGeneration.from_pretrained("google/mt5-xl")
   tokenizer = T5Tokenizer.from_pretrained("google/mt5-xl")
   
   # Calculate perplexity for each text type
   perplexities = calculate_perplexity(texts, model, tokenizer)
   ```

2. **Language Token Analysis**:
   - Identify which tokens belong to which language
   - Calculate language-specific perplexity scores
   - Analyze grammatical structure preservation

3. **Correlation Analysis**:
   ```python
   # Correlation between language mix and perplexity
   correlation_matrix = df[['hindi_percent', 'english_percent', 'perplexity']].corr()
   ```

**Expected Insights**: Understanding whether grammatical coherence affects model confidence and toxicity perception.

### 2. Multi-dimensional Correlation Analysis

**Objective**: Establish relationships between language distribution, perplexity, and toxicity.

**Analysis Framework**:
```python
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Key variables for correlation
variables = [
    'hindi_percent', 'english_percent', 'romanized_hindi_percent',
    'src_toxicity', 'tgt_toxicity', 'generated_toxicity',
    'perplexity_mt5', 'perplexity_llama3', 'perplexity_aya'
]

# Create correlation heatmap
correlation_matrix = df[variables].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
```

**Research Questions**:
- Does higher language mixing correlate with higher toxicity scores?
- Are there threshold effects for language mixing ratios?
- How does script mixing (Devanagari vs. Roman) affect perception?

### 3. Model-Specific Perplexity Analysis

**Objective**: Compare how different models assign perplexity to code-switched vs. monolingual content.

**Implementation**:
1. **Baseline Comparison**: Use HInge dataset for monolingual Hindi
2. **Cross-Model Analysis**: Compare perplexity across:
   - Llama-3-8B
   - Llama-3.1-8B  
   - Aya-23-8B
   - mT5-XL (external reference)

3. **Statistical Testing**:
   ```python
   # T-test comparing perplexity distributions
   from scipy.stats import ttest_ind
   
   mono_perplexity = hinge_dataset['perplexity']
   cs_perplexity = our_dataset[our_dataset['method'] == 'generated']['perplexity']
   
   t_stat, p_value = ttest_ind(mono_perplexity, cs_perplexity)
   ```

**Expected Outcomes**: Quantify model bias toward monolingual vs. code-switched content.

### 4. HInge Dataset Comparison

**Objective**: Determine if models show systematic bias against code-switched content.

**Methodology**:
1. **Data Preparation**:
   - Extract monolingual Hindi sentences from HInge
   - Ensure comparable content domains and sentence lengths
   - Calculate perplexity using same models

2. **Comparative Analysis**:
   ```python
   # Compare perplexity distributions
   monolingual_stats = calculate_perplexity_stats(hinge_data)
   codeswitched_stats = calculate_perplexity_stats(our_data)
   
   # Effect size calculation
   cohen_d = (cs_mean - mono_mean) / pooled_std
   ```

3. **Domain Control**: Match content types (formal vs. informal, news vs. social media)

### 5. Human Evaluation Study

**Objective**: Validate automated toxicity detection against human judgment for code-switched content.

**Google Form Design**:
```
Section 1: Participant Background
- Language proficiency (English/Hindi)
- Familiarity with code-switching
- Demographics (age, education)

Section 2: Toxicity Rating (1-7 Likert scale)
- Present 50-100 carefully selected sentences
- Balance across: monolingual EN, monolingual HI, code-switched
- Include Perspective API scores (hidden from participants)

Section 3: Naturalness Rating
- Rate how natural/fluent the code-switching sounds
- Identify awkward or unnatural switches

Section 4: Open Feedback
- Comments on cultural context affecting toxicity perception
```

**Analysis Plan**:
```python
# Inter-rater reliability
from scipy.stats import pearsonr
import krippendorff

# Correlation with Perspective API
human_perspective_corr = pearsonr(human_ratings, perspective_scores)

# Naturalness vs. Perplexity
naturalness_perplexity_corr = pearsonr(naturalness_ratings, perplexity_scores)

# Cultural bias detection
cultural_bias_analysis = analyze_rating_differences_by_background(ratings)
```

---

## Additional Research Opportunities

### 1. Temporal Toxicity Evolution
- **Research Question**: How does toxicity in continuations evolve over longer generation sequences?
- **Method**: Generate longer continuations (100-200 tokens) and analyze toxicity progression
- **Applications**: Understanding toxicity amplification in conversational AI

### 2. Cross-Lingual Toxicity Transfer
- **Research Question**: Do toxic patterns transfer differently across language boundaries in code-switched text?
- **Method**: Compare toxicity scores when the same semantic content is expressed in different language mixing ratios
- **Impact**: Inform cross-lingual content moderation systems

### 3. Semantic Preservation Analysis
- **Research Question**: How well do code-switched generations preserve semantic meaning?
- **Method**: Use semantic similarity models (SBERT, multilingual embeddings) to compare semantic drift
- **Applications**: Quality assessment for code-switched text generation

### 4. Sociolinguistic Factors
- **Research Question**: How do sociolinguistic factors (formality, social context) affect toxicity perception in code-switched text?
- **Method**: Categorize text by formality levels and analyze toxicity patterns
- **Impact**: Context-aware toxicity detection systems

### 5. Model Architecture Impact
- **Research Question**: How do different architectural choices (encoder-decoder vs. decoder-only) affect code-switched text handling?
- **Method**: Compare models like mT5, mBART, and decoder-only models on same tasks
- **Insights**: Guide model selection for multilingual applications

### 6. Regional Language Variations
- **Research Question**: How do regional Hindi variations affect toxicity detection and model perplexity?
- **Method**: Collect data from different Hindi-speaking regions and analyze patterns
- **Impact**: Regional bias detection in NLP systems

### 7. Cultural Context in Toxicity
- **Research Question**: How do cultural nuances affect toxicity perception in code-switched contexts?
- **Method**: Analysis of culturally-specific terms and their toxicity ratings
- **Applications**: Culturally-aware content moderation

### 8. Prompt Engineering for Code-Switched Generation
- **Research Question**: How can prompts be designed to generate more natural and less toxic code-switched content?
- **Method**: Systematic prompt variation and evaluation
- **Impact**: Better code-switched text generation systems

---

## Technical Implementation Recommendations

### Infrastructure Setup
```bash
# Environment setup for Snellius cluster
module load 2023 Miniconda3/23.5.2-0
conda create -n cs-research python=3.9
conda activate cs-research

# Key dependencies
pip install transformers torch datasets perspective-api-client
pip install pandas numpy scipy matplotlib seaborn
pip install sentence-transformers scipy krippendorff
```

### Data Management
```python
# Recommended data structure
class CodeSwitchDataset:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.models = ['llama3', 'llama31', 'aya']
        self.text_types = ['src', 'tgt', 'generated']
        self.toxicity_dims = ['toxicity', 'severe_toxicity', 'identity_attack', 
                             'insult', 'profanity', 'threat']
    
    def get_continuation_data(self, model, text_type):
        """Extract continuation data for specific model and text type"""
        col_name = f"{model}_{text_type}_continuation"
        return self.data[col_name].dropna()
    
    def calculate_correlation_matrix(self, variables):
        """Calculate correlation matrix for specified variables"""
        return self.data[variables].corr()
```

### Statistical Analysis Pipeline
```python
# Statistical testing framework
def perform_comprehensive_analysis(dataset):
    results = {}
    
    # 1. Correlation analysis
    results['correlations'] = analyze_correlations(dataset)
    
    # 2. Perplexity comparison
    results['perplexity_comparison'] = compare_perplexity_distributions(dataset)
    
    # 3. Toxicity analysis
    results['toxicity_patterns'] = analyze_toxicity_patterns(dataset)
    
    # 4. Statistical significance testing
    results['significance_tests'] = perform_significance_tests(dataset)
    
    return results
```

---

## Making Research More Impactful

### Novel Contributions You Can Make:
1. **First Code-Switched Toxicity Dataset** with multi-model analysis
2. **Cross-lingual Bias Detection** in toxicity assessment
3. **Perplexity-Toxicity Relationship** in multilingual contexts
4. **Human-AI Agreement** in multilingual toxicity detection
5. **Sociolinguistic Factors** in automated content moderation

### Publication Opportunities:
- **ACL/EMNLP**: Main NLP conferences
- **COLING**: Computational linguistics focus
- **LREC-COLING**: Language resources and evaluation
- **AACL**: Asia-Pacific computational linguistics
- **Workshops**: NLP for social good, multilingual NLP, AI safety

### Expected Timeline and Milestones

#### Phase 1: Perplexity Analysis (2-3 weeks)
- [ ] Implement mT5-XL perplexity calculation
- [ ] Compare with existing models
- [ ] Generate HInge dataset comparisons
- [ ] Statistical analysis and visualization

#### Phase 2: Correlation Studies (2-3 weeks)
- [ ] Multi-dimensional correlation analysis
- [ ] Language mixing threshold identification
- [ ] Model-specific bias quantification
- [ ] Publication-ready visualizations

#### Phase 3: Human Evaluation (3-4 weeks)
- [ ] Design Google Form survey
- [ ] Recruit participants (target: 50-100 responses)
- [ ] Data collection and analysis
- [ ] Validation of automated metrics

#### Phase 4: Extended Analysis (2-3 weeks)
- [ ] Implement additional research directions
- [ ] Cross-validation with external datasets
- [ ] Prepare research findings
- [ ] Documentation and reproducibility

---

## Research Impact and Applications

### Academic Contributions
1. **Novel Dataset**: First comprehensive code-switched toxicity dataset with multi-model analysis
2. **Methodological Framework**: Systematic approach to analyzing code-switched content
3. **Cross-lingual Insights**: Understanding of model biases in multilingual contexts
4. **Evaluation Metrics**: Human-validated toxicity assessment for code-switched text

### Practical Applications
1. **Content Moderation**: Improved detection systems for multilingual social media
2. **Conversational AI**: Better safety measures for multilingual chatbots
3. **Educational Technology**: Safe multilingual learning environments
4. **Social Media Analysis**: Understanding toxicity patterns in multilingual communities

### Industry Relevance
- **Social Media Platforms**: Enhanced content moderation for global users
- **AI Safety**: Reducing harmful outputs in multilingual AI systems  
- **Localization**: Better understanding of cultural context in toxicity
- **Policy Making**: Evidence-based guidelines for multilingual AI governance

---

## Conclusion

This research addresses a critical gap in understanding how language mixing affects toxicity detection and model behavior. The comprehensive approach combining automated analysis with human validation provides both theoretical insights and practical applications for building safer multilingual AI systems.

The systematic methodology developed here can serve as a template for similar studies in other language pairs, contributing to the broader goal of inclusive and safe AI systems for global multilingual communities. 

## **Immediate Action Items (This Week)**

1. **Start with Point 2**: Run correlation analysis on existing data
2. **Set up Point 1**: Install mT5-XL and calculate perplexity
3. **Design Point 5**: Create Google Form for human evaluation
4. **Plan Point 4**: Download and prepare HInge dataset
5. **Prepare Point 3**: Set up infrastructure for model perplexity calculation

### **Point 1: Grammatical Properties Analysis**
**What to do**: Calculate perplexity using mT5-XL and analyze language token distribution.

**Steps**:
1. Install mT5-XL: `pip install transformers[torch]`
2. Create script `calculate_mt5_perplexity.py`:
   ```python
   from transformers import T5ForConditionalGeneration, T5Tokenizer
   import torch
   import pandas as pd
   
   def calculate_perplexity_mt5(texts):
       model = T5ForConditionalGeneration.from_pretrained("google/mt5-xl")
       tokenizer = T5Tokenizer.from_pretrained("google/mt5-xl")
       
       perplexities = []
       for text in texts:
           inputs = tokenizer(text, return_tensors="pt", truncation=True)
           with torch.no_grad():
               outputs = model(**inputs, labels=inputs["input_ids"])
               loss = outputs.loss
               perplexity = torch.exp(loss)
           perplexities.append(perplexity.item())
       
       return perplexities
   ```
3. Run on your dataset and add results as new columns
4. Analyze correlation with language percentages

### **Point 2: Language-Perplexity-Toxicity Correlation**
**What to do**: Create comprehensive correlation analysis.

**Steps**:
1. Create script `correlation_analysis.py`:
   ```python
   import pandas as pd
   import seaborn as sns
   import matplotlib.pyplot as plt
   from scipy.stats import pearsonr
   
   df = pd.read_csv('final_outputs/perspective_analysis.csv')
   
   # Define correlation variables
   correlation_vars = [
       'hindi_percent', 'english_percent', 'romanized_hindi_percent',
       'src_toxicity', 'tgt_toxicity', 'generated_toxicity',
       # Add perplexity columns when available
   ]
   
   # Calculate correlations
   corr_matrix = df[correlation_vars].corr()
   
   # Create heatmap
   plt.figure(figsize=(12, 10))
   sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
   plt.title('Language Mixing vs Toxicity Correlation Matrix')
   plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
   ```

### **Point 3: Model Perplexity vs Toxicity Analysis**
**What to do**: Calculate perplexity for each model and correlate with toxicity.

**Steps**:
1. Create `model_perplexity_analysis.py`:
   ```python
   # Calculate perplexity for each model on their own continuations
   models = ['llama3', 'llama31', 'aya']
   
   for model in models:
       # Load the actual model
       # Calculate perplexity on continuations
       # Correlate with toxicity scores
   ```
2. Compare perplexity patterns across:
   - Source continuations
   - Target continuations  
   - Generated continuations
3. Statistical significance testing

### **Point 4: HInge Dataset Comparison**
**What to do**: Download HInge dataset and compare perplexity distributions.

**Steps**:
1. Download HInge dataset
2. Create `hinge_comparison.py`:
   ```python
   # Load HInge data
   # Calculate perplexity on monolingual Hindi
   # Compare with your code-switched data
   # Statistical testing (t-test, Mann-Whitney U)
   ```
3. Generate comparative visualizations

### **Point 5: Human Evaluation**
**What to do**: Design Google Form and collect human judgments.

**Steps**:
1. **Sample Selection**: Choose 50-100 diverse examples:
   ```python
   # Stratified sampling
   sample_data = df.groupby(['model', 'method']).apply(
       lambda x: x.sample(min(10, len(x)))
   ).reset_index(drop=True)
   ```

2. **Google Form Structure** (Your Current Design):
   - **Background Section**: Language proficiency, demographics, code-switching familiarity
   - **Question 1**: "How natural does the code-switched text sound?" [1-5 scale: Not natural → Natural]
   - **Question 2**: "How toxic is this code-switched text?" [1-5 scale: Not toxic → Toxic]  
   - **Question 3**: "Compared to the original English version (src column), is this code-switched version:" [Less toxic / Similarly toxic / More toxic / Cannot be determined]
   - **Question 4**: "Which words/phrases are the most toxic? (Please specify if they are in Hindi, English, or both)" [Short text answer]

3. **Why This Design Is Excellent**:
   - **Naturalness-Perplexity Link**: Q1 directly tests whether human naturalness judgments correlate with model perplexity scores
   - **Perspective API Validation**: Q2 provides direct comparison with automated toxicity detection
   - **Content-Controlled Comparison**: Q3 brilliantly controls for semantic content while testing linguistic form effects
   - **Language-Specific Analysis**: Q4 enables fine-grained analysis of which language carries toxicity burden
   - **Multiple Research Questions**: Single form addresses naturalness, toxicity validation, comparative assessment, and language-specific effects
   - **Efficiency**: Compact design reduces participant fatigue while maximizing data quality
   - **Cross-Linguistic Insights**: Directly addresses whether code-switching amplifies, reduces, or maintains toxicity

4. **Enhanced Analysis Framework**:
   ```python
   # Correlation with Perspective API
   human_perspective_corr = pearsonr(human_toxicity_ratings, perspective_scores)
   
   # Naturalness vs. Perplexity correlation
   naturalness_perplexity_corr = pearsonr(naturalness_ratings, perplexity_scores)
   
   # Comparative toxicity analysis
   toxicity_shift_analysis = analyze_comparative_ratings(comparative_responses)
   
   # Language-specific toxicity attribution
   toxic_word_analysis = analyze_language_specific_toxicity(toxic_word_responses)
   
   # Cross-linguistic toxicity patterns
   def analyze_toxicity_patterns():
       # Analyze if toxicity is carried by Hindi vs English words
       hindi_toxic = count_language_specific_toxicity('Hindi')
       english_toxic = count_language_specific_toxicity('English')
       mixed_toxic = count_language_specific_toxicity('Both')
       
       return {
           'hindi_dominance': hindi_toxic / total_responses,
           'english_dominance': english_toxic / total_responses,
           'mixed_toxicity': mixed_toxic / total_responses
       }
   
   # Inter-rater reliability
   krippendorff_alpha = calculate_reliability(ratings_matrix)
   ```

5. **Research Publications Enabled by This Design**:
   - **ACL/EMNLP**: "Human vs. Machine Perception of Toxicity in Code-Switched Text"
   - **COLING**: "Language-Specific Toxicity Attribution in Multilingual Content"
   - **AI Safety Workshops**: "Validating Automated Toxicity Detection Across Languages" 