# Model Fine-tuning and Feature Attribution Analysis

This document provides a comprehensive analysis of our approach to fine-tuning toxicity classifiers and analyzing feature attribution across different languages and models.

## 1. Pipeline Overview

Our pipeline consists of three main stages:

1. **Fine-tuning toxicity classifiers** for different language models (10_finetune_classifier.sh)
2. **Feature attribution analysis** to understand which tokens contribute to toxicity predictions (11_feature_attribution.sh)
3. **Comparative analysis** of feature attribution across languages (12_compare_attribution.sh)

## 2. Fine-tuning Classifiers (10_finetune_classifier.sh)

### Purpose

This script fine-tunes a multilingual BERT-based model to detect toxicity in generated content from various LLMs (Aya, Llama 3, Llama 3.1).

### Key Components

- **Base Model**: We use `google/muril-base-cased`, a multilingual model particularly effective for Hindi-English code-switched text.
- **Training Data**: Outputs from different LLMs with perspective API toxicity scores as labels.
- **LLMs Analyzed**: Aya, Llama 3, Llama 3.1

### Hyperparameters

- **Batch Size**: 16
- **Training Epochs**: 5
- **Max Sequence Length**: 128
- **Test Set Size**: 20% of data
- **Maximum Samples**: 2000 (for faster experimentation)

The script creates separate fine-tuned models for each LLM's outputs, allowing for model-specific toxicity detection.

## 3. Feature Attribution Analysis (11_feature_attribution.sh)

### Purpose

This script applies feature attribution techniques to understand which tokens contribute most to toxicity predictions across different language conditions.

### Key Components

- **Input Sources**: Code-switched text, source language text, and target language text
- **Attribution Method**: Simple attribution (attribution of importance to each token). This method typically calculates the element-wise product of the input token embeddings and the gradient of the output (toxicity score) with respect to those embeddings (Input * Gradient). This provides a basic measure of how much each input token contributes to the final prediction.
- **Sample Size**: Limited to 100 samples for quicker analysis

### Process

For each LLM (Aya, Llama 3, Llama 3.1), the script:
1. Loads the fine-tuned toxicity classifier
2. Processes inputs from three language conditions:
   - Code-switched text
   - Source language text
   - Target language text
3. Generates attribution scores for tokens in each text
4. Saves results for further analysis

### 3.1 Individual Analysis Plots

Within each model's feature attribution directory (e.g., `new_outputs/feature_attribution/aya/code_switched/`), the `feature_attribution.py` script generates several plots:

- **`prediction_distribution.png`**: This plot shows a histogram or density plot of the toxicity classifier's predicted scores for the analyzed samples. It helps visualize the overall distribution of predicted toxicity levels (e.g., are most samples predicted as highly toxic, non-toxic, or somewhere in between?).

- **`top_tokens.png`**: This plot displays the tokens with the *highest* positive attribution scores. These are the words or subwords that the model identifies as contributing most strongly *towards* a toxicity prediction. Examining these tokens can reveal specific terms or patterns the model associates with toxicity.

- **`bottom_tokens.png`**: This plot displays the tokens with the *lowest* attribution scores (often negative or close to zero). These tokens contribute least to toxicity predictions, or potentially contribute towards a *non-toxic* prediction. Analyzing these can help understand what the model considers neutral or non-toxic language.

These plots provide a snapshot of the model's behavior and the key influential tokens for a specific LLM and language condition before the comparative analysis across conditions is performed.

## 4. Comparative Analysis (12_compare_attribution.sh)

### Purpose

This script compares feature attribution results across language conditions to understand how toxicity manifests differently in monolingual vs. code-switched text.

### Key Components

- **Analysis Method**: `simple_attribution` (now the main method used)
- **Visualization**: Distribution plots and mean comparison plots
- **Output Format**: HTML reports with embedded visualizations

### Process

The script:
1. Loads attribution results for each LLM and language condition
2. Analyzes token importance by language (Hindi vs. English)
3. Generates comparative visualizations
4. Creates HTML reports summarizing findings

## 5. Python Implementation (compare_attribution_languages.py)

### Key Functions

- **is_hindi()**: Identifies Hindi tokens using Unicode character ranges
- **analyze_token_importance()**: Categorizes tokens by language and calculates attribution statistics
- **compare_language_importance()**: Compares attribution patterns across languages
- **create_comparison_plots()**: Generates visualizations of attribution patterns
- **extract_top_tokens()**: Identifies the most influential tokens for toxicity predictions

### Analysis Methods

The script employs several techniques to analyze language-specific patterns:
- Separation of tokens by language
- Statistical analysis of attribution scores
- Visualization of score distributions
- Identification of top influential tokens
- Analysis of token overlap between datasets

## 6. Token Importance Analysis

The feature attribution analysis produces two key visualizations:

### Mean Attribution Comparison

The `simple_attribution_mean_comparison.png` shows the average attribution scores for Hindi and English tokens across three conditions:
- Code-switched text
- Source language (presumably Hindi)
- Target language (presumably English)

This visualization helps identify whether one language tends to receive higher attribution scores for toxicity than the other, and whether this pattern changes in code-switched contexts.

### Distribution Analysis

The `simple_attribution_distribution_violin.png` shows the distribution of attribution scores across languages and conditions. This provides deeper insights than simple averages, revealing:
- The range of attribution scores
- The concentration of scores (where most tokens fall)
- Outliers (particularly toxic or non-toxic tokens)
- Differences in attribution patterns between languages

## 6.1 Interpreting Attribution Scores

### Understanding Mean Attribution Scores

The mean attribution score represents the average importance or influence that tokens of a particular language have on the model's toxicity prediction. Specifically:

- **Higher scores** indicate tokens that more strongly contribute to the model predicting toxicity
- **Lower scores** indicate tokens with less influence on toxicity predictions or that may contribute to non-toxic predictions

### Language Distribution Across Conditions

In the attribution visualizations, you may notice different patterns across the three language conditions:

1. **Code-switched text**: Contains both Hindi and English attribution scores because these texts naturally mix both languages. The relative heights of Hindi vs. English bars indicate which language's tokens tend to contribute more to toxicity predictions in mixed-language contexts.

2. **Source language text**: Ideally should contain primarily one language (Hindi in a Hindi-English code-switching study). If only one language bar appears, it means tokens of only that language were detected or had significant attribution. If both languages appear, it suggests:
   - The source corpus may contain some mixed-language content
   - Some tokens might be incorrectly classified as Hindi/English
   - The boundary between languages isn't always clear-cut (e.g., borrowed words)

3. **Target language text**: Similarly, should primarily contain the other language (English in a Hindi-English study). The presence of both language bars suggests similar possibilities as with source language.

### How to Interpret Results

When analyzing the attribution visualizations:

1. **Compare means within a condition**: Are Hindi tokens given higher attribution than English tokens in code-switched text? This suggests the model associates one language more strongly with toxicity.

2. **Compare across conditions**: Does the attribution for Hindi tokens differ between code-switched text and monolingual Hindi text? This might reveal how language context affects toxicity attribution.

3. **Examine distributions**: The violin plots show not just means but the full distribution of scores. Look for:
   - Differences in score ranges between languages
   - Bimodal distributions (suggesting some tokens are strongly toxic while others are neutral)
   - Outliers that might represent particularly influential tokens

4. **Consider token frequency**: A language might show higher mean attribution simply because toxic terms appear more frequently in that language in your dataset.

By analyzing these patterns, we can understand whether toxicity manifests differently in monolingual versus code-switched contexts, and whether the model exhibits any biases in how it attributes toxicity across languages.

## 7. Feature Attribution Methods

While our current implementation focuses on simple attribution, the field offers several sophisticated methods for model interpretability:

### Integrated Gradients

Integrated Gradients is a gradient-based attribution method that assigns importance scores to input features by considering the gradients of the output with respect to the input along a straight-line path from a baseline to the input.

**Importance**: This method satisfies desirable axioms like completeness and sensitivity, providing theoretically sound attribution scores that sum to the difference between the model output and a baseline prediction.

### Layer Integrated Gradients

Layer Integrated Gradients extends the integrated gradients approach to internal layers of neural networks, allowing attribution of importance to intermediate representations rather than just input features.

**Importance**: This method provides insights into how internal network components contribute to predictions, offering a more detailed view of model decision-making.

### Occlusion

Occlusion is a perturbation-based method that measures feature importance by observing how the model's prediction changes when certain input features are masked or "occluded."

**Importance**: This method provides a straightforward and intuitive measure of feature importance based on the direct impact of removing information, making it particularly useful for understanding which tokens most strongly influence toxicity predictions.

## 8. Conclusion

The pipeline implemented through these scripts provides a comprehensive framework for:
1. Fine-tuning toxicity classifiers for different language models
2. Analyzing which tokens contribute most to toxicity predictions
3. Comparing attribution patterns across languages and contexts

This approach enables a deeper understanding of how toxicity manifests in multilingual and code-switched texts, with potential implications for improving content moderation in multilingual settings. 