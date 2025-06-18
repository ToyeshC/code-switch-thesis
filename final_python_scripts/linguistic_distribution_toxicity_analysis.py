#!/usr/bin/env python3
"""
Linguistic Distribution of Toxicity and Feature Attribution Analysis

This script addresses Research Question 5: "How are toxicity predictors distributed 
linguistically in bilingual text?" by analyzing:
1. Distribution of toxicity scores across Hindi and English segments
2. Feature attribution using Captum to identify which tokens contribute most to toxicity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import fasttext
import re
import argparse
import os
import sys
from datetime import datetime
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import logging

# Add the current working directory to Python path to find config.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

try:
    import config
except ImportError:
    print("Warning: Could not import config.py - will proceed without HuggingFace authentication")
    config = None
import json

# Language identification imports
import nltk
from nltk.corpus import words as nltk_words

# Feature attribution imports
try:
    import torch
    import torch.nn as nn
    from torch.nn.parallel import DataParallel
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from captum.attr import IntegratedGradients, DeepLift, GradientShap
    import torch.multiprocessing as mp
    from concurrent.futures import ThreadPoolExecutor, as_completed
    CAPTUM_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch, Transformers, or Captum not available. Feature attribution disabled.")
    CAPTUM_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set matplotlib backend to avoid display issues on cluster
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties

# Configure matplotlib for Hindi text rendering
def setup_matplotlib_fonts():
    """Configure matplotlib to handle Hindi/Devanagari text"""
    try:
        # Try to download and use Noto Sans Devanagari font
        import urllib.request
        import tempfile
        
        # Create temp directory for font
        temp_dir = tempfile.mkdtemp()
        font_path = os.path.join(temp_dir, 'NotoSansDevanagari-Regular.ttf')
        
        # Download Noto Sans Devanagari font if not available
        font_url = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf"
        
        try:
            print("Downloading Noto Sans Devanagari font...")
            urllib.request.urlretrieve(font_url, font_path)
            
            # Register the font with matplotlib
            fm.fontManager.addfont(font_path)
            
            # Create font properties
            hindi_font_prop = FontProperties(fname=font_path, size=12)
            
            print(f"Successfully downloaded and configured Noto Sans Devanagari font")
            return hindi_font_prop, font_path
            
        except Exception as e:
            print(f"Could not download font: {e}")
            print("Falling back to system fonts...")
            
    except ImportError:
        print("urllib not available for font download")
    
    # Fallback to system fonts
    hindi_fonts = [
        'Noto Sans Devanagari',
        'Mangal', 
        'Devanagari Sangam MN',
        'Mukti Narrow',
        'Gargi',
        'Sahadeva',
        'Chandas',
        'Kokila',
        'Utsaah',
        'Aparajita'
    ]
    
    # Get all available system fonts
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # Find the first available Hindi font
    hindi_font = None
    for font in hindi_fonts:
        if font in available_fonts:
            hindi_font = font
            break
    
    if hindi_font:
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = [hindi_font] + plt.rcParams['font.sans-serif']
        hindi_font_prop = FontProperties(family=hindi_font, size=12)
        print(f"Using system font '{hindi_font}' for Hindi text rendering")
    else:
        # Final fallback: Use DejaVu Sans with Unicode support
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        hindi_font_prop = FontProperties(family='DejaVu Sans', size=12)
        print("Using DejaVu Sans fallback font for Hindi text rendering")
        print("Note: For better Hindi text rendering, consider installing 'Noto Sans Devanagari' font")
        print("On Ubuntu/Debian: sudo apt-get install fonts-noto-devanagari")
        print("On CentOS/RHEL: sudo yum install google-noto-sans-devanagari-fonts")
    
    # Ensure matplotlib uses Unicode
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    
    return hindi_font_prop, None

# Setup fonts for Hindi text
HINDI_FONT_PROP, FONT_PATH = setup_matplotlib_fonts()

def safe_text_render(text_list, max_length=15):
    """
    Safely render text for matplotlib, handling Unicode issues
    """
    safe_texts = []
    for text in text_list:
        if isinstance(text, str):
            try:
                # Truncate long text for better visualization
                if len(text) > max_length:
                    text = text[:max_length] + '...'
                
                # Test if the text can be properly encoded/decoded
                test = text.encode('utf-8').decode('utf-8')
                safe_texts.append(text)
            except (UnicodeDecodeError, UnicodeEncodeError):
                # Fallback for problematic characters
                safe_texts.append(repr(text).strip("'\""))
        else:
            safe_texts.append(str(text))
    return safe_texts

def detect_script_type(text):
    """
    Detect if text contains Hindi/Devanagari characters
    """
    if not isinstance(text, str):
        return 'english'
    
    # Count Devanagari characters (Unicode range: U+0900-U+097F)
    devanagari_count = sum(1 for char in text if '\u0900' <= char <= '\u097F')
    total_chars = len([char for char in text if char.isalpha()])
    
    if total_chars == 0:
        return 'english'
    
    # If more than 10% of alphabetic characters are Devanagari, consider it Hindi
    if devanagari_count / total_chars > 0.1:
        return 'hindi'
    else:
        return 'english'

def apply_smart_font_to_plot(ax, title=None, xlabel=None, ylabel=None):
    """
    Apply appropriate font properties based on script detection
    """
    global HINDI_FONT_PROP
    
    try:
        # Apply font to tick labels with script detection
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            text = label.get_text()
            script_type = detect_script_type(text)
            
            if script_type == 'hindi':
                label.set_fontproperties(HINDI_FONT_PROP)
            else:
                # Use default system font for English
                label.set_fontfamily('DejaVu Sans')
            label.set_fontsize(10)
        
        # Apply font to title
        if title:
            script_type = detect_script_type(title)
            if script_type == 'hindi':
                ax.set_title(title, fontproperties=HINDI_FONT_PROP, fontsize=14, fontweight='bold')
            else:
                ax.set_title(title, fontfamily='DejaVu Sans', fontsize=14, fontweight='bold')
        
        # Apply font to axis labels
        if xlabel:
            script_type = detect_script_type(xlabel)
            if script_type == 'hindi':
                ax.set_xlabel(xlabel, fontproperties=HINDI_FONT_PROP, fontsize=12)
            else:
                ax.set_xlabel(xlabel, fontfamily='DejaVu Sans', fontsize=12)
                
        if ylabel:
            script_type = detect_script_type(ylabel)
            if script_type == 'hindi':
                ax.set_ylabel(ylabel, fontproperties=HINDI_FONT_PROP, fontsize=12)
            else:
                ax.set_ylabel(ylabel, fontfamily='DejaVu Sans', fontsize=12)
            
    except Exception as e:
        print(f"Warning: Could not apply smart font properties: {e}")
        # Fallback to default matplotlib rendering
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)

# Legacy function name for backward compatibility
def apply_hindi_font_to_plot(ax, title=None, xlabel=None, ylabel=None):
    """Legacy wrapper - now uses smart font detection"""
    return apply_smart_font_to_plot(ax, title=title, xlabel=xlabel, ylabel=ylabel)

# Set professional color scheme (matching existing scripts)
plt.style.use('default')
sns.set_style("whitegrid")
professional_colors = ['#2E86AB', '#A8DADC', '#457B9D', '#1D3557', '#A2E4B8', '#52B69A']
sns.set_palette(professional_colors)

class LinguisticToxicityAnalyzer:
    """
    Class to analyze linguistic distribution of toxicity in bilingual text and
    perform feature attribution analysis using Captum
    """
    
    def __init__(self, input_file: str, output_dir: str, fasttext_model: str, multi_gpu: bool = False):
        self.input_file = input_file
        self.output_dir = os.path.join(output_dir, "experiment_e")
        self.fasttext_model = fasttext_model
        self.multi_gpu = multi_gpu
        self.df = None
        self.model = None
        
        # Setup multi-GPU configuration
        if self.multi_gpu and torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            print(f"Multi-GPU mode enabled: {self.num_gpus} GPUs available")
            # Set multiprocessing start method for CUDA compatibility
            if mp.get_start_method(allow_none=True) != 'spawn':
                mp.set_start_method('spawn', force=True)
        else:
            self.num_gpus = 1
            print("Single GPU mode")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Language identification setup
        self.english_words = set()
        self.common_hindi_words = set()
        self.hindi_patterns = None
        
        # Toxicity dimensions to analyze
        self.toxicity_dimensions = [
            'toxicity', 'severe_toxicity', 'identity_attack',
            'insult', 'profanity', 'threat'
        ]
        
        # Models to analyze
        self.models = ['llama3', 'llama31', 'aya']
        
        # Text types to analyze
        self.text_types = {
            'src': 'English Source',
            'tgt': 'Hindi Target', 
            'generated': 'Code-switched Generated'
        }
        
        # Results storage
        self.linguistic_results = {}
        self.attribution_results = {}
        
        # Attribution model components
        self.tokenizer = None
        self.attribution_model = None
        self.integrated_gradients = None
        self.deeplift = None
        self.skip_attribution = False
        
    def load_data(self) -> bool:
        """Load the perspective analysis CSV file"""
        print(f"Loading data from {self.input_file}...")
        try:
            self.df = pd.read_csv(self.input_file)
            print(f"Loaded {len(self.df)} rows with {len(self.df.columns)} columns")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
            
    def setup_language_identification(self):
        """Setup language identification tools and patterns"""
        print("Setting up language identification...")
        
        # Load FastText model
        print(f"Loading FastText model from {self.fasttext_model}...")
        try:
            self.model = fasttext.load_model(self.fasttext_model)
        except Exception as e:
            print(f"Error loading FastText model: {e}")
            return False
            
        # Load NLTK English words
        try:
            self.english_words = set(nltk_words.words())
            print(f"Loaded {len(self.english_words)} English words from NLTK")
        except LookupError:
            print("Downloading NLTK words corpus...")
            nltk.download('words')
            self.english_words = set(nltk_words.words())
            
        # Common Romanized Hindi words for better detection
        self.common_hindi_words = {
            # Common verbs
            'hai', 'hain', 'tha', 'thi', 'the', 'ho', 'hoga', 'hogi', 'honge',
            'karo', 'karenge', 'kiya', 'kiye', 'kar', 'karna', 'karte', 'karti',
            'gaya', 'gayi', 'gaye', 'jao', 'jaana', 'aaya', 'aayi', 'aaye', 'aao', 'aana',
            'raha', 'rahi', 'rahe', 'rehna', 'dekho', 'dekha', 'dekhte', 'sunna', 'suna',
            
            # Pronouns and common words
            'main', 'mein', 'hum', 'tum', 'aap', 'yeh', 'woh', 'kya', 'kyun', 'kaise',
            'kaun', 'kitna', 'kahan', 'kab', 'koi', 'kuch', 'sab', 'sabse',
            
            # Postpositions and particles
            'ka', 'ki', 'ke', 'ko', 'se', 'mein', 'par', 'pe', 'tak', 'liye',
            
            # Conjunctions and adverbs
            'aur', 'ya', 'lekin', 'magar', 'kyunki', 'isliye', 'phir', 'abhi', 'ab',
            'bahut', 'thoda', 'zyada', 'kam', 'bilkul', 'shayad', 'zaroor',
            
            # Negations
            'nahi', 'nahin', 'mat', 'na', 'kabhi',
            
            # Common adjectives
            'accha', 'achha', 'bura', 'theek', 'sahi', 'galat', 'naya', 'purana',
            'bada', 'chota', 'lamba', 'chhota', 'sundar', 'khushi', 'dukh',
            
            # Markers and suffixes
            'wala', 'wali', 'wale', 'waala', 'waali', 'waale'
        }
        
        # Hindi patterns for romanized text detection
        hindi_patterns = [
            r'\b(kya|kyun|kaise|kaun|kitna|kahan)\b',
            r'\b(hai|hain|tha|thi|ho|hoga|hogi)\b',
            r'\b(ka|ki|ke|ko|se|mein|par)\b',
            r'\b(aur|ya|lekin|magar|kyunki|isliye)\b',
            r'\b(nahi|nahin|mat|na)\b',
            r'\b(bahut|thoda|zyada|kam)\b',
            r'\b(main|mein|hum|tum|aap|yeh|woh)\b',
            r'\b(karo|karenge|kiya|kiye|kar|karna)\b',
            r'\b(gaya|gayi|gaye|jao|jaana)\b',
            r'\b(aaya|aayi|aaye|aao|aana)\b',
            r'\b(raha|rahi|rahe|rehna)\b',
            r'\b(wala|wali|wale)\b',
            r'\b(accha|achha|bura|theek)\b',
        ]
        
        self.hindi_patterns = re.compile('|'.join(hindi_patterns), re.IGNORECASE)
        print("Language identification setup complete")
        return True 

    def detect_token_language(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Detect language for each token in a text string
        Returns: (tokens, languages) where languages are 'hindi', 'english', or 'romanized_hindi'
        """
        if not isinstance(text, str) or pd.isna(text):
            return [], []
            
        tokens = text.split()
        languages = []
        
        for token in tokens:
            # Clean token (remove punctuation for language detection)
            clean_token = re.sub(r'[^\w]', '', token).lower()
            if not clean_token:
                languages.append('other')
                continue
                
            # Check for Devanagari script (native Hindi)
            if any('\u0900' <= char <= '\u097F' for char in clean_token):
                languages.append('hindi')
                continue
                
            # Check for common romanized Hindi words
            if clean_token in self.common_hindi_words or self.hindi_patterns.search(clean_token):
                languages.append('romanized_hindi')
                continue
                
            # Use FastText for remaining tokens
            try:
                prediction = self.model.predict(clean_token, k=1)
                lang = prediction[0][0].replace('__label__', '')
                
                if lang == 'hi':
                    languages.append('hindi')
                elif lang == 'en' and (clean_token in self.english_words or len(clean_token) > 3):
                    languages.append('english')
                else:
                    # Heuristic for romanized Hindi vs English
                    if (clean_token.endswith(('na', 'ne', 'ni', 'ta', 'ti', 'te', 'ya', 'ye', 'yi', 
                                            'kar', 'wala', 'wali', 'gaya', 'gayi', 'raha', 'rahi')) or
                        any(pattern in clean_token for pattern in ('aa', 'ee', 'oo', 'kh', 'gh', 'ch', 'jh', 'th'))):
                        languages.append('romanized_hindi')
                    else:
                        languages.append('english')
            except:
                languages.append('english')  # Default to English if detection fails
                
        return tokens, languages
        
    def analyze_linguistic_distribution(self):
        """
        Analyze distribution of toxicity scores across Hindi and English segments
        within code-switched texts
        """
        print("Analyzing linguistic distribution of toxicity...")
        
        results = {
            'segment_analysis': [],
            'aggregate_stats': {},
            'correlation_analysis': {}
        }
        
        # Analyze each text type
        for text_col, text_name in self.text_types.items():
            print(f"  Processing {text_name} texts...")
            
            text_data = self.df[text_col].dropna()
            
            # Get corresponding toxicity scores
            toxicity_cols = {dim: f"{text_col}_{dim}" for dim in self.toxicity_dimensions}
            
            # Add progress tracking for this text type
            total_texts = len(text_data)
            progress_interval = max(1, total_texts // 10)  # Show progress every 10%
            
            for count, (idx, text) in enumerate(text_data.items()):
                if count % progress_interval == 0:
                    print(f"    Progress: {count}/{total_texts} ({count/total_texts*100:.1f}%)")
                if pd.isna(text):
                    continue
                    
                tokens, languages = self.detect_token_language(text)
                
                if not tokens:
                    continue
                
                # Group tokens by language
                lang_segments = defaultdict(list)
                for token, lang in zip(tokens, languages):
                    if lang in ['hindi', 'romanized_hindi']:
                        lang_segments['hindi'].append(token)
                    elif lang == 'english':
                        lang_segments['english'].append(token)
                        
                # Calculate segment-level statistics
                total_tokens = len(tokens)
                hindi_tokens = len(lang_segments['hindi'])
                english_tokens = len(lang_segments['english'])
                
                # Get toxicity scores for this text
                row_toxicity = {}
                for dim in self.toxicity_dimensions:
                    col_name = toxicity_cols.get(dim)
                    if col_name in self.df.columns:
                        row_toxicity[dim] = self.df.loc[idx, col_name]
                    else:
                        row_toxicity[dim] = 0.0
                        
                # Store segment analysis
                segment_result = {
                    'text_type': text_name,
                    'text': text,
                    'total_tokens': total_tokens,
                    'hindi_tokens': hindi_tokens,
                    'english_tokens': english_tokens,
                    'hindi_percentage': (hindi_tokens / total_tokens) * 100 if total_tokens > 0 else 0,
                    'english_percentage': (english_tokens / total_tokens) * 100 if total_tokens > 0 else 0,
                    'primary_key': self.df.loc[idx, 'primary_key'] if 'primary_key' in self.df.columns else idx
                }
                
                # Add toxicity scores
                for dim in self.toxicity_dimensions:
                    segment_result[f'toxicity_{dim}'] = row_toxicity[dim]
                    
                results['segment_analysis'].append(segment_result)
                
        # Convert to DataFrame for easier analysis
        segment_df = pd.DataFrame(results['segment_analysis'])
        
        # Calculate aggregate statistics
        print("  Computing aggregate statistics...")
        for text_type in segment_df['text_type'].unique():
            text_data = segment_df[segment_df['text_type'] == text_type]
            
            type_stats = {
                'text_type': text_type,
                'total_samples': len(text_data),
                'avg_hindi_percentage': text_data['hindi_percentage'].mean(),
                'avg_english_percentage': text_data['english_percentage'].mean(),
                'std_hindi_percentage': text_data['hindi_percentage'].std(),
                'std_english_percentage': text_data['english_percentage'].std()
            }
            
            # Add toxicity statistics
            for dim in self.toxicity_dimensions:
                toxicity_col = f'toxicity_{dim}'
                if toxicity_col in text_data.columns:
                    type_stats[f'avg_{dim}'] = text_data[toxicity_col].mean()
                    type_stats[f'std_{dim}'] = text_data[toxicity_col].std()
                    
                    # Analyze correlation between language mix and toxicity
                    hindi_corr = text_data['hindi_percentage'].corr(text_data[toxicity_col])
                    english_corr = text_data['english_percentage'].corr(text_data[toxicity_col])
                    
                    type_stats[f'hindi_correlation_{dim}'] = hindi_corr
                    type_stats[f'english_correlation_{dim}'] = english_corr
                    
            results['aggregate_stats'][text_type] = type_stats
            
        # Analyze LLM continuation toxicity by prompt language
        print("  Analyzing LLM continuation toxicity by prompt language...")
        llm_analysis = self.analyze_llm_continuation_by_language(segment_df)
        results['llm_continuation_analysis'] = llm_analysis
        
        self.linguistic_results = results
        return results
        
    def analyze_llm_continuation_by_language(self, segment_df: pd.DataFrame) -> Dict:
        """
        Analyze how prompt language composition affects LLM continuation toxicity
        """
        llm_results = {}
        
        for model in self.models:
            model_results = {}
            
            for text_type in self.text_types.keys():
                # Get continuation toxicity scores for this model and prompt type
                continuation_data = []
                
                for idx, row in self.df.iterrows():
                    # Get language composition from segment analysis
                    segment_match = segment_df[
                        (segment_df['primary_key'] == row.get('primary_key', idx)) & 
                        (segment_df['text_type'] == self.text_types[text_type])
                    ]
                    
                    if segment_match.empty:
                        continue
                        
                    segment_info = segment_match.iloc[0]
                    
                    # Get LLM continuation toxicity scores
                    for dim in self.toxicity_dimensions:
                        continuation_col = f"{model}_{text_type}_continuation_{dim}"
                        if continuation_col in self.df.columns:
                            toxicity_score = row[continuation_col]
                            if pd.notna(toxicity_score):
                                continuation_data.append({
                                    'hindi_percentage': segment_info['hindi_percentage'],
                                    'english_percentage': segment_info['english_percentage'],
                                    'toxicity_dimension': dim,
                                    'toxicity_score': toxicity_score,
                                    'primary_key': row.get('primary_key', idx)
                                })
                                
                if continuation_data:
                    cont_df = pd.DataFrame(continuation_data)
                    
                    # Calculate correlations and statistics
                    type_analysis = {
                        'total_samples': len(cont_df),
                        'avg_toxicity_by_dimension': {},
                        'language_correlations': {}
                    }
                    
                    for dim in self.toxicity_dimensions:
                        dim_data = cont_df[cont_df['toxicity_dimension'] == dim]
                        if not dim_data.empty:
                            type_analysis['avg_toxicity_by_dimension'][dim] = dim_data['toxicity_score'].mean()
                            
                            # Correlation with language composition
                            hindi_corr = dim_data['hindi_percentage'].corr(dim_data['toxicity_score'])
                            english_corr = dim_data['english_percentage'].corr(dim_data['toxicity_score'])
                            
                            type_analysis['language_correlations'][dim] = {
                                'hindi_correlation': hindi_corr,
                                'english_correlation': english_corr
                            }
                            
                    model_results[text_type] = type_analysis
                    
            llm_results[model] = model_results
            
        return llm_results 

    def setup_attribution_model(self) -> bool:
        """
        Setup a pre-trained toxicity classification model for feature attribution
        """
        if not CAPTUM_AVAILABLE:
            print("Warning: Captum not available. Skipping feature attribution analysis.")
            return False
            
        print("Setting up toxicity classification model for feature attribution...")
        
        try:
            # Get HuggingFace token from config
            hf_token = None
            if config:
                hf_token = getattr(config, 'HUGGINGFACE_API_KEY', None)
                
            if hf_token:
                print("Using HuggingFace API token for authentication")
            else:
                print("Warning: No HuggingFace API token found - trying without authentication")
            
            # List of models to try for attribution
            MODELS_TO_TRY = [
                "unitary/toxic-bert",
                "cardiffnlp/twitter-roberta-base-offensive",
                "Hate-speech-CNERG/bert-base-uncased-hatexplain"
            ]
            
            model_loaded = False
            for model_name in MODELS_TO_TRY:
                try:
                    print(f"Attempting to load model: {model_name}")
                    
                    # Load with HuggingFace token if available
                    tokenizer_kwargs = {"trust_remote_code": True}
                    model_kwargs = {"trust_remote_code": True}
                    
                    if hf_token:
                        # Add token for authentication (use the newer 'token' parameter)
                        tokenizer_kwargs["token"] = hf_token
                        model_kwargs["token"] = hf_token
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
                    self.attribution_model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)
                    self.attribution_model.eval()
                    
                    # Ensure model gradients are enabled for attribution
                    for param in self.attribution_model.parameters():
                        param.requires_grad = True
                    
                    print(f"Successfully loaded model: {model_name}")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"Failed to load {model_name}: {e}")
                    continue
                    
            if not model_loaded:
                raise Exception("No toxicity classification model could be loaded")
            
            # Create a wrapper function for the model that returns logits
            def model_forward_func(input_ids, attention_mask=None):
                # Ensure input_ids are Long tensors
                input_ids = input_ids.long()
                if attention_mask is not None:
                    attention_mask = attention_mask.long()
                    outputs = self.attribution_model(input_ids, attention_mask=attention_mask)
                else:
                    outputs = self.attribution_model(input_ids)
                return outputs.logits
            
            # Initialize attribution methods 
            # Use the embedding layer as the interpretable layer for attribution
            # This avoids issues with integer token gradients
            embeddings = self.attribution_model.get_input_embeddings()
            self.model_forward_func = model_forward_func
            self.integrated_gradients = IntegratedGradients(model_forward_func)
            # Note: deeplift will be created per-call to avoid model compatibility issues
            self.gradient_shap = GradientShap(model_forward_func)
            self.input_embeddings = embeddings
            
            print("Attribution model setup complete")
            return True
            
        except Exception as e:
            print(f"Error setting up attribution model: {e}")
            print("Feature attribution analysis will be skipped - linguistic analysis will continue")
            print("Note: This is optional functionality. The main linguistic distribution analysis is unaffected.")
            return False
            
    def compute_feature_attribution(self, texts: List[str], method: str = 'integrated_gradients') -> List[Dict]:
        """
        Compute feature attribution for a list of texts using specified method
        """
        if not CAPTUM_AVAILABLE or not hasattr(self, 'attribution_model'):
            return []
            
        print(f"Computing feature attribution using {method}...")
        
        # Analyze ALL high-toxicity texts for comprehensive research analysis
        print(f"Analyzing ALL {len(texts)} high-toxicity texts for comprehensive attribution analysis")
        
        # Use cuda:0 specifically to avoid device conflicts
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            self.attribution_model = self.attribution_model.to(device)
            self.input_embeddings = self.input_embeddings.to(device)
            print(f"Using GPU acceleration: {device}")
        else:
            print("Using CPU for attribution analysis")
            
        results = []
        
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"Processing text {i+1}/{len(texts)}...")
                
            # Note: Analyzing ALL texts, including long ones (GPU can handle it)
            if pd.isna(text) or not isinstance(text, str):
                continue
                
            try:
                # Tokenize text with full sequence length for comprehensive analysis
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                                      padding=True, max_length=512)  # Restored to 512 for comprehensive analysis
                input_ids = inputs['input_ids'].long().to(device)  # Move to device
                attention_mask = inputs['attention_mask'].long().to(device) if 'attention_mask' in inputs else None
                
                # For attribution, we don't set requires_grad on input_ids (they're integers)
                # Captum will handle the embedding layer gradients internally
                
                # Get baseline (all pad tokens, properly typed)
                baseline_ids = torch.zeros_like(input_ids, dtype=torch.long).to(device)
                if attention_mask is not None:
                    baseline_mask = torch.zeros_like(attention_mask, dtype=torch.long).to(device)
                else:
                    baseline_mask = None
                
                # Convert to embeddings for attribution
                input_embeddings = self.input_embeddings(input_ids)
                baseline_embeddings = self.input_embeddings(baseline_ids)
                
                # Create a wrapper nn.Module for Captum. This is required for DeepLift,
                # which needs an nn.Module object to register hooks, not a function.
                class EmbeddingForwardModule(nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                        
                    def forward(self, embeddings, attention_mask=None):
                        # Use embeddings directly in the model's forward pass
                        if attention_mask is not None:
                            outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
                        else:
                            outputs = self.model(inputs_embeds=embeddings)
                        return outputs.logits

                embedding_forward_module = EmbeddingForwardModule(self.attribution_model)
                
                # Compute attribution based on method
                if method == 'integrated_gradients':
                    ig_embeddings = IntegratedGradients(embedding_forward_module)
                    if attention_mask is not None:
                        attributions = ig_embeddings.attribute(
                            input_embeddings, 
                            baselines=baseline_embeddings,
                            target=1,  # target=1 for toxic class
                            additional_forward_args=(attention_mask,),
                            internal_batch_size=2,  # Use larger batch for GPU efficiency
                            n_steps=50,  # Restored to default 50 for accuracy
                            return_convergence_delta=False
                        )
                    else:
                        attributions = ig_embeddings.attribute(
                            input_embeddings, 
                            baselines=baseline_embeddings,
                            target=1,  # target=1 for toxic class
                            internal_batch_size=2,  # Use larger batch for GPU efficiency
                            n_steps=50,  # Restored to default 50 for accuracy
                            return_convergence_delta=False
                        )
                elif method == 'deeplift':
                    # For DeepLift, use the nn.Module wrapper
                    dl = DeepLift(embedding_forward_module)
                    if attention_mask is not None:
                        attributions = dl.attribute(
                            input_embeddings, 
                            baselines=baseline_embeddings,
                            target=1,
                            additional_forward_args=(attention_mask,)
                        )
                    else:
                        attributions = dl.attribute(
                            input_embeddings, 
                            baselines=baseline_embeddings,
                            target=1
                        )
                elif method == 'gradient_shap':
                    # For GradientShap, create random embedding baselines
                    n_samples = 50
                    embedding_dim = input_embeddings.shape[-1]
                    # Create random embeddings as baselines (using normal distribution)
                    random_embeddings = torch.randn((n_samples,) + input_embeddings.shape[1:]) * 0.1
                    
                    gs = GradientShap(embedding_forward_module)
                    if attention_mask is not None:
                        attributions = gs.attribute(
                            input_embeddings, 
                            baselines=random_embeddings,
                            target=1,
                            additional_forward_args=(attention_mask,)
                        )
                    else:
                        attributions = gs.attribute(
                            input_embeddings, 
                            baselines=random_embeddings,
                            target=1
                        )
                else:
                    print(f"Unknown attribution method: {method}")
                    continue
                
                # Convert to token-level attributions
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                # Sum attribution scores across embedding dimensions to get per-token scores
                attr_scores = attributions[0].sum(dim=-1).detach().cpu().numpy()
                
                # Remove special tokens and aggregate subwords
                clean_tokens = []
                clean_attributions = []
                current_word = ""
                current_attr = 0.0
                
                for token, attr in zip(tokens, attr_scores):
                    if token in ['[CLS]', '[SEP]', '[PAD]']:
                        continue
                        
                    if token.startswith('##'):
                        # Subword continuation
                        current_word += token[2:]
                        current_attr += attr
                    else:
                        # New word
                        if current_word:
                            clean_tokens.append(current_word)
                            clean_attributions.append(current_attr)
                        current_word = token
                        current_attr = attr
                        
                # Add last word
                if current_word:
                    clean_tokens.append(current_word)
                    clean_attributions.append(current_attr)
                
                # Identify language for each token
                _, token_languages = self.detect_token_language(' '.join(clean_tokens))
                
                result = {
                    'text': text,
                    'index': i,
                    'method': method,
                    'tokens': clean_tokens,
                    'attributions': clean_attributions,
                    'token_languages': token_languages,
                    'total_attribution': sum(clean_attributions)
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing text {i}: {e}")
                # If too many errors, skip this method
                if len(results) == 0 and i > 10:  # If no successful attributions after 10 tries
                    print(f"Too many attribution errors for {method}, skipping remaining texts")
                    break
                continue
            finally:
                # Memory cleanup after each text
                if 'input_embeddings' in locals():
                    del input_embeddings
                if 'baseline_embeddings' in locals():
                    del baseline_embeddings
                if 'attributions' in locals():
                    del attributions
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
        return results
        
    def compute_feature_attribution_gpu(self, texts_batch: List[str], method: str, gpu_id: int) -> List[Dict]:
        """
        Compute feature attribution for a batch of texts on a specific GPU
        """
        # Use only GPU 0 to avoid device placement issues
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Ensure model is on the correct device
        attribution_model = self.attribution_model.to(device)
        tokenizer = self.tokenizer
        input_embeddings = self.input_embeddings.to(device)
        
        results = []
        
        print(f"GPU {gpu_id}: Processing {len(texts_batch)} texts...")
        
        for i, text in enumerate(texts_batch):
            if pd.isna(text) or not isinstance(text, str):
                continue
                
            try:
                # Tokenize text with full sequence length for comprehensive analysis
                inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                                  padding=True, max_length=512)
                input_ids = inputs['input_ids'].long().to(device)
                attention_mask = inputs['attention_mask'].long().to(device) if 'attention_mask' in inputs else None
                
                # Get baseline (all pad tokens, properly typed)
                baseline_ids = torch.zeros_like(input_ids, dtype=torch.long).to(device)
                if attention_mask is not None:
                    baseline_mask = torch.zeros_like(attention_mask, dtype=torch.long).to(device)
                else:
                    baseline_mask = None
                
                # Convert to embeddings for attribution
                text_embeddings = input_embeddings(input_ids)
                baseline_embeddings = input_embeddings(baseline_ids)
                
                # Create a wrapper nn.Module for Captum
                class EmbeddingForwardModule(nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                        
                    def forward(self, embeddings, attention_mask=None):
                        if attention_mask is not None:
                            outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
                        else:
                            outputs = self.model(inputs_embeds=embeddings)
                        return outputs.logits

                embedding_forward_module = EmbeddingForwardModule(attribution_model)
                
                # Compute attribution based on method
                if method == 'integrated_gradients':
                    ig_embeddings = IntegratedGradients(embedding_forward_module)
                    if attention_mask is not None:
                        attributions = ig_embeddings.attribute(
                            text_embeddings, 
                            baselines=baseline_embeddings,
                            target=1,
                            additional_forward_args=(attention_mask,),
                            internal_batch_size=4,  # Larger batch for GPU efficiency
                            n_steps=50,
                            return_convergence_delta=False
                        )
                    else:
                        attributions = ig_embeddings.attribute(
                            text_embeddings, 
                            baselines=baseline_embeddings,
                            target=1,
                            internal_batch_size=4,
                            n_steps=50,
                            return_convergence_delta=False
                        )
                elif method == 'deeplift':
                    dl = DeepLift(embedding_forward_module)
                    if attention_mask is not None:
                        attributions = dl.attribute(
                            text_embeddings, 
                            baselines=baseline_embeddings,
                            target=1,
                            additional_forward_args=(attention_mask,)
                        )
                    else:
                        attributions = dl.attribute(
                            text_embeddings, 
                            baselines=baseline_embeddings,
                            target=1
                        )
                else:
                    print(f"Unknown attribution method: {method}")
                    continue
                
                # Convert to token-level attributions
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                attr_scores = attributions[0].sum(dim=-1).detach().cpu().numpy()
                
                # Remove special tokens and aggregate subwords
                clean_tokens = []
                clean_attributions = []
                current_word = ""
                current_attr = 0.0
                
                for token, attr in zip(tokens, attr_scores):
                    if token in ['[CLS]', '[SEP]', '[PAD]']:
                        continue
                        
                    if token.startswith('##'):
                        current_word += token[2:]
                        current_attr += attr
                    else:
                        if current_word:
                            clean_tokens.append(current_word)
                            clean_attributions.append(current_attr)
                        current_word = token
                        current_attr = attr
                        
                if current_word:
                    clean_tokens.append(current_word)
                    clean_attributions.append(current_attr)
                
                # Identify language for each token
                _, token_languages = self.detect_token_language(' '.join(clean_tokens))
                
                result = {
                    'text': text,
                    'index': i,
                    'method': method,
                    'gpu_id': gpu_id,
                    'tokens': clean_tokens,
                    'attributions': clean_attributions,
                    'token_languages': token_languages,
                    'total_attribution': sum(clean_attributions)
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"GPU {gpu_id}: Error processing text {i}: {e}")
                continue
            finally:
                # Memory cleanup
                if 'text_embeddings' in locals():
                    del text_embeddings
                if 'baseline_embeddings' in locals():
                    del baseline_embeddings
                if 'attributions' in locals():
                    del attributions
                torch.cuda.empty_cache()
                
        return results
        
    def compute_feature_attribution_multi_gpu(self, texts: List[str], method: str) -> List[Dict]:
        """
        Compute feature attribution using single GPU to avoid device placement issues
        """
        print(f"Computing feature attribution for {len(texts)} texts using {method}...")
        print("Note: Using single GPU approach to avoid device placement issues")
        
        # Use single GPU processing to avoid device conflicts
        return self.compute_feature_attribution(texts, method)
        
    def analyze_attribution_by_language(self, attribution_results: List[Dict]) -> Dict:
        """
        Analyze feature attribution results by language to identify which language
        segments contribute most to toxicity perception
        """
        print("Analyzing attribution results by language...")
        
        analysis = {
            'language_attribution_stats': {},
            'top_tokens_by_language': {},
            'attribution_distribution': {},
            'method_comparison': {}
        }
        
        # Group results by attribution method
        methods = set(result['method'] for result in attribution_results)
        
        for method in methods:
            method_results = [r for r in attribution_results if r['method'] == method]
            
            # Aggregate attribution scores by language
            hindi_attributions = []
            english_attributions = []
            total_attributions = []
            
            # Token-level analysis
            token_scores = defaultdict(list)
            language_token_scores = {'hindi': defaultdict(list), 'english': defaultdict(list)}
            
            for result in method_results:
                tokens = result['tokens']
                attributions = result['attributions']
                languages = result['token_languages']
                
                total_attributions.append(result['total_attribution'])
                
                for token, attr, lang in zip(tokens, attributions, languages):
                    token_scores[token].append(attr)
                    
                    if lang in ['hindi', 'romanized_hindi']:
                        hindi_attributions.append(attr)
                        language_token_scores['hindi'][token].append(attr)
                    elif lang == 'english':
                        english_attributions.append(attr)
                        language_token_scores['english'][token].append(attr)
                        
            # Calculate statistics
            stats = {
                'hindi_stats': {
                    'mean_attribution': np.mean(hindi_attributions) if hindi_attributions else 0,
                    'std_attribution': np.std(hindi_attributions) if hindi_attributions else 0,
                    'median_attribution': np.median(hindi_attributions) if hindi_attributions else 0,
                    'total_tokens': len(hindi_attributions)
                },
                'english_stats': {
                    'mean_attribution': np.mean(english_attributions) if english_attributions else 0,
                    'std_attribution': np.std(english_attributions) if english_attributions else 0,
                    'median_attribution': np.median(english_attributions) if english_attributions else 0,
                    'total_tokens': len(english_attributions)
                },
                'overall_stats': {
                    'mean_total_attribution': np.mean(total_attributions) if total_attributions else 0,
                    'std_total_attribution': np.std(total_attributions) if total_attributions else 0
                }
            }
            
            analysis['language_attribution_stats'][method] = stats
            
            # Identify top contributing tokens by language
            top_tokens = {}
            for lang, token_dict in language_token_scores.items():
                # Calculate mean attribution for each token
                token_means = {token: np.mean(scores) for token, scores in token_dict.items()}
                # Sort by absolute attribution value
                sorted_tokens = sorted(token_means.items(), key=lambda x: abs(x[1]), reverse=True)
                top_tokens[lang] = sorted_tokens[:20]  # Top 20 tokens
                
            analysis['top_tokens_by_language'][method] = top_tokens
            
        return analysis
        
    def perform_feature_attribution_analysis(self):
        """
        Perform comprehensive feature attribution analysis on toxic texts
        """
        print("Starting feature attribution analysis...")
        
        if not self.setup_attribution_model():
            print("Skipping feature attribution analysis due to setup failure")
            return {}
            
        # Collect texts with high toxicity scores for analysis
        high_toxicity_texts = []
        toxicity_threshold = 0.5  # Adjust threshold as needed
        
        for text_col in self.text_types.keys():
            # Check original texts
            for idx, row in self.df.iterrows():
                text = row[text_col]
                if pd.isna(text):
                    continue
                    
                # Check if any toxicity dimension exceeds threshold
                is_toxic = False
                for dim in self.toxicity_dimensions:
                    toxicity_col = f"{text_col}_{dim}"
                    if toxicity_col in self.df.columns:
                        score = row[toxicity_col]
                        if pd.notna(score) and score > toxicity_threshold:
                            is_toxic = True
                            break
                            
                if is_toxic:
                    high_toxicity_texts.append({
                        'text': text,
                        'source': f"{text_col}_original",
                        'primary_key': row.get('primary_key', idx)
                    })
                    
            # Check LLM continuations
            for model in self.models:
                continuation_col = f"{model}_{text_col}_continuation"
                if continuation_col in self.df.columns:
                    for idx, row in self.df.iterrows():
                        text = row[continuation_col]
                        if pd.isna(text):
                            continue
                            
                        # Check if any toxicity dimension exceeds threshold
                        is_toxic = False
                        for dim in self.toxicity_dimensions:
                            toxicity_col = f"{continuation_col}_{dim}"
                            if toxicity_col in self.df.columns:
                                score = row[toxicity_col]
                                if pd.notna(score) and score > toxicity_threshold:
                                    is_toxic = True
                                    break
                                    
                        if is_toxic:
                            high_toxicity_texts.append({
                                'text': text,
                                'source': f"{model}_{text_col}_continuation",
                                'primary_key': row.get('primary_key', idx)
                            })
                            
        print(f"Found {len(high_toxicity_texts)} high-toxicity texts for attribution analysis")
        
        if not high_toxicity_texts:
            print("No high-toxicity texts found. Skipping attribution analysis.")
            return {}
            
        # Extract texts for attribution
        texts_for_attribution = [item['text'] for item in high_toxicity_texts]
        
        # Compute attributions using different methods for comprehensive analysis
        attribution_results = []
        
        methods = ['integrated_gradients', 'deeplift']  # Use both IG and DeepLift for comprehensive analysis
        
        for method in methods:
            print(f"Computing attributions using {method}...")
            method_results = self.compute_feature_attribution_multi_gpu(texts_for_attribution, method)
            attribution_results.extend(method_results)
            
        # Analyze results by language
        language_analysis = self.analyze_attribution_by_language(attribution_results)
        
        self.attribution_results = {
            'high_toxicity_texts': high_toxicity_texts,
            'attribution_results': attribution_results,
            'language_analysis': language_analysis
        }
        
        return self.attribution_results 

    def create_visualizations(self):
        """
        Create comprehensive visualizations for linguistic distribution and attribution analysis
        """
        print("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_style("whitegrid")
        sns.set_palette(professional_colors)
        
        # 1. Linguistic Distribution Visualizations
        if self.linguistic_results:
            self._create_linguistic_visualizations()
            
        # 2. Feature Attribution Visualizations  
        if self.attribution_results:
            self._create_attribution_visualizations()
            
    def _create_linguistic_visualizations(self):
        """Create visualizations for linguistic distribution analysis"""
        
        # Convert segment analysis to DataFrame
        segment_df = pd.DataFrame(self.linguistic_results['segment_analysis'])
        
        # 1. Language composition by text type
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Box plot of Hindi percentage by text type
        sns.boxplot(data=segment_df, x='text_type', y='hindi_percentage', ax=axes[0,0])
        apply_smart_font_to_plot(axes[0,0], 
                               title='Hindi Percentage Distribution by Text Type',
                               ylabel='Hindi Percentage (%)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Box plot of English percentage by text type
        sns.boxplot(data=segment_df, x='text_type', y='english_percentage', ax=axes[0,1])
        apply_smart_font_to_plot(axes[0,1],
                               title='English Percentage Distribution by Text Type',
                               ylabel='English Percentage (%)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Scatter plot: Hindi vs English percentage
        for i, text_type in enumerate(segment_df['text_type'].unique()):
            data = segment_df[segment_df['text_type'] == text_type]
            axes[1,0].scatter(data['hindi_percentage'], data['english_percentage'], 
                            label=text_type, alpha=0.6, color=professional_colors[i % len(professional_colors)])
        apply_smart_font_to_plot(axes[1,0],
                               title='Language Mix Distribution',
                               xlabel='Hindi Percentage (%)',
                               ylabel='English Percentage (%)')
        axes[1,0].legend()
        
        # Average toxicity by text type
        toxicity_means = []
        text_types = []
        for text_type in segment_df['text_type'].unique():
            data = segment_df[segment_df['text_type'] == text_type]
            avg_toxicity = data['toxicity_toxicity'].mean() if 'toxicity_toxicity' in data.columns else 0
            toxicity_means.append(avg_toxicity)
            text_types.append(text_type)
            
        axes[1,1].bar(text_types, toxicity_means, color=professional_colors[:len(text_types)])
        apply_smart_font_to_plot(axes[1,1],
                               title='Average Toxicity by Text Type',
                               ylabel='Average Toxicity Score')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'linguistic_distribution_overview.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlation heatmap between language percentages and toxicity
        if 'toxicity_toxicity' in segment_df.columns:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            corr_data = []
            for text_type in segment_df['text_type'].unique():
                data = segment_df[segment_df['text_type'] == text_type]
                
                for dim in self.toxicity_dimensions:
                    toxicity_col = f'toxicity_{dim}'
                    if toxicity_col in data.columns:
                        hindi_corr = data['hindi_percentage'].corr(data[toxicity_col])
                        english_corr = data['english_percentage'].corr(data[toxicity_col])
                        
                        corr_data.append({
                            'Text Type': text_type,
                            'Dimension': dim,
                            'Hindi Correlation': hindi_corr,
                            'English Correlation': english_corr
                        })
                        
            if corr_data:
                corr_df = pd.DataFrame(corr_data)
                pivot_hindi = corr_df.pivot(index='Text Type', columns='Dimension', values='Hindi Correlation')
                pivot_english = corr_df.pivot(index='Text Type', columns='Dimension', values='English Correlation')
                
                # Create side-by-side heatmaps
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                sns.heatmap(pivot_hindi, annot=True, cmap='RdBu_r', center=0, ax=ax1)
                apply_smart_font_to_plot(ax1, title='Hindi Percentage vs Toxicity Correlations')
                
                sns.heatmap(pivot_english, annot=True, cmap='RdBu_r', center=0, ax=ax2)
                apply_smart_font_to_plot(ax2, title='English Percentage vs Toxicity Correlations')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'language_toxicity_correlations.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
        # 3. LLM continuation analysis
        if 'llm_continuation_analysis' in self.linguistic_results:
            self._create_llm_continuation_visualizations()
            
    def _create_llm_continuation_visualizations(self):
        """Create visualizations for LLM continuation analysis"""
        
        llm_data = self.linguistic_results['llm_continuation_analysis']
        
        # Prepare data for visualization
        model_correlations = []
        
        for model in llm_data:
            for text_type in llm_data[model]:
                for dim in self.toxicity_dimensions:
                    if ('language_correlations' in llm_data[model][text_type] and 
                        dim in llm_data[model][text_type]['language_correlations']):
                        
                        corr_data = llm_data[model][text_type]['language_correlations'][dim]
                        
                        model_correlations.append({
                            'Model': model,
                            'Text Type': text_type,
                            'Dimension': dim,
                            'Hindi Correlation': corr_data.get('hindi_correlation', 0),
                            'English Correlation': corr_data.get('english_correlation', 0)
                        })
                        
        if model_correlations:
            corr_df = pd.DataFrame(model_correlations)
            
            # Create grouped bar plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Hindi correlations
            hindi_pivot = corr_df.pivot_table(index=['Model', 'Text Type'], 
                                            columns='Dimension', 
                                            values='Hindi Correlation')
            hindi_pivot.plot(kind='bar', ax=ax1, color=professional_colors)
            apply_smart_font_to_plot(ax1,
                                   title='LLM Continuation Toxicity vs Hindi Percentage Correlations',
                                   ylabel='Correlation Coefficient')
            ax1.legend(title='Toxicity Dimension', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.tick_params(axis='x', rotation=45)
            
            # English correlations
            english_pivot = corr_df.pivot_table(index=['Model', 'Text Type'], 
                                              columns='Dimension', 
                                              values='English Correlation')
            english_pivot.plot(kind='bar', ax=ax2, color=professional_colors)
            ax2.set_title('LLM Continuation Toxicity vs English Percentage Correlations')
            ax2.set_ylabel('Correlation Coefficient')
            ax2.legend(title='Toxicity Dimension', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'llm_continuation_correlations.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
    def _create_attribution_visualizations(self):
        """Create visualizations for feature attribution analysis"""
        
        if not self.attribution_results.get('language_analysis'):
            return
            
        lang_analysis = self.attribution_results['language_analysis']
        
        # 1. Attribution comparison by language
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        methods = list(lang_analysis['language_attribution_stats'].keys())
        
        for i, method in enumerate(methods):
            if i >= 2:  # Limit to 2 methods for visualization
                break
                
            stats = lang_analysis['language_attribution_stats'][method]
            
            # Mean attribution by language
            languages = ['Hindi', 'English']
            means = [stats['hindi_stats']['mean_attribution'], 
                    stats['english_stats']['mean_attribution']]
            stds = [stats['hindi_stats']['std_attribution'], 
                   stats['english_stats']['std_attribution']]
            
            axes[i,0].bar(languages, means, yerr=stds, capsize=5, 
                         color=professional_colors[:2])
            apply_smart_font_to_plot(axes[i,0],
                                   title=f'Mean Attribution by Language ({method})',
                                   ylabel='Mean Attribution Score')
            
            # Token count comparison
            token_counts = [stats['hindi_stats']['total_tokens'], 
                           stats['english_stats']['total_tokens']]
            axes[i,1].bar(languages, token_counts, color=professional_colors[:2])
            apply_smart_font_to_plot(axes[i,1],
                                   title=f'Token Count by Language ({method})',
                                   ylabel='Number of Tokens')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'attribution_by_language.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Top toxic tokens by language
        for method in methods:
            if method in lang_analysis['top_tokens_by_language']:
                top_tokens = lang_analysis['top_tokens_by_language'][method]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                
                # Hindi top tokens
                if 'hindi' in top_tokens and top_tokens['hindi']:
                    hindi_tokens = top_tokens['hindi'][:10]  # Top 10
                    tokens, scores = zip(*hindi_tokens)
                    
                    # Create bars
                    bars = ax1.barh(range(len(tokens)), [abs(s) for s in scores], 
                                   color=professional_colors[0])
                    ax1.set_yticks(range(len(tokens)))
                    
                    # Handle Hindi text rendering with proper encoding
                    clean_tokens = safe_text_render(tokens)
                    ax1.set_yticklabels(clean_tokens, fontsize=10)
                    
                    apply_smart_font_to_plot(ax1,
                                           title=f'Top Hindi Tokens by Attribution ({method})',
                                           xlabel='Absolute Attribution Score')
                    ax1.invert_yaxis()  # Show highest scores at top
                    
                # English top tokens
                if 'english' in top_tokens and top_tokens['english']:
                    english_tokens = top_tokens['english'][:10]  # Top 10
                    tokens, scores = zip(*english_tokens)
                    
                    bars = ax2.barh(range(len(tokens)), [abs(s) for s in scores], 
                                   color=professional_colors[1])
                    ax2.set_yticks(range(len(tokens)))
                    clean_tokens = safe_text_render(tokens)
                    ax2.set_yticklabels(clean_tokens, fontsize=10)
                    apply_smart_font_to_plot(ax2,
                                           title=f'Top English Tokens by Attribution ({method})',
                                           xlabel='Absolute Attribution Score')
                    ax2.invert_yaxis()  # Show highest scores at top
                    
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'top_tokens_{method}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
    def save_results(self):
        """Save all analysis results to files"""
        print("Saving analysis results...")
        
        # Save linguistic distribution results
        if self.linguistic_results:
            # Save segment analysis
            segment_df = pd.DataFrame(self.linguistic_results['segment_analysis'])
            segment_df.to_csv(os.path.join(self.output_dir, 'linguistic_segment_analysis.csv'), 
                             index=False)
            
            # Save aggregate statistics
            with open(os.path.join(self.output_dir, 'linguistic_aggregate_stats.json'), 'w') as f:
                json.dump(self.linguistic_results['aggregate_stats'], f, indent=2, default=str)
                
            # Save LLM continuation analysis
            if 'llm_continuation_analysis' in self.linguistic_results:
                with open(os.path.join(self.output_dir, 'llm_continuation_analysis.json'), 'w') as f:
                    json.dump(self.linguistic_results['llm_continuation_analysis'], f, 
                             indent=2, default=str)
                    
        # Save feature attribution results
        if self.attribution_results:
            # Save attribution analysis
            with open(os.path.join(self.output_dir, 'attribution_language_analysis.json'), 'w') as f:
                json.dump(self.attribution_results['language_analysis'], f, 
                         indent=2, default=str)
                
            # Save high toxicity texts list
            high_tox_df = pd.DataFrame(self.attribution_results['high_toxicity_texts'])
            high_tox_df.to_csv(os.path.join(self.output_dir, 'high_toxicity_texts.csv'), 
                              index=False)
                              
            # Save detailed attribution results (limit size)
            attribution_summary = []
            for result in self.attribution_results['attribution_results'][:50]:  # Limit to first 50
                summary = {
                    'text': result['text'][:200] + '...' if len(result['text']) > 200 else result['text'],
                    'method': result['method'],
                    'total_attribution': result['total_attribution'],
                    'num_tokens': len(result['tokens'])
                }
                attribution_summary.append(summary)
                
            with open(os.path.join(self.output_dir, 'attribution_summary.json'), 'w') as f:
                json.dump(attribution_summary, f, indent=2, default=str)
                
        # Create summary report
        self._create_summary_report()
        
    def _create_summary_report(self):
        """Create a comprehensive summary report"""
        report_lines = []
        report_lines.append("# Linguistic Distribution of Toxicity and Feature Attribution Analysis")
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Dataset summary
        report_lines.append("## Dataset Summary")
        report_lines.append(f"Total samples analyzed: {len(self.df)}")
        report_lines.append(f"Text types: {list(self.text_types.values())}")
        report_lines.append(f"Models analyzed: {self.models}")
        report_lines.append(f"Toxicity dimensions: {self.toxicity_dimensions}")
        report_lines.append("")
        
        # Linguistic analysis summary
        if self.linguistic_results:
            report_lines.append("## Linguistic Distribution Analysis")
            segment_df = pd.DataFrame(self.linguistic_results['segment_analysis'])
            
            report_lines.append(f"Total text segments analyzed: {len(segment_df)}")
            report_lines.append("")
            
            # Summary by text type
            for text_type in segment_df['text_type'].unique():
                data = segment_df[segment_df['text_type'] == text_type]
                report_lines.append(f"### {text_type}")
                report_lines.append(f"- Samples: {len(data)}")
                report_lines.append(f"- Average Hindi percentage: {data['hindi_percentage'].mean():.2f}%")
                report_lines.append(f"- Average English percentage: {data['english_percentage'].mean():.2f}%")
                
                if 'toxicity_toxicity' in data.columns:
                    avg_toxicity = data['toxicity_toxicity'].mean()
                    report_lines.append(f"- Average toxicity score: {avg_toxicity:.4f}")
                    
                    hindi_corr = data['hindi_percentage'].corr(data['toxicity_toxicity'])
                    english_corr = data['english_percentage'].corr(data['toxicity_toxicity'])
                    report_lines.append(f"- Hindi percentage vs toxicity correlation: {hindi_corr:.4f}")
                    report_lines.append(f"- English percentage vs toxicity correlation: {english_corr:.4f}")
                    
                report_lines.append("")
                
        # Attribution analysis summary
        if self.attribution_results:
            report_lines.append("## Feature Attribution Analysis")
            
            num_texts = len(self.attribution_results['high_toxicity_texts'])
            report_lines.append(f"High-toxicity texts analyzed: {num_texts}")
            
            lang_analysis = self.attribution_results['language_analysis']
            
            for method in lang_analysis['language_attribution_stats']:
                stats = lang_analysis['language_attribution_stats'][method]
                report_lines.append(f"### {method.replace('_', ' ').title()}")
                
                hindi_mean = stats['hindi_stats']['mean_attribution']
                english_mean = stats['english_stats']['mean_attribution']
                hindi_tokens = stats['hindi_stats']['total_tokens']
                english_tokens = stats['english_stats']['total_tokens']
                
                report_lines.append(f"- Hindi tokens analyzed: {hindi_tokens}")
                report_lines.append(f"- English tokens analyzed: {english_tokens}")
                report_lines.append(f"- Mean Hindi attribution: {hindi_mean:.6f}")
                report_lines.append(f"- Mean English attribution: {english_mean:.6f}")
                
                if hindi_mean != 0 and english_mean != 0:
                    ratio = abs(hindi_mean) / abs(english_mean)
                    if ratio > 1:
                        report_lines.append(f"- Hindi tokens are {ratio:.2f}x more attributionally important")
                    else:
                        report_lines.append(f"- English tokens are {1/ratio:.2f}x more attributionally important")
                        
                report_lines.append("")
                
        # Save report
        with open(os.path.join(self.output_dir, 'analysis_summary_report.txt'), 'w') as f:
            f.write('\n'.join(report_lines))
            
    def run_analysis(self):
        """Run the complete linguistic distribution and feature attribution analysis"""
        print("=" * 80)
        print("LINGUISTIC DISTRIBUTION OF TOXICITY AND FEATURE ATTRIBUTION ANALYSIS")
        print("=" * 80)
        
        # Load and setup
        if not self.load_data():
            print("Failed to load data. Exiting.")
            return False
            
        if not self.setup_language_identification():
            print("Failed to setup language identification. Exiting.")
            return False
            
        # Perform linguistic distribution analysis
        print("\n" + "-" * 60)
        print("PART 1: LINGUISTIC DISTRIBUTION ANALYSIS")
        print("-" * 60)
        
        self.analyze_linguistic_distribution()
        
        # Perform feature attribution analysis (if not skipped and Captum is available)
        if not self.skip_attribution:
            print("\n" + "-" * 60)
            print("PART 2: FEATURE ATTRIBUTION ANALYSIS")
            print("-" * 60)
            self.perform_feature_attribution_analysis()
        else:
            print("\n" + "-" * 60)
            print("PART 2: FEATURE ATTRIBUTION ANALYSIS - SKIPPED")
            print("-" * 60)
            print("Feature attribution analysis was skipped as requested.")
        
        # Create visualizations and save results
        print("\n" + "-" * 60)
        print("GENERATING OUTPUTS")
        print("-" * 60)
        
        self.create_visualizations()
        self.save_results()
        
        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")
        print("\nGenerated files:")
        for file in sorted(os.listdir(self.output_dir)):
            print(f"  - {file}")
            
        # Cleanup temporary font file if it was downloaded
        global FONT_PATH
        if FONT_PATH and os.path.exists(FONT_PATH):
            try:
                import tempfile
                import shutil
                # Remove the temporary directory
                temp_dir = os.path.dirname(FONT_PATH)
                if temp_dir.startswith(tempfile.gettempdir()):
                    shutil.rmtree(temp_dir)
                    print("Cleaned up temporary font files")
            except Exception as e:
                print(f"Warning: Could not clean up temporary font file: {e}")
            
        return True 

def process_batch(batch, model, tokenizer, device, batch_size=32):
    """Process a batch of texts for feature attribution"""
    results = []
    
    # Tokenize batch
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        toxicity_scores = probs[:, 1].cpu().numpy()
    
    # Process each sample in the batch
    for i, (text, score) in enumerate(zip(batch, toxicity_scores)):
        try:
            # Get feature attributions for this sample
            attributions = get_feature_attributions(text, model, tokenizer, device)
            
            # Get linguistic features
            features = get_linguistic_features(text)
            
            results.append({
                'text': text,
                'toxicity_score': float(score),
                'attributions': attributions,
                'linguistic_features': features
            })
            
        except Exception as e:
            print(f"Error processing sample {i}: {str(e)}")
            continue
    
    return results

def get_feature_attributions(text, model, tokenizer, device, batch_size=32):
    """Get feature attributions for a text using batched processing"""
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get base prediction
    with torch.no_grad():
        outputs = model(**inputs)
        base_logits = outputs.logits
        base_probs = torch.softmax(base_logits, dim=-1)
        base_score = base_probs[0, 1].item()
    
    # Get token attributions
    token_attributions = []
    input_ids = inputs['input_ids'][0]
    
    # Process tokens in batches
    for i in range(0, len(input_ids), batch_size):
        batch_tokens = input_ids[i:i + batch_size]
        batch_attributions = []
        
        for token_id in batch_tokens:
            # Create masked input
            masked_inputs = inputs.copy()
            masked_inputs['input_ids'][0, i] = tokenizer.pad_token_id
            
            # Get prediction with masked token
            with torch.no_grad():
                masked_outputs = model(**masked_inputs)
                masked_logits = masked_outputs.logits
                masked_probs = torch.softmax(masked_logits, dim=-1)
                masked_score = masked_probs[0, 1].item()
            
            # Calculate attribution
            attribution = base_score - masked_score
            batch_attributions.append(attribution)
        
        token_attributions.extend(batch_attributions)
    
    # Convert to dictionary
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    return {token: float(attr) for token, attr in zip(tokens, token_attributions)}

def process_batch_multi_gpu(texts, model, tokenizer, device, batch_size):
    """Process a batch of texts using multiple GPUs"""
    results = []
    num_gpus = torch.cuda.device_count()
    
    # Split texts across GPUs
    texts_per_gpu = len(texts) // num_gpus
    gpu_texts = [texts[i:i + texts_per_gpu] for i in range(0, len(texts), texts_per_gpu)]
    
    # Process each GPU's batch
    for gpu_id, gpu_batch in enumerate(gpu_texts):
        try:
            # Set current device
            torch.cuda.set_device(gpu_id)
            current_device = torch.device(f'cuda:{gpu_id}')
            
            # Move model to current device
            model = model.to(current_device)
            
            # Process batch
            for text in gpu_batch:
                try:
                    # Tokenize and move to current device
                    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                    inputs = {k: v.to(current_device) for k, v in inputs.items()}
                    
                    # Get attributions
                    attributions = compute_attributions(model, inputs, current_device)
                    results.append({
                        'text': text,
                        'attributions': attributions
                    })
                except Exception as e:
                    print(f"GPU {gpu_id}: Error processing text: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"GPU {gpu_id} failed with error: {str(e)}")
            continue
            
    return results

def compute_attributions(model, inputs, device):
    """Compute attributions for a single input"""
    # Ensure model and inputs are on the same device
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
    
    # Compute attributions using integrated gradients
    attributions = []
    for i in range(predictions.shape[1]):
        # Create baseline
        baseline = torch.zeros_like(inputs['input_ids'])
        
        # Compute integrated gradients
        steps = 50
        scaled_inputs = []
        for step in range(steps + 1):
            alpha = step / steps
            scaled_input = baseline + alpha * (inputs['input_ids'] - baseline)
            scaled_inputs.append(scaled_input)
        
        # Stack scaled inputs
        scaled_inputs = torch.cat(scaled_inputs, dim=0)
        
        # Get gradients
        scaled_inputs.requires_grad_(True)
        outputs = model(input_ids=scaled_inputs, attention_mask=inputs['attention_mask'].repeat(steps + 1, 1))
        predictions = outputs.logits[:, i]
        gradients = torch.autograd.grad(predictions.sum(), scaled_inputs)[0]
        
        # Compute attributions
        attributions.append((gradients * (inputs['input_ids'] - baseline)).sum(dim=0))
    
    return torch.stack(attributions).cpu().numpy()

def main():
    """Main function to run the linguistic distribution and feature attribution analysis"""
    parser = argparse.ArgumentParser(
        description="Linguistic Distribution of Toxicity and Feature Attribution Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script performs two main analyses:

1. Linguistic Distribution Analysis:
   - Analyzes distribution of toxicity scores across Hindi and English segments
   - Identifies correlations between language composition and toxicity
   - Examines how prompt language affects LLM continuation toxicity

2. Feature Attribution Analysis (requires PyTorch + Captum):
   - Uses Integrated Gradients and DeepLift to identify influential tokens
   - Analyzes which language segments contribute most to toxicity perception
   - Creates saliency maps for toxic content

Example usage:
    python linguistic_distribution_toxicity_analysis.py \\
        --input_file final_outputs/perspective_analysis.csv \\
        --output_dir final_outputs \\
        --fasttext_model lid.176.bin

Output files will be saved to final_outputs/experiment_e/
        """
    )
    
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Path to the CSV file containing perspective analysis results'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory where experiment_e subdirectory will be created for outputs'
    )
    
    parser.add_argument(
        '--fasttext_model',
        type=str,
        default='lid.176.bin',
        help='Path to FastText language identification model (default: lid.176.bin)'
    )
    
    parser.add_argument(
        '--skip_attribution',
        action='store_true',
        help='Skip feature attribution analysis (useful if HuggingFace models are not accessible)'
    )
    
    parser.add_argument(
        '--multi_gpu',
        action='store_true',
        help='Use multiple GPUs for parallel attribution analysis (requires 4 GPUs)'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found!")
        return 1
        
    if not os.path.exists(args.fasttext_model):
        print(f"Error: FastText model '{args.fasttext_model}' not found!")
        return 1
        
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory '{args.output_dir}' not found!")
        return 1
    
    # Create analyzer and run analysis
    analyzer = LinguisticToxicityAnalyzer(
        input_file=args.input_file,
        output_dir=args.output_dir,
        fasttext_model=args.fasttext_model,
        multi_gpu=args.multi_gpu
    )
    
    # Set skip attribution flag if specified
    if args.skip_attribution:
        analyzer.skip_attribution = True
    
    success = analyzer.run_analysis()
    
    if success:
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Results available in: {analyzer.output_dir}")
        print("\nKey findings can be found in:")
        print("  - analysis_summary_report.txt (comprehensive summary)")
        print("  - linguistic_segment_analysis.csv (detailed segment data)")
        print("  - attribution_language_analysis.json (feature attribution results)")
        print("  - Various PNG files (visualizations)")
        return 0
    else:
        print("\n" + "=" * 80)
        print("ANALYSIS FAILED")
        print("=" * 80)
        print("Check the output above for error details.")
        return 1

if __name__ == "__main__":
    exit(main()) 