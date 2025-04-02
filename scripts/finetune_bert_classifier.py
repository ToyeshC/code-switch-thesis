#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to fine-tune a BERT-based classifier for toxicity detection in code-switched (Hindi-English) text.
Uses mBERT (multilingual BERT) which is pre-trained on 104 languages including Hindi and English.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    AdamW, 
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# Set up command line arguments
parser = argparse.ArgumentParser(description='Fine-tune BERT for toxicity classification')
parser.add_argument('--input_file', type=str, required=True, help='CSV file with prompts and toxicity labels')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model and results')
parser.add_argument('--model_name', type=str, default='bert-base-multilingual-cased', 
                    help='Pre-trained model name (default: bert-base-multilingual-cased)')
parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
parser.add_argument('--epochs', type=int, default=4, help='Number of training epochs')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

# Set random seeds for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Define Dataset class
class ToxicityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

def prepare_data(input_file):
    """Load and prepare data for training."""
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Check if 'toxicity' column exists in the dataframe
    if 'TOXICITY' in df.columns:
        toxicity_col = 'TOXICITY'
    elif 'toxicity' in df.columns:
        toxicity_col = 'toxicity'
    else:
        # If neither exists, create a binary label from scores
        # This is a fallback if the exact format is not known
        if any(col.startswith('TOXICITY:') for col in df.columns):
            toxicity_scores = [col for col in df.columns if col.startswith('TOXICITY:')]
            df['toxicity'] = df[toxicity_scores].mean(axis=1)
            toxicity_col = 'toxicity'
    
    # Create binary labels (toxic if score >= 0.5)
    df['toxic_label'] = (df[toxicity_col] >= 0.5).astype(int)
    
    # Get text column
    if 'text' in df.columns:
        text_col = 'text'
    elif 'prompt' in df.columns:
        text_col = 'prompt'
    elif 'response' in df.columns:
        text_col = 'response'
    else:
        # Try to find a text column
        possible_text_cols = ['sentence', 'content', 'message']
        for col in possible_text_cols:
            if col in df.columns:
                text_col = col
                break
        else:
            # If no text column found, use the first string column
            text_cols = df.select_dtypes(include=['object']).columns
            text_col = text_cols[0]
    
    print(f"Using '{text_col}' as text column and '{toxicity_col}' for toxicity scores")
    
    # Balance dataset - ensure equal representation of toxic and non-toxic examples
    toxic_samples = df[df['toxic_label'] == 1]
    non_toxic_samples = df[df['toxic_label'] == 0]
    
    # If one class dominates, sample with replacement to balance
    if len(toxic_samples) < len(non_toxic_samples):
        toxic_samples = toxic_samples.sample(len(non_toxic_samples), replace=True, random_state=args.seed)
    elif len(non_toxic_samples) < len(toxic_samples):
        non_toxic_samples = non_toxic_samples.sample(len(toxic_samples), replace=True, random_state=args.seed)
    
    # Combine balanced datasets
    balanced_df = pd.concat([toxic_samples, non_toxic_samples]).sample(frac=1, random_state=args.seed)
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(balanced_df, test_size=0.2, random_state=args.seed, stratify=balanced_df['toxic_label'])
    
    print(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}")
    print(f"Toxic samples in training: {train_df['toxic_label'].sum()}, Non-toxic: {len(train_df) - train_df['toxic_label'].sum()}")
    
    return train_df, val_df, text_col

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Clear gradients
        optimizer.zero_grad()
        
        # Get inputs
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        scheduler.step()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """Evaluate the model on the validation set."""
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get inputs
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get predictions
            logits = outputs.logits
            preds = torch.sigmoid(logits).cpu().numpy()
            preds = (preds > 0.5).astype(int)
            
            # Add to lists
            predictions.extend(preds)
            actual_labels.extend(batch['label'].cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(actual_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_labels, predictions, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    # Load and prepare data
    train_df, val_df, text_col = prepare_data(args.input_file)
    
    # Load tokenizer and model
    print(f"Loading {args.model_name} tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,  # Binary classification
        problem_type="binary_classification"
    )
    
    # Create datasets
    train_dataset = ToxicityDataset(
        texts=train_df[text_col].values,
        labels=train_df['toxic_label'].values,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    val_dataset = ToxicityDataset(
        texts=val_df[text_col].values,
        labels=val_df['toxic_label'].values,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size
    )
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    print("Starting training...")
    best_f1 = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        print(f"Training loss: {train_loss:.4f}")
        
        # Evaluate
        metrics = evaluate(model, val_dataloader, device)
        print(f"Validation metrics: {metrics}")
        
        # Save best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            print(f"New best F1: {best_f1:.4f}, saving model...")
            
            # Save model
            model_save_path = os.path.join(args.output_dir, 'best_model')
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            
            # Save metrics
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(os.path.join(args.output_dir, 'best_metrics.csv'), index=False)
    
    print(f"Training complete! Best F1: {best_f1:.4f}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model')
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Save config and arguments
    with open(os.path.join(args.output_dir, 'training_args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

if __name__ == "__main__":
    main() 