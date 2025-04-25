#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            padding="max_length", 
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __getitem__(self, idx):
        item = {
            key: val[idx] for key, val in self.encodings.items()
        }
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.squeeze()
    
    # Convert to binary for classification metrics
    binary_preds = (preds > 0.5).astype(int)
    binary_labels = (labels > 0.5).astype(int)
    
    # Calculate regression metrics
    mse = ((preds - labels) ** 2).mean()
    rmse = np.sqrt(mse)
    
    # Calculate classification metrics on binarized data
    precision, recall, f1, _ = precision_recall_fscore_support(
        binary_labels, binary_preds, average="binary", zero_division=0
    )
    acc = accuracy_score(binary_labels, binary_preds)
    
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "mse": mse,
        "rmse": rmse,
    }

def train_model(
    train_texts, 
    train_labels, 
    eval_texts, 
    eval_labels, 
    model_name, 
    output_dir,
    batch_size=8,
    learning_rate=2e-5,
    epochs=3,
    warmup_steps=500,
    weight_decay=0.01,
    max_length=512,
):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1
    )
    
    # Create datasets
    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer, max_length
    )
    eval_dataset = TextClassificationDataset(
        eval_texts, eval_labels, tokenizer, max_length
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        save_steps=100,  # Save every 100 steps
        learning_rate=learning_rate,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    logger.info("Starting model training...")
    trainer.train()
    
    # Evaluate the model
    logger.info("Evaluating model...")
    eval_result = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_result}")
    
    # Save model and tokenizer
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a classifier model")
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--text_column", 
        type=str, 
        required=True, 
        help="Column name containing text"
    )
    parser.add_argument(
        "--label_column", 
        type=str, 
        required=True, 
        help="Column name containing labels"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="xlm-roberta-base", 
        help="Pretrained model name"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to save model"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8, 
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=2e-5, 
        help="Learning rate"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3, 
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=512, 
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--test_size", 
        type=float, 
        default=0.2, 
        help="Test set size for splitting"
    )
    parser.add_argument(
        "--random_state", 
        type=int, 
        default=42, 
        help="Random state for reproducibility"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {args.input_file}")
    df = pd.read_csv(args.input_file)
    
    # Limit samples if specified
    if args.max_samples and args.max_samples < len(df):
        df = df.sample(args.max_samples, random_state=args.random_state)
        logger.info(f"Using {args.max_samples} samples out of {len(df)}")
    
    # Extract texts and labels
    texts = df[args.text_column].tolist()
    labels = df[args.label_column].tolist()
    
    # Split data into train and evaluation sets
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts, labels, test_size=args.test_size, random_state=args.random_state
    )
    
    logger.info(f"Training set size: {len(train_texts)}")
    logger.info(f"Evaluation set size: {len(eval_texts)}")
    
    # Train model
    train_model(
        train_texts=train_texts,
        train_labels=train_labels,
        eval_texts=eval_texts,
        eval_labels=eval_labels,
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_length=args.max_length,
    )
    
    logger.info("Fine-tuning completed successfully")

if __name__ == "__main__":
    main() 