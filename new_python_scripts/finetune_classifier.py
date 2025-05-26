#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import logging
import json
import matplotlib.pyplot as plt
from datasets import Dataset
import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Main metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def preprocess_data(df, text_column, label_column, max_samples):
    # Ensure label column is numeric
    if df[label_column].dtype == 'object':
        df[label_column] = pd.to_numeric(df[label_column], errors='coerce')
    
    # Drop NaN values
    df = df.dropna(subset=[text_column, label_column])
    
    # Convert continuous toxicity scores to binary labels (threshold = 0.5)
    if df[label_column].min() >= 0 and df[label_column].max() <= 1:
        print(f"Converting continuous scores to binary labels using threshold 0.5")
        df['label'] = (df[label_column] >= 0.5).astype(int)
    else:
        df['label'] = df[label_column]
    
    # Limit samples if specified
    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)
    
    return df

def evaluate_model(trainer, eval_dataset, metrics_dir):
    # Get test predictions
    test_results = trainer.evaluate(eval_dataset)
    print(f"Test results: {test_results}")
    
    # Get detailed predictions for analysis
    test_predictions = trainer.predict(eval_dataset)
    
    # Handle different versions of transformers predict output
    if hasattr(test_predictions, 'predictions'):
        logits = test_predictions.predictions
        labels = test_predictions.label_ids
    else:
        # For older versions where predict returns a tuple
        logits, labels = test_predictions[:2]
    
    predicted_labels = np.argmax(logits, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted_labels, average='binary')
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predicted_labels)
    
    # Calculate ROC curve and AUC
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'roc_auc': float(roc_auc),
        'test_loss': float(test_results.get('eval_loss', 0.0))
    }
    
    # Create metrics directory if it doesn't exist
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    
    # Save metrics as JSON
    with open(os.path.join(metrics_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['Non-toxic', 'Toxic']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(metrics_dir, 'confusion_matrix.png'))
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(metrics_dir, 'roc_curve.png'))
    
    # Save class distribution
    class_distribution = pd.Series(labels).value_counts().sort_index().to_dict()
    with open(os.path.join(metrics_dir, 'class_distribution.json'), 'w') as f:
        json.dump(class_distribution, f, indent=4)
    
    # Print summary
    print(f"\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    return metrics

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
    parser.add_argument(
        "--metrics_dir",
        type=str,
        help="Directory to save evaluation metrics"
    )
    parser.add_argument(
        "--evaluate_accuracy",
        type=str,
        default="false",
        help="Whether to perform detailed accuracy evaluation"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {args.input_file}")
    df = pd.read_csv(args.input_file)
    
    # Preprocess data
    df = preprocess_data(df, args.text_column, args.label_column, args.max_samples)
    
    # Split data into train and test sets
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.random_state, stratify=df['label']
    )
    
    logger.info(f"Training set size: {len(train_df)}")
    logger.info(f"Evaluation set size: {len(test_df)}")
    
    # Convert to datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples[args.text_column], 
            padding="max_length", 
            truncation=True, 
            max_length=args.max_length
        )
    
    # Apply tokenization
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    # Set format for pytorch
    tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        learning_rate=args.learning_rate,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        save_total_limit=2,
        do_eval=True,
        eval_accumulation_steps=1,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )
    
    # Train model
    logger.info("Starting model training...")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Evaluate and save metrics if requested
    if args.evaluate_accuracy.lower() == "true" and args.metrics_dir:
        logger.info(f"Evaluating model and saving metrics to {args.metrics_dir}")
        evaluate_model(trainer, tokenized_test, args.metrics_dir)

if __name__ == "__main__":
    main() 