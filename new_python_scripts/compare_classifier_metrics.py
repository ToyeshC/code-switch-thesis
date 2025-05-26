import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Compare classifier metrics across different models")
    parser.add_argument("--metrics_dir", type=str, required=True, help="Directory containing metrics for different models")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the HTML report")
    parser.add_argument("--models", type=str, help="Comma-separated list of model names to compare")
    return parser.parse_args()

def load_metrics(metrics_dir, model_name):
    """Load metrics for a specific model from its JSON file"""
    metrics_file = os.path.join(metrics_dir, model_name, 'metrics.json')
    
    if not os.path.exists(metrics_file):
        print(f"Warning: Metrics file not found for model {model_name}")
        return None
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def create_comparison_plots(metrics_data, output_dir):
    """Create comparison plots for different metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Extract metrics for comparison
        models = list(metrics_data.keys())
        accuracy = [metrics_data[model]['accuracy'] for model in models]
        precision = [metrics_data[model]['precision'] for model in models]
        recall = [metrics_data[model]['recall'] for model in models]
        f1 = [metrics_data[model]['f1'] for model in models]
        roc_auc = [metrics_data[model]['roc_auc'] for model in models]
        
        # Create dataframe for plotting
        df = pd.DataFrame({
            'Model': models,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC': roc_auc
        })
        
        # Melt dataframe for better plotting
        df_melted = pd.melt(df, id_vars=['Model'], var_name='Metric', value_name='Value')
        
        # Create bar plots for each metric
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Model', y='Value', hue='Metric', data=df_melted)
        plt.title('Classifier Performance Metrics Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'metrics_comparison.png')
        plt.savefig(plot_path)
        plt.close()
        
        # Create individual metric plots
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
        metric_plots = {}
        
        for metric in metrics:
            try:
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Model', y='Value', data=df_melted[df_melted['Metric'] == metric])
                plt.title(f'{metric} Comparison')
                plt.xticks(rotation=45)
                plt.tight_layout()
                metric_plot_path = os.path.join(output_dir, f'{metric.lower().replace(" ", "_")}_comparison.png')
                plt.savefig(metric_plot_path)
                plt.close()
                metric_plots[metric] = metric_plot_path
            except Exception as e:
                print(f"Error creating plot for {metric}: {e}")
                continue
        
        return {
            'all_metrics': plot_path,
            'individual_metrics': metric_plots
        }
    except Exception as e:
        print(f"Error creating comparison plots: {e}")
        # Return empty paths to avoid breaking the report generation
        return {
            'all_metrics': '',
            'individual_metrics': {}
        }

def generate_html_report(metrics_data, plots, output_file):
    """Generate an HTML report with the comparison results"""
    # Create a DataFrame for the metrics table
    metrics_table = pd.DataFrame({
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'ROC AUC': []
    })
    
    # Add metrics for each model to the table
    rows = []
    for model, metrics in metrics_data.items():
        rows.append({
            'Model': model,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1 Score': f"{metrics['f1']:.4f}",
            'ROC AUC': f"{metrics['roc_auc']:.4f}"
        })
    
    # Combine rows into metrics table
    if rows:
        metrics_table = pd.concat([metrics_table, pd.DataFrame(rows)], ignore_index=True)
    
    # Sort by accuracy (descending)
    metrics_table = metrics_table.sort_values(by='Accuracy', ascending=False)
    
    # Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Classifier Metrics Comparison</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .best {{ font-weight: bold; color: green; }}
            img {{ max-width: 100%; height: auto; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>Classifier Metrics Comparison</h1>
        <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Summary Table</h2>
    """
    
    # Add metrics table if we have data
    if not metrics_table.empty:
        html += f"{metrics_table.to_html(index=False, classes='dataframe')}"
    else:
        html += "<p>No metrics data available for comparison.</p>"
    
    # Add overall comparison plot if available
    html += "<h2>Metrics Comparison</h2>"
    if plots.get('all_metrics') and os.path.exists(plots['all_metrics']):
        rel_path = os.path.relpath(plots['all_metrics'], os.path.dirname(output_file))
        html += f"<img src='{rel_path}' alt='Metrics Comparison'>"
    else:
        html += "<p>No comparison plot available.</p>"
    
    # Add individual metric plots if available
    html += "<h2>Individual Metric Comparisons</h2>"
    if plots.get('individual_metrics'):
        for metric, plot_path in plots['individual_metrics'].items():
            if os.path.exists(plot_path):
                rel_path = os.path.relpath(plot_path, os.path.dirname(output_file))
                html += f"""
                <h3>{metric}</h3>
                <img src="{rel_path}" alt="{metric} Comparison">
                """
    else:
        html += "<p>No individual metric plots available.</p>"
    
    html += """
    </body>
    </html>
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write HTML to file
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"HTML report saved to {output_file}")

def main():
    args = parse_args()
    
    # Get list of models to compare
    if args.models:
        models = args.models.split(',')
    else:
        # Use all subdirectories in metrics_dir as model names
        models = [d for d in os.listdir(args.metrics_dir) 
                 if os.path.isdir(os.path.join(args.metrics_dir, d)) and 
                 os.path.exists(os.path.join(args.metrics_dir, d, 'metrics.json'))]
    
    # Load metrics for each model
    metrics_data = {}
    for model in models:
        metrics = load_metrics(args.metrics_dir, model)
        if metrics:
            metrics_data[model] = metrics
    
    if not metrics_data:
        print("No valid metrics found for any model")
        return
    
    # Create plots directory
    plots_dir = os.path.join(os.path.dirname(args.output_file), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create comparison plots
    plots = create_comparison_plots(metrics_data, plots_dir)
    
    # Generate HTML report
    generate_html_report(metrics_data, plots, args.output_file)
    
    print(f"Comparison completed for {len(metrics_data)} models")

if __name__ == "__main__":
    main() 