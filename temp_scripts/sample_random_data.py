import pandas as pd
import random

# Set random seed for reproducibility
random.seed(42)

# Read the CSV file
input_file = 'temp_scripts/perspective_analysis_outputs/perspective_analysis_results.csv'
output_file = 'temp_scripts/zzzz.csv'

# Read the CSV file
df = pd.read_csv(input_file)

# Randomly sample n rows
num_rows = 50
sampled_df = df.sample(n=num_rows, random_state=42)

# Save the sampled data to a new CSV file
sampled_df.to_csv(output_file, index=False)

print(f"Successfully sampled {num_rows} rows and saved to {output_file}") 