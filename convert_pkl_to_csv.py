import pickle
import pandas as pd

# Read the pickle file
with open('ezswitch/data/hinge/train_human_generated.pkl', 'rb') as f:
    data = pickle.load(f)

# Convert dictionary to DataFrame
df = pd.DataFrame.from_dict(data, orient='index')

# Save as CSV
output_path = 'ezswitch/data/hinge/train_human_generated.csv'
df.to_csv(output_path, index=True)
print(f"CSV file has been saved to: {output_path}") 