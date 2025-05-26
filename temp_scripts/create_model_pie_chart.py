import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
input_file = 'temp_scripts/zzzz.csv'
output_file = 'temp_scripts/model_distribution_pie.png'

# Read the CSV file
df = pd.read_csv(input_file)

# Calculate the percentage of each model
model_counts = df['model'].value_counts()
model_percentages = (model_counts / len(df) * 100).round(2)

# Create a pie chart
plt.figure(figsize=(10, 8))
plt.pie(model_percentages, labels=model_percentages.index, autopct='%1.1f%%')
plt.title('Distribution of Models in Sampled Data')

# Save the plot
plt.savefig(output_file, bbox_inches='tight', dpi=300)
plt.close()

print(f"Pie chart has been saved to {output_file}")

# Print the exact percentages
print("\nExact percentages:")
for model, percentage in model_percentages.items():
    print(f"{model}: {percentage}%") 