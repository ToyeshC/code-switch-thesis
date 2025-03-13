import pandas as pd

def extract_generated_sentences(input_file, output_file):
    # Read the original CSV file
    df = pd.read_csv(input_file)
    
    # Extract the 'generated' column
    generated_sentences = df['generated']
    
    # Create a new DataFrame with just the generated sentences
    generated_df = pd.DataFrame(generated_sentences)
    
    # Save the new DataFrame to a new CSV file
    generated_df.to_csv(output_file, index=False, header=False)
    print(f"Generated sentences saved to {output_file}")

if __name__ == "__main__":
    input_file = 'data/output/compile_english.csv'
    output_file = 'data/output/generated_sentences_english.csv'
    extract_generated_sentences(input_file, output_file) 