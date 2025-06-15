import argparse
import pandas as pd
import os

def filter_by_column_value(input_file, output_file, column_name, value_to_keep):
    """
    Filters a CSV file to keep only rows where a specific column has a specific value.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the filtered CSV file.
        column_name (str): The name of the column to filter on.
        value_to_keep (str): The value to look for in the specified column.
    """
    try:
        print(f"Reading input file: {input_file}")
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return

    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in the input file.")
        print(f"Available columns are: {list(df.columns)}")
        return

    original_count = len(df)
    print(f"Original number of rows: {original_count}")

    # Filter the DataFrame using a case-insensitive match
    filtered_df = df[df[column_name].str.lower() == value_to_keep.lower()].copy()

    filtered_count = len(filtered_df)
    print(f"Number of rows with '{column_name}' as '{value_to_keep}': {filtered_count}")
    
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving {filtered_count} filtered rows to: {output_file}")
    filtered_df.to_csv(output_file, index=False)
    
    removed_count = original_count - filtered_count
    print(f"Removed {removed_count} rows.")
    print("Filtering complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter a CSV file based on a specific column value.")
    parser.add_argument("--input_file", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_file", required=True, help="Path to the output CSV file.")
    parser.add_argument("--column", required=True, help="The column to filter on (e.g., 'sentiment').")
    parser.add_argument("--value", required=True, help="The value to keep in the specified column (e.g., 'negative').")

    args = parser.parse_args()
    filter_by_column_value(args.input_file, args.output_file, args.column, args.value) 