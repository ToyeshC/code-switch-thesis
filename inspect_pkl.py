import pickle
import sys
import pandas as pd

def inspect_pkl(filepath):
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Successfully loaded: {filepath}")
        print(f"Data type: {type(data)}")
        
        if isinstance(data, list):
            print(f"Length of list: {len(data)}")
            if len(data) > 0:
                first_element = data[0]
                print(f"Type of first element: {type(first_element)}")
                if isinstance(first_element, dict):
                    print(f"Keys in first element dictionary: {list(first_element.keys())}")
                    print("\nContent of first element:")
                    for key, value in first_element.items():
                        print(f"  {key}: {repr(value)[:100]}...") # Print truncated value representation
                else:
                    print(f"First element content: {repr(first_element)[:200]}...")
            else:
                print("List is empty.")
        elif isinstance(data, pd.DataFrame):
            print(f"DataFrame shape: {data.shape}")
            print(f"DataFrame columns: {list(data.columns)}")
            print("\nFirst 2 rows of DataFrame:")
            print(data.head(2).to_string())
        elif isinstance(data, dict):
             print(f"Dictionary keys: {list(data.keys())}")
             # Optionally print some values if needed
        else:
            print(f"Data content (truncated): {repr(data)[:200]}...")
            
    except Exception as e:
        print(f"Error inspecting file {filepath}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_pkl.py <path_to_pkl_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    inspect_pkl(file_path) 