import json
import argparse

def extract_prompts(input_json_path, output_text_path):
    try:
        with open(input_json_path, 'r', encoding='utf-8') as infile, \
             open(output_text_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                if line.strip():  # Ensure the line is not empty
                    try:
                        data = json.loads(line)
                        prompt = data.get("Prompt", "")
                        outfile.write(prompt + '\n')
                    except json.JSONDecodeError as e:
                        print(f"Skipping invalid JSON line: {e}")
        print(f"Prompts have been successfully extracted to {output_text_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Extract Prompts from JSON file to a plain text file.')
    parser.add_argument('input_json', help='Path to the input JSON file')
    parser.add_argument('output_text', help='Path to the output plain text file')
    args = parser.parse_args()
    
    extract_prompts(args.input_json, args.output_text)

if __name__ == "__main__":
    main()