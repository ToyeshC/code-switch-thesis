#!/usr/bin/env python3
"""
Extract 'Prompt' field from JSON input files.

This script processes input files containing JSON objects on each line
and extracts the 'Prompt' field from each JSON object. If a line is not
valid JSON or does not contain a 'Prompt' field, it is written as is.
"""

import json
import argparse
import os
import sys

def extract_prompts(input_file, output_file, placeholder="[INVALID_PROMPT]"):
    """
    Extract 'Prompt' field from each line of JSON in the input file.
    
    Args:
        input_file (str): Path to the input file containing JSON objects.
        output_file (str): Path to write the extracted prompts.
        placeholder (str): Text to use when a prompt can't be extracted.
        
    Returns:
        tuple: (total_lines, extracted_prompts) counts
    """
    total_lines = 0
    extracted_prompts = 0
    skipped_lines = 0
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line_num, line in enumerate(f_in, 1):
            total_lines += 1
            try:
                # Try to parse as JSON
                data = json.loads(line.strip())
                if 'Prompt' in data:
                    f_out.write(data['Prompt'] + '\n')
                    extracted_prompts += 1
                else:
                    # If 'Prompt' field is not found, use placeholder
                    f_out.write(placeholder + '\n')
                    skipped_lines += 1
                    print(f"Warning: Line {line_num} in {input_file} contains JSON but no 'Prompt' field")
            except json.JSONDecodeError:
                # If not valid JSON, use placeholder
                f_out.write(placeholder + '\n')
                skipped_lines += 1
                print(f"Warning: Line {line_num} in {input_file} is not valid JSON")
    
    return total_lines, extracted_prompts, skipped_lines

def main():
    parser = argparse.ArgumentParser(description='Extract Prompt field from JSON input files')
    parser.add_argument('--input', required=True, help='Input file path')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--placeholder', default='[INVALID_PROMPT]', 
                        help='Text to use for invalid/missing prompts')
    
    args = parser.parse_args()
    
    try:
        total, extracted, skipped = extract_prompts(args.input, args.output, args.placeholder)
        print(f"Processed {total} lines from {args.input}")
        print(f"Extracted {extracted} prompts to {args.output}")
        if skipped > 0:
            print(f"Warning: {skipped} lines had no valid 'Prompt' field and were replaced with placeholder")
    except Exception as e:
        print(f"Error processing file {args.input}: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 