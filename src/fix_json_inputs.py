#!/usr/bin/env python3
"""
Fix JSON input files that are not in valid JSON format.
This script checks if each line in the input files is valid JSON and if not,
converts it to a proper JSON format with a 'Prompt' field.
"""

import json
import argparse
import os
import sys

def fix_json_file(input_file, output_file):
    """
    Fix each line in the input file to ensure it's valid JSON with a 'Prompt' field.
    
    Args:
        input_file (str): Path to the input file with raw or invalid JSON.
        output_file (str): Path to write the fixed JSON.
        
    Returns:
        tuple: (total_lines, fixed_lines) counts
    """
    total_lines = 0
    fixed_lines = 0
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line_num, line in enumerate(f_in, 1):
            total_lines += 1
            line = line.strip()
            
            try:
                # Try to parse as JSON
                data = json.loads(line)
                # Make sure it has a 'Prompt' field
                if 'Prompt' not in data:
                    data['Prompt'] = line
                    fixed_lines += 1
            except json.JSONDecodeError:
                # If not valid JSON, create a new JSON object
                data = {'Prompt': line}
                fixed_lines += 1
            
            # Write the fixed JSON
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    return total_lines, fixed_lines

def main():
    parser = argparse.ArgumentParser(description='Fix JSON input files')
    parser.add_argument('--input', required=True, help='Input file path')
    parser.add_argument('--output', required=True, help='Output file path')
    
    args = parser.parse_args()
    
    try:
        total, fixed = fix_json_file(args.input, args.output)
        print(f"Processed {total} lines from {args.input}")
        print(f"Fixed {fixed} lines in {args.output}")
    except Exception as e:
        print(f"Error processing file {args.input}: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 