#!/usr/bin/env python3
"""
Fix for the inference.py script that addresses the issue with arrays of different lengths
by collecting all data first and then creating the DataFrame only once at the end.
"""

import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Fix the code-switched outputs from inference.py')
    parser.add_argument('--script_path', type=str, required=True, 
                        help='Path to the original inference.py script')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path where the fixed script will be saved')
    
    args = parser.parse_args()
    
    # Read the original script
    with open(args.script_path, 'r') as f:
        original_script = f.read()
    
    # Identify the part that needs to be fixed (the main function)
    start_marker = "def main(args):"
    end_marker = "if __name__ == \"__main__\":"
    
    # Split the script to isolate the part before and after the main function
    parts = original_script.split(start_marker)
    before_main = parts[0]
    
    parts = parts[1].split(end_marker)
    main_function = parts[0]
    after_main = end_marker + parts[1]
    
    # Fix the main function by making it write to CSV only once at the end
    # Note: We don't include "def main(args):" here since it's part of start_marker
    fixed_main = """
    # Keep most of the original code...
"""
    
    # Extract the initialization part of the main function up to the data dictionary creation
    data_dict_marker = "data = {"
    init_parts = main_function.split(data_dict_marker)
    fixed_main += init_parts[0]
    
    # Add the data dictionary creation
    fixed_main += data_dict_marker + init_parts[1].split("}")[0] + "}\n"
    fixed_main += """    for key in list(data.keys()):
        if not data[key]:
            del data[key]
    
    print("Baseline")
    if baseline_input_src:
        baseline_src = get_outputs(baseline_input_src, pipeline, terminators)
        data['baseline_src'] = baseline_src
    else:
        baseline_src = None
        
    if baseline_input_tgt:
        baseline_tgt = get_outputs(baseline_input_tgt, pipeline, terminators)
        data['baseline_tgt'] = baseline_tgt
    else:
        baseline_tgt = None
    
    print("Silver")
    if alignment_silver_src_translated:
        silver_src = get_outputs(alignment_silver_src_translated, pipeline, terminators)
        data['silver_src'] = silver_src
    else:
        silver_src = None
        
    if alignment_silver_tgt_translated:
        silver_tgt = get_outputs(alignment_silver_tgt_translated, pipeline, terminators)
        data['silver_tgt'] = silver_tgt
    else:
        silver_tgt = None

    print("Gold")
    if alignment_gold_src:
        gold_src = get_outputs(alignment_gold_src, pipeline, terminators)
        data['gold_src'] = gold_src
    else:
        gold_src = None
        
    if alignment_gold_tgt:
        gold_tgt = get_outputs(alignment_gold_tgt, pipeline, terminators)
        data['gold_tgt'] = gold_tgt
    else:
        gold_tgt = None
    
    print("Peek Eval")
    if ground_truth_src:
        gt_src = get_outputs(ground_truth_src, pipeline, terminators)
        data['gt_src'] = gt_src
    else:
        gt_src = None
        
    if ground_truth_tgt:
        gt_tgt = get_outputs(ground_truth_tgt, pipeline, terminators)
        data['gt_tgt'] = gt_tgt
    else:
        gt_tgt = None
        
    # Write the DataFrame to CSV only once at the end
    # First, check if any columns have different lengths and fix them
    max_length = 0
    for key, value in data.items():
        if value is not None:
            max_length = max(max_length, len(value))
    
    # Pad any shorter columns with None values
    for key, value in list(data.items()):
        if value is not None and len(value) < max_length:
            data[key] = value + [None] * (max_length - len(value))
    
    # Now we can safely create the DataFrame and write it to CSV
    pd.DataFrame(data).to_csv(args.output, index=False)
"""
    
    # Combine all parts to create the fixed script
    fixed_script = before_main + start_marker + fixed_main + after_main
    
    # Write the fixed script to the output path
    with open(args.output_path, 'w') as f:
        f.write(fixed_script)
    
    print(f"Fixed script has been written to {args.output_path}")
    print(f"Now you can run the fixed script instead of the original one.")

if __name__ == "__main__":
    main() 