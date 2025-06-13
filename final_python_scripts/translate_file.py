import argparse
import asyncio
import os
from googletrans import Translator

async def main():
    # Set up command-line argument parsing.
    parser = argparse.ArgumentParser(
        description="Translate a text file line by line using the googletrans Python module."
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help="Path to the input text file containing the sentences to translate."
    )
    parser.add_argument(
        '--target', type=str, required=True,
        help="Target language code (e.g., 'hi' for Hindi)."
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help="Path to the output file where the translated sentences will be written."
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize the googletrans Translator.
    translator = Translator()
    
    # Read the input file.
    with open(args.input, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    translated_lines = []
    for line in lines:
        line = line.strip()
        if line:
            # Translate the line.
            try:
                # Await the coroutine for the translation
                translation = await translator.translate(line, dest=args.target)
                translated_lines.append(translation.text)
            except Exception as err:
                print(f"Error translating line: {line}\nError: {err}")
                translated_lines.append(line)  # Fallback: use original line
        else:
            # Preserve empty lines.
            translated_lines.append('')

    # Write the translated content to the output file.
    with open(args.output, 'w', encoding='utf-8') as outfile:
        for translated_line in translated_lines:
            outfile.write(translated_line + "\n")

if __name__ == "__main__":
    asyncio.run(main())
