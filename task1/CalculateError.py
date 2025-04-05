import os
import re
import pandas as pd

folder = 'check'
base_folder = f'../../result/{folder}'
color_types = ['gray', 'Blues', 'hot', 'cubehelix', 'magma',
               'coolwarm', 'rainbow', 'spectral', 'blueyellow']

# Pattern to extract actual and predicted values
pattern = re.compile(r'(\d*\.\d+):\s*\(\d+, \d+\)\s*-\s*\[(\d*\.?\d*)\]')


def process_txt_file(file_path):
    """Process a single TXT file and return value differences"""
    with open(file_path, 'r', encoding='utf-8') as file:
        diffs = []
        for line in file:
            match = pattern.search(line)
            if match:
                actual = float(match.group(1))
                predicted = float(match.group(2))
                diffs.append(abs(actual - predicted))
        return diffs


def main():
    # Dictionary to store all differences data
    results = {color: {} for color in color_types}

    # Process each subject folder
    for subject in range(1, 11):
        folder_path = os.path.join(base_folder, str(subject))

        for color in color_types:
            file_path = os.path.join(folder_path, f'{color}_result.txt')
            if os.path.exists(file_path):
                diffs = process_txt_file(file_path)
                results[color][f'Subject {subject}'] = diffs

    # Ensure output directory exists
    output_dir = f'../../result/{folder}/data'
    os.makedirs(output_dir, exist_ok=True)

    # Save results to Excel
    output_file = os.path.join(output_dir, 'colormap_diff_results.xlsx')
    with pd.ExcelWriter(output_file) as writer:
        for color, data in results.items():
            if data:  # Only save if data exists
                max_len = max(len(v) for v in data.values())
                # Pad data to equal length
                padded_data = {k: v + [''] * (max_len - len(v)) for k, v in data.items()}
                pd.DataFrame(padded_data).to_excel(
                    writer,
                    sheet_name=color,
                    index=False
                )
                print(f"Saved {color} data ({len(data)} subjects)")

    print(f"All results saved to: {output_file}")


if __name__ == "__main__":
    main()