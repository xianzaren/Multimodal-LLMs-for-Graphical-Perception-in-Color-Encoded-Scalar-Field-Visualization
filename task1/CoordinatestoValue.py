import os
import re
import pandas as pd


def clean_zero_width_spaces(text):
    """Remove zero-width characters from text"""
    return re.sub(r'[\u200B\u200C\u200D\uFEFF]', '', text)


def process_coordinates(txt_file, csv_mapping, output_txt, log_file):
    """Process coordinate data from text file and match with CSV values"""
    if not os.path.exists(txt_file):
        with open(log_file, 'a', encoding='utf-8') as log:
            log.write(f"Error: File not found - {txt_file}\n")
        return False

    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = [clean_zero_width_spaces(line.strip()) for line in f]

    weight_values = [1000, 750, 500, 250, 0]
    results = []

    for line in lines:
        match = re.match(r'^(.*?):\s*\[(.*)\]$', line)
        if not match:
            with open(log_file, 'a') as log:
                log.write(f"Error: Invalid line format - {line}\n")
            continue

        img_path, coords_str = match.groups()
        coords = re.findall(r'[\[\(](\d+),\s*(\d+)[\]\)]', coords_str)

        if len(coords) != len(weight_values):
            with open(log_file, 'a') as log:
                log.write(f"Error: Coordinate count mismatch - {line}\n")
            continue

        # Extract CSV key from image path
        key_match = re.search(r'\(\d+, \d+\)_\((\d+), \d+\)', img_path)
        if not key_match:
            results.append(f"{img_path} - Invalid format")
            continue

        csv_path = csv_mapping.get(key_match.group(1))
        if not csv_path or not os.path.exists(csv_path):
            results.extend(f"{w}: {c} - CSV Not Found" for w, c in zip(weight_values, coords))
            continue

        try:
            df = pd.read_csv(csv_path, header=None, skiprows=1)
        except Exception as e:
            with open(log_file, 'a') as log:
                log.write(f"CSV read error {csv_path}: {str(e)}\n")
            continue

        # Process each coordinate
        for weight, (x, y) in zip(weight_values, coords):
            x, y = int(x), int(y)
            if 0 <= x < len(df.columns) and 0 <= y < len(df):
                value = df.iloc[y, x]
                results.append(f"{weight}: ({x}, {y}) - [{value}]")
            else:
                results.append(f"{weight}: ({x}, {y}) - OutOfRange")

        results.append("")  # Add separator

    # Write results to file
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(results))

    return True


def main():
    base_dir = os.path.abspath("../../result/check")
    csv_dir = os.path.abspath("../../data/uncolor")
    shape = '(630, 820)'

    # CSV file mapping
    csv_mapping = {
        str(k): os.path.join(csv_dir, f"Noise_{shape}_({k}, {k})_5.csv")
        for k in [1, 3, 5, 7, 9]
    }

    # Setup error log
    log_file = os.path.abspath("../../error_log.txt")
    with open(log_file, 'w') as log:
        log.write("Processing Error Log\n")

    # Process each subject and colormap
    colormaps = ['gray', 'Blues', 'hot', 'cubehelix', 'extbodyheat',
                 'coolwarm', 'rainbow', 'spectral', 'blueyellow']

    for subject in range(1, 11):
        subject_dir = os.path.join(base_dir, str(subject))
        if not os.path.exists(subject_dir):
            continue

        for cmap in colormaps:
            input_file = os.path.join(subject_dir, f"{cmap}.txt")
            output_file = os.path.join(subject_dir, f"{cmap}_result.txt")

            if os.path.exists(input_file):
                success = process_coordinates(input_file, csv_mapping, output_file, log_file)
                if success and os.path.getsize(output_file) == 0:
                    print(f"Warning: Empty results in {output_file}")


if __name__ == "__main__":
    main()