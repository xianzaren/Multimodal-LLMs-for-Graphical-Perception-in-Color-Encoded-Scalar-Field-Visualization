import os
import re
import pandas as pd
from collections import defaultdict

filename = 'result.txt'

expected_files = {
    "gray": fr"../../color/oringinal test/texp2/gray/{filename}",
    "hot": fr"../../color/oringinal test/texp2/hot/{filename}",
    "rainbow": fr"../../color/oringinal test/texp2/rainbow/{filename}",
    "Blues": fr"../../color/oringinal test/texp2/Blues/{filename}",
    "blueyellow": fr"../../color/oringinal test/texp2/blueyellow/{filename}",
    "spectral": fr"../../color/oringinal test/texp2/spectral/{filename}",
    "magma": fr"../../color/oringinal test/texp2/magma/{filename}",
    "cubehelix": fr"../../color/oringinal test/texp2/cubehelix/{filename}",
    "coolwarm": fr"../../color/oringinal test/texp2/coolwarm/{filename}"
}

def parse_expected_results(file_path):
    """Parse expected result file and extract filename-color pairs"""
    results = {}
    if not os.path.exists(file_path):
        print(f"Missing expected result file: {file_path}")
        return results
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.match(r"ScalarField_WithBoxes_Noise_(.*)\.png: (\w+ Box Avg)", line)
            if match:
                filename = match.group(1)
                color = match.group(2).split()[0].lower()
                results[filename] = color
    return results

def parse_actual_results(file_path):
    """Parse actual experiment results file"""
    results = {}
    if not os.path.exists(file_path):
        print(f"Missing experiment file: {file_path}")
        return results
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            match = re.match(r".*Noise_((\(\d+, \d+\)_\(\d+, \d+\)_\d+_\w+_\d+))\.png:(\w+)", line)
            if match:
                filename = match.group(1)
                color = match.group(3).lower()
                results[filename] = color
    return results

def calculate_accuracy(actual_results, expected_results):
    """Calculate accuracy and return file-level correctness"""
    correct = 0
    total = 0
    file_accuracy_data = []
    for filename, expected_label in expected_results.items():
        if filename in actual_results:
            total += 1
            actual_label = actual_results[filename]
            is_correct = actual_label == expected_label
            file_accuracy_data.append({
                "File": filename,
                "Expected Color": expected_label,
                "Actual Color": actual_label,
                "Correct": is_correct
            })
            if is_correct:
                correct += 1
    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy, file_accuracy_data

def calculate_frequency_accuracy(actual_results, expected_results):
    """Calculate accuracy per frequency for each colormap"""
    frequency_accuracy_data = defaultdict(lambda: defaultdict(list))

    for filename, expected_label in expected_results.items():
        if filename in actual_results:
            actual_label = actual_results[filename]
            match = re.search(r"\((\d+), (\d+)\)_\((\d+), (\d+)\)", filename)
            if match:
                frequency = f"({match.group(3)}, {match.group(4)})"
                is_correct = actual_label == expected_label
                frequency_accuracy_data[expected_label][frequency].append(1 if is_correct else 0)

    frequency_avg_accuracy_data = []
    for colormap, frequencies in frequency_accuracy_data.items():
        for frequency, accuracies in frequencies.items():
            avg_accuracy = (sum(accuracies) / len(accuracies)) * 100 if accuracies else 0
            frequency_avg_accuracy_data.append({
                "Colormap": colormap,
                "Frequency": frequency,
                "Average Accuracy": avg_accuracy
            })
    return frequency_avg_accuracy_data

# Main execution
expected_results = {colormap: parse_expected_results(path) for colormap, path in expected_files.items()}
all_users_data = []
file_level_data = []
file_avg_accuracy = {}
folder = 'exp2'

for i in range(1, 11):  # Subjects 1-10
    user_folder = f"../../result/{folder}/{i}/"
    if not os.path.exists(user_folder):
        continue

    available_colormaps = [f.split(".txt")[0] for f in os.listdir(user_folder) if f.endswith(".txt")]
    for colormap in available_colormaps:
        if colormap not in expected_results:
            continue

        path = os.path.join(user_folder, f"{colormap}.txt")
        actual_results = parse_actual_results(path)
        accuracy, file_accuracy = calculate_accuracy(actual_results, expected_results[colormap])

        all_users_data.append({"User": i, "Colormap": colormap, "Accuracy": accuracy})

        for entry in file_accuracy:
            entry["User"] = i
            entry["Colormap"] = colormap
            file_level_data.append(entry)
            file_name = entry["File"]
            if file_name not in file_avg_accuracy:
                file_avg_accuracy[file_name] = []
            file_avg_accuracy[file_name].append(1 if entry["Correct"] else 0)

# Calculate file average accuracy
file_avg_accuracy_data = []
for file_name, accuracies in file_avg_accuracy.items():
    avg_accuracy = (sum(accuracies) / len(accuracies)) * 100 if accuracies else 0
    file_avg_accuracy_data.append({"File": file_name, "Average Accuracy": avg_accuracy})

# Calculate frequency accuracy
frequency_avg_accuracy_data = []
for i in range(1, 11):
    user_folder = f"../../result/{folder}/{i}/"
    if not os.path.exists(user_folder):
        continue

    available_colormaps = [f.split(".txt")[0] for f in os.listdir(user_folder) if f.endswith(".txt")]
    for colormap in available_colormaps:
        if colormap not in expected_results:
            continue

        path = os.path.join(user_folder, f"{colormap}.txt")
        actual_results = parse_actual_results(path)
        frequency_avg_accuracy_data.extend(calculate_frequency_accuracy(actual_results, expected_results[colormap]))

# Save Excel report
accuracy_report_path = os.path.join(f"../../result/{folder}/", "accuracy_report_with_frequency.xlsx")
with pd.ExcelWriter(accuracy_report_path) as writer:
    pd.DataFrame(all_users_data).to_excel(writer, sheet_name="All_Users_Accuracy", index=False)
    pd.DataFrame(file_level_data).to_excel(writer, sheet_name="File_Level_Accuracy", index=False)
    pd.DataFrame(file_avg_accuracy_data).to_excel(writer, sheet_name="File_Avg_Accuracy", index=False)
    pd.DataFrame(frequency_avg_accuracy_data).to_excel(writer, sheet_name="Frequency_Avg_Accuracy", index=False)

print(f"Report saved to: {accuracy_report_path}")