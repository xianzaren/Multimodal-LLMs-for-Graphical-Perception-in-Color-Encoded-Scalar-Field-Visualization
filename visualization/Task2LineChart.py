import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

# Read the Excel file
df = pd.read_excel(r'F:\submit data\experiment result\task2\ori\CoT\8b_ori_cot.xlsx', sheet_name='File_Level_Accuracy')

# Extract frequency (f1–f5) from the "File" column
df['Frequency'] = df['File'].str.extract(r'(f\d)')

# Convert "Correct" column to boolean
df['Correct'] = df['Correct'].astype(bool)

# Group by Frequency and Colormap
grouped = df.groupby(['Frequency', 'Colormap'])

# Calculate accuracy and Wilson confidence intervals
results = []
for (freq, cmap), group in grouped:
    n_total = len(group)
    n_correct = group['Correct'].sum()
    accuracy = n_correct / n_total
    ci_low, ci_upp = proportion_confint(count=n_correct, nobs=n_total, method='wilson')
    results.append({
        'Frequency': freq,
        'Colormap': cmap,
        'Accuracy': accuracy,
        '95% CI Lower': ci_low,
        '95% CI Upper': ci_upp
    })

results_df = pd.DataFrame(results)

# Map Frequency to numerical values
freq_mapping = {'f1': 1, 'f2': 3, 'f3': 5, 'f4': 7, 'f5': 9}
results_df['Freq_Num'] = results_df['Frequency'].map(freq_mapping)

# Plot
colors = ['#919191', '#57b4e9', '#d55e01', '#019e73', '#bd7ddf', '#ce0418', '#ffdb45', '#e79f01', '#5d63e1']
labels = ['greyscale', 'singlehue', 'bodyheat', 'cubehelix', 'extbodyheat', 'coolwarm', 'rainbow', 'spectral', 'blueyellow']
frequencies = [1, 2, 3, 4, 5]

fig, ax = plt.subplots(figsize=(6, 4))

for color, label in zip(colors, labels):
    cmap = label.lower()
    data = results_df[results_df['Colormap'].str.lower() == cmap].sort_values('Freq_Num')
    acc = data['Accuracy'].values * 100
    ci_lower = data['95% CI Lower'].values * 100
    ci_upper = data['95% CI Upper'].values * 100
    ax.fill_between(frequencies, ci_lower, ci_upper, color=color, alpha=0.05)
    ax.plot(frequencies, acc, marker='o', color=color, linestyle='-', label=label)

ax.set_xlabel('Spatial Frequency', fontsize=12)
ax.set_ylabel('Correct (%)', fontsize=12)
ax.set_xticks(frequencies)
ax.set_ylim(0, 105)
ax.set_yticks(range(10, 101, 10))
ax.legend(title="Color Map", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_facecolor('white')

plt.tight_layout()
line_chart_path = r'exp2_line.png'
plt.savefig(line_chart_path, dpi=300, bbox_inches='tight')
plt.close()
