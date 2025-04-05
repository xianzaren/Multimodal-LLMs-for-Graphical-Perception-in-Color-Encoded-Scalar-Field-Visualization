import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Settings
file_path = r'filepath'
labels = ['greyscale', 'singlehue', 'bodyheat', 'cubehelix', 'extbodyheat', 'coolwarm', 'rainbow', 'spectral', 'blueyellow']
colors = ['#919191', '#57b4e9', '#d55e01', '#019e73', '#bd7ddf', '#ce0418', '#ffdb45', '#e79f01', '#5d63e1']
frequencies = [1, 3, 5, 7, 9]

all_results = []

# 2. Read each sheet and process
for label in labels:
    df = pd.read_excel(file_path, sheet_name=label)
    df = df.dropna(how='all').reset_index(drop=True)

    for i in range(5):
        part = df.iloc[i * 5:(i + 1) * 5, :]
        values = part.values.flatten()
        values = values[~np.isnan(values)]

        # Transform data
        transformed_values = np.log2((values * 100) + (1 / 8))

        mean = np.mean(transformed_values)
        std = np.std(transformed_values, ddof=1)
        n = len(transformed_values)

        # 95% confidence interval
        ci_half_width = 1.96 * (std / np.sqrt(n))
        ci_low = mean - ci_half_width
        ci_upp = mean + ci_half_width

        all_results.append({
            'Frequency': f'f{i + 1}',
            'Freq_Num': i + 1,
            'Colormap': label,
            'Mean': mean,
            '95% CI Lower': ci_low,
            '95% CI Upper': ci_upp
        })

# 3. Organize results
results_df = pd.DataFrame(all_results)

# 4. Plot
fig, ax = plt.subplots(figsize=(6, 4))

for color, label in zip(colors, labels):
    data = results_df[results_df['Colormap'] == label].sort_values('Freq_Num')

    mean = data['Mean'].values
    ci_lower = data['95% CI Lower'].values
    ci_upper = data['95% CI Upper'].values

    ax.fill_between(frequencies, ci_lower, ci_upper, color=color, alpha=0.25)
    ax.plot(frequencies, mean, marker='o', color=color, linestyle='-', label=label)

ax.set_xlabel('Spatial Frequency', fontsize=12)
ax.set_ylabel('Transformed Mean (log2)', fontsize=12)
ax.set_xticks(frequencies)

# ax.set_ylim(lower_bound, upper_bound)  # Uncomment if needed

ax.legend(title="Color Map", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_facecolor('white')

plt.tight_layout()

line_chart_path = r'exp1_line.png'
plt.savefig(line_chart_path, dpi=300, bbox_inches='tight')
plt.close()
