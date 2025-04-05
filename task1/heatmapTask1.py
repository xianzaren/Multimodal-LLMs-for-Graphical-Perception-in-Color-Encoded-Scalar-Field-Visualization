import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from matplotlib.colors import PowerNorm


prompt_kind = ['baseline', 'cot']
module_name = ['gpt', 'gemini', 'glm', '8b', '40b']


def remove_iqr(data, k=1.5):
    data = np.array(data, dtype=float)
    Q1 = np.nanpercentile(data, 25)
    Q3 = np.nanpercentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    mask = (data >= lower_bound) & (data <= upper_bound)
    filtered = data[mask]
    return filtered if len(filtered) > 0 else data


def logerror1000(judged_percent):
    return np.abs(judged_percent) * 0.1

def logerror1(judged_percent):
    return np.abs(judged_percent) * 100

# def logerror1000(judged_percent):
#     return np.log2(np.abs(judged_percent) * 0.1 + 1 / 8)
#
# def logerror1(judged_percent):
#     return np.log2(np.abs(judged_percent) * 100 + 1 / 8)


def process_step1():
    model_results = {}
    csv_columns =['f1', 'f2', 'f3', 'f4', 'f5']
    for model in module_name:
        filename = f"{model}.csv"
        excel_path = os.path.join('../../result_excel/exp1_step1', filename)
        if not os.path.exists(excel_path):
            continue
        df = pd.read_csv(excel_path)
        column_means = []
        for column in csv_columns:
            column_data = df[column].dropna()
            filter_data = remove_iqr(column_data)
            column_mean = np.mean(filter_data)
            column_means.append(column_mean)
        average_result = np.mean(column_means)
        log_result = logerror1000(average_result)
        model_results[model] = log_result

    return model_results


def process_step2():
    module_name = ['gpt', 'gemini', 'glm', '8b', '40b', 'FT+CoT']
    frequency_rows = {
        1: (0, 5), 3: (6, 11), 5: (12, 17),
        7: (18, 23), 9: (24, 29)
    }
    model_results = {}
    for model in module_name:
        if model != 'FT+CoT':
            filename = f"{model}_ori.xlsx"
        else:
            filename = f"8b_linkGT.xlsx"
        excel_path = os.path.join('../../result_excel/exp1_linking', filename)
        if not os.path.exists(excel_path):
            continue
        df = pd.read_excel(excel_path, sheet_name=None, header=0)
        error_values = []
        for cmap_name, data in df.items():
            for freq, (start, end) in frequency_rows.items():
                data_block = data.iloc[start:end + 1, 0:].values
                if data_block.size == 0:
                    continue

                row_means = []
                for row in data_block:
                    filtered_row = remove_iqr(row, k=1.5)
                    row_mean = np.nanmean(filtered_row)
                    row_means.append(row_mean)

                frequency_mean = np.nanmean(row_means)
                transformed_error = logerror1(frequency_mean)
                error_values.append(transformed_error)

        model_mean = np.nanmean(error_values) if error_values else np.nan
        model_results[model] = model_mean

    return model_results


def process_baseline():
    module_name = ['gpt', 'gemini', 'glm', '8b', '40b', 'FT+Baseline', 'FT+CoT']
    #module_name = ['FT+Baseline', 'FT+CoT']
    frequency_rows = {
        1: (0, 5), 3: (6, 11), 5: (12, 17),
        7: (18, 23), 9: (24, 29)
    }
    model_results = {}
    for model in module_name:
        if model != 'FT+Baseline' and model != 'FT+CoT':
            filename = f"{model}_ori_baseline.xlsx"
            excel_path = os.path.join('../../result_excel/exp1/ori', filename)
        elif model == 'FT+Baseline':
            filename = f"8b_baseGT_baseline.xlsx"
            excel_path = os.path.join('../../result_excel/exp1', filename)
        elif model == 'FT+CoT':
            filename = f"8b_cotGT_baseline.xlsx"
            excel_path = os.path.join('../../result_excel/exp1', filename)

        if not os.path.exists(excel_path):
            continue
        df = pd.read_excel(excel_path, sheet_name=None, header=0)
        error_values = []
        for cmap_name, data in df.items():
            for freq, (start, end) in frequency_rows.items():
                data_block = data.iloc[start:end + 1, 0:].values
                if data_block.size == 0:
                    continue

                row_means = []
                for row in data_block:
                    filtered_row = remove_iqr(row, k=1.5)
                    row_mean = np.nanmean(filtered_row)
                    #row_mean = np.nanmean(row)
                    row_means.append(row_mean)

                frequency_mean = np.nanmean(row_means)
                # error_values.append(frequency_mean)
                transformed_error = logerror1(frequency_mean)
                error_values.append(transformed_error)

        model_mean = np.nanmean(error_values) if error_values else np.nan
        model_results[model] = model_mean

    return model_results


def process_cot():
    module_name = ['gpt', 'gemini', 'glm', '8b', '40b', 'FT+Baseline', 'FT+CoT']
    frequency_rows = {
        1: (0, 5), 3: (6, 11), 5: (12, 17),
        7: (18, 23), 9: (24, 29)
    }
    model_results = {}
    for model in module_name:
        if model != 'FT+Baseline' and model != 'FT+CoT':
            filename = f"{model}_ori_cot.xlsx"
            excel_path = os.path.join('../../result_excel/exp1/ori', filename)
        elif model == 'FT+Baseline':
            filename = f"8b_baseGT_cot.xlsx"
            excel_path = os.path.join('../../result_excel/exp1', filename)
        elif model == 'FT+CoT':
            filename = f"8b_cotGT_cot.xlsx"
            excel_path = os.path.join('../../result_excel/exp1', filename)

        if not os.path.exists(excel_path):
            continue
        df = pd.read_excel(excel_path, sheet_name=None, header=0)
        error_values = []
        for cmap_name, data in df.items():
            for freq, (start, end) in frequency_rows.items():
                data_block = data.iloc[start:end + 1, 0:].values
                if data_block.size == 0:
                    continue

                row_means = []
                for row in data_block:
                    filtered_row = remove_iqr(row, k=1.5)
                    row_mean = np.nanmean(filtered_row)
                    row_means.append(row_mean)

                frequency_mean = np.nanmean(row_means)
                transformed_error = logerror1(frequency_mean)
                error_values.append(transformed_error)
        model_mean = np.nanmean(error_values) if error_values else np.nan
        model_results[model] = model_mean

    return model_results


def plot(step1, step2, baseline, cot):
    data = np.full((4, 6), np.nan)
    model_to_col = {
        'gpt': 0,
        'gemini': 1,
        '8b': 2,
        '40b': 3,
        'FT+Baseline': 4,
        'FT+CoT': 5
    }
    for model, value in step1.items():
        if model in model_to_col:
            data[0, model_to_col[model]] = value

    for model, value in step2.items():
        if model in model_to_col:
            data[1, model_to_col[model]] = value

    for model, value in baseline.items():
        if model in model_to_col:
            data[2, model_to_col[model]] = value

    for model, value in cot.items():
        if model in model_to_col:
            data[3, model_to_col[model]] = value

    row_labels = ['Step 1', 'Step 2', 'Baseline', 'CoT']
    col_labels = ['GPT', 'Gemini', 'Intern8B', 'Intern40B', 'FT+Baseline', 'FT+CoT']

    mask = np.isnan(data)
    valid_data = data[~np.isnan(data)]
    norm = PowerNorm(gamma=0.9, vmin=np.min(valid_data), vmax=np.max(valid_data))
    if len(valid_data) > 0:
        plt.clf()
        plt.figure(figsize=(7, 4))
        with sns.axes_style("white"):
            ax = sns.heatmap(data, square=True, annot=True, fmt='.2f', cmap='Blues', cbar=True,
                             linewidths=10, linecolor='white', xticklabels=col_labels, yticklabels=row_labels,
                             cbar_kws={'label': 'Error'}, mask=mask,
                             annot_kws={"size":10},
                             norm=norm,
                             alpha=0.9
                             )
        ax.set_facecolor('white')

        # # 添加额外的间距 - 在Step 1和Step 2之间
        # for i, label in enumerate(row_labels):
        #     if label == 'Step 1':
        #         ax.axhline(y=i + 1, color='white', linewidth=20)
        #
        # # 增加'Intern40B'和'FT+Baseline'之间的间距
        # for i, label in enumerate(col_labels):
        #     if label == 'Intern40B':
        #         ax.axvline(x=i + 1, color='white', linewidth=20)

        plt.setp(ax.get_xticklabels(), ha="center")
        plt.setp(ax.get_yticklabels(), va="center")

    ax.tick_params(length=0)
    plt.tight_layout()
    plt.savefig('../../PAPER/heatmap/Task1.png', dpi=500, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    step1_results = process_step1()
    step2_results = process_step2()
    baseline_results = process_baseline()
    cot_results = process_cot()
    plot(step1_results, step2_results, baseline_results, cot_results)
    print(step1_results)
    print(step2_results)
    print(baseline_results)
    print(cot_results)
