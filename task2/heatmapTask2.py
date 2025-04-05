import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.colors import PowerNorm


prompt_kind = ['baseline', 'cot']
module_name = ['gpt', 'gemini', 'glm', '8b', '40b']
colormaps = ['extbodyheat', 'greyscale', 'singlehue', 'bodyheat', 'cubehelix', 'coolwarm', 'rainbow', 'spectral', 'blueyellow']


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


def process_step1():
    model_results = {}
    csv_columns =['f1', 'f2']
    for model in module_name:
        filename = f"{model}.csv"
        excel_path = os.path.join('../../result_excel/exp2_step1', filename)
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
    model_results = {}
    for model in module_name:
        if model != 'FT+CoT':
            filename = f"{model}_ori.xlsx"
        else:
            filename = f"8b_linkGT.xlsx"
        excel_path = os.path.join('../../result_excel/exp2_linking', filename)
        if not os.path.exists(excel_path):
            continue
        file_avg_accuracy_df = pd.read_excel(excel_path, sheet_name='File_Avg_Accuracy')

        acc_values = []
        for cmap in colormaps:
            cmap_data = file_avg_accuracy_df[file_avg_accuracy_df['File'].str.contains(cmap)]
            mean_accuracy = cmap_data['Average Accuracy'].mean()
            mean_accuracy = 100 - mean_accuracy
            acc_values.append(mean_accuracy)
        model_mean = np.nanmean(acc_values) if acc_values else np.nan
        model_results[model] = model_mean

    return model_results


def process_baseline():
    module_name = ['gpt', 'gemini', 'glm', '8b', '40b', 'FT+Baseline', 'FT+CoT']
    model_results = {}
    for model in module_name:
        if model != 'FT+Baseline' and model != 'FT+CoT':
            filename = f"{model}_ori_baseline.xlsx"
            excel_path = os.path.join('../../result_excel/exp2', filename)
        elif model == 'FT+Baseline':
            filename = f"8b_baseGT_baseline.xlsx"
            excel_path = os.path.join('../../result_excel/exp2', filename)
        elif model == 'FT+CoT':
            filename = f"8b_cotGT_baseline.xlsx"
            excel_path = os.path.join('../../result_excel/exp2', filename)

        if not os.path.exists(excel_path):
            continue
        file_avg_accuracy_df = pd.read_excel(excel_path, sheet_name='File_Avg_Accuracy')
        acc_values = []
        for cmap in colormaps:
            cmap_data = file_avg_accuracy_df[file_avg_accuracy_df['File'].str.contains(cmap)]
            mean_accuracy = cmap_data['Average Accuracy'].mean()
            mean_accuracy = 100 - mean_accuracy
            acc_values.append(mean_accuracy)
        model_mean = np.nanmean(acc_values) if acc_values else np.nan
        model_results[model] = model_mean

    return model_results


def process_cot():
    module_name = ['gpt', 'gemini', 'glm', '8b', '40b', 'FT+Baseline', 'FT+CoT']
    model_results = {}
    for model in module_name:
        if model != 'FT+Baseline' and model != 'FT+CoT':
            filename = f"{model}_ori_cot.xlsx"
            excel_path = os.path.join('../../result_excel/exp2', filename)
        elif model == 'FT+Baseline':
            filename = f"8b_baseGT_cot.xlsx"
            excel_path = os.path.join('../../result_excel/exp2', filename)
        elif model == 'FT+CoT':
            filename = f"8b_cotGT_cot.xlsx"
            excel_path = os.path.join('../../result_excel/exp2', filename)

        if not os.path.exists(excel_path):
            continue
        file_avg_accuracy_df = pd.read_excel(excel_path, sheet_name='File_Avg_Accuracy')
        acc_values = []
        for cmap in colormaps:
            cmap_data = file_avg_accuracy_df[file_avg_accuracy_df['File'].str.contains(cmap)]
            mean_accuracy = cmap_data['Average Accuracy'].mean()
            mean_accuracy = 100 - mean_accuracy
            acc_values.append(mean_accuracy)
        model_mean = np.nanmean(acc_values) if acc_values else np.nan
        model_results[model] = model_mean

    return model_results


def plot(step1, step2, baseline, cot):
    data = np.full((4, 7), np.nan)
    model_to_col = {
        'gpt': 0,
        'gemini': 1,
        'glm': 2,
        '8b': 3,
        '40b': 4,
        'FT+Baseline': 5,
        'FT+CoT': 6
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
    col_labels = ['GPT', 'Gemini', 'GLM', 'Intern8B', 'Intern40B', 'FT+Baseline', 'FT+CoT']

    mask = np.isnan(data)

    plt.figure(figsize=(8, 5))
    data1 = data[0:1]
    valid_data = data1[~np.isnan(data1)]
    norm = PowerNorm(gamma=1.0, vmin=np.min(valid_data), vmax=np.max(valid_data))
    ax1 = sns.heatmap(data[0:1],square=True,annot=True,fmt='.2f',cmap='Reds',cbar=True,
                      linewidths=10,linecolor='white',yticklabels=[row_labels[0]],xticklabels=[],
                      mask=mask[0:1],vmin=2.0,vmax=5.0,annot_kws={"size": 10},
                      norm=norm,alpha=0.9,
                      cbar_kws={'shrink': 0.2, 'aspect': 4, 'pad': 0.02}
                      )
    ax1.set_facecolor('white')
    ax1.tick_params(length=0, labelsize=12)
    plt.tight_layout()
    plt.savefig('../../PAPER/heatmap/Task2_Step1.png', dpi=500, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8, 5))
    data2 = data[1:]
    valid_data = data2[~np.isnan(data2)]
    norm = PowerNorm(gamma=0.9, vmin=np.min(valid_data), vmax=np.max(valid_data))
    ax2 = sns.heatmap(data2,square=True,annot=True,fmt='.2f',cmap='Blues',cbar=True,
                      linewidths=10,linecolor='white',
                      yticklabels=row_labels[1:],mask=mask[1:],
                      annot_kws={"size": 10},norm=norm,alpha=0.9,
                      cbar_kws={'shrink': 0.55, 'aspect': 10, 'pad': 0.02}
                      )

    ax2.set_facecolor('white')
    ax2.set_xticklabels(col_labels, rotation=0,fontsize=9)
    ax2.tick_params(length=0)
    plt.tight_layout()
    plt.savefig('../../PAPER/heatmap/Task2.png', dpi=500, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    step1_results = process_step1()
    step2_results = process_step2()
    baseline_results = process_baseline()
    cot_results = process_cot()
    plot(step1_results, step2_results, baseline_results, cot_results)
    # print(step1_results)
    # print(step2_results)
    # print(baseline_results)
    # print(cot_results)

