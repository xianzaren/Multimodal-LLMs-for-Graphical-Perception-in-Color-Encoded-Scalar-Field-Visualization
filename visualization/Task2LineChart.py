import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

# 【1】读取原始数据并计算准确率与置信区间
# ===========================================
# 读取Excel文件中的特定Sheet
df = pd.read_excel(r'filepath', sheet_name='File_Level_Accuracy')

# 从File字段提取Frequency（比如f1, f2, ..., f5）
df['Frequency'] = df['File'].str.extract(r'(f\d)')

# 把Correct列统一为布尔类型
df['Correct'] = df['Correct'].astype(bool)

# 按 Frequency 和 Colormap 分组
grouped = df.groupby(['Frequency', 'Colormap'])

# 计算准确率和置信区间
results = []
for (freq, cmap), group in grouped:
    n_total = len(group)
    n_correct = group['Correct'].sum()
    accuracy = n_correct / n_total

    # Wilson置信区间 (95%)
    ci_low, ci_upp = proportion_confint(count=n_correct, nobs=n_total, method='wilson')

    results.append({
        'Frequency': freq,
        'Colormap': cmap,
        'Accuracy': accuracy,
        '95% CI Lower': ci_low,
        '95% CI Upper': ci_upp
    })

# 转成DataFrame
results_df = pd.DataFrame(results)

# （可选）保存中间计算结果
# results_df.to_excel('E:/submit data/experiment result/task2/ori/CoT/accuracy_results.xlsx', index=False)

# 【2】绘制折线图
# ===========================================

# 将Frequency映射成数字（方便画X轴）
freq_mapping = {'f1': 1, 'f2': 3, 'f3': 5, 'f4': 7, 'f5': 9}
results_df['Freq_Num'] = results_df['Frequency'].map(freq_mapping)

# 颜色和标签
colors = ['#919191', '#57b4e9', '#d55e01', '#019e73', '#bd7ddf', '#ce0418', '#ffdb45', '#e79f01', '#5d63e1']
labels = ['greyscale', 'singlehue', 'bodyheat', 'cubehelix', 'extbodyheat', 'coolwarm', 'rainbow', 'spectral',
          'blueyellow']

# frequencies横坐标
frequencies = [1, 2, 3, 4, 5]

# 创建图
fig, ax = plt.subplots(figsize=(6, 4))

# 遍历每个colormap
for color, label in zip(colors, labels):
    cmap = label.lower()

    # 选出当前colormap的数据
    data = results_df[results_df['Colormap'].str.lower() == cmap]

    # 按频率排序
    data = data.sort_values('Freq_Num')

    acc = data['Accuracy'].values * 100  # 转成百分比
    ci_lower = data['95% CI Lower'].values * 100
    ci_upper = data['95% CI Upper'].values * 100

    # 填充置信区间
    ax.fill_between(frequencies, ci_lower, ci_upper, color=color, alpha=0.05)

    # 绘制准确率折线
    ax.plot(frequencies, acc, marker='o', color=color, linestyle='-', label=label)

# 设置X轴、Y轴
ax.set_xlabel('Spatial Frequency', fontsize=12)
ax.set_ylabel('Correct (%)', fontsize=12)
ax.set_xticks(frequencies)
ax.set_ylim(0, 105)
ax.set_yticks(range(10, 101, 10))

# 添加图例
ax.legend(title="Color Map", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

# 设置网格
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_facecolor('white')

# 可选：添加黑色边框
# for spine in ax.spines.values():
#     spine.set_linewidth(1.5)
#     spine.set_color('black')

# 调整布局
plt.tight_layout()

# 保存图像
line_chart_path = r'exp2_line.png'
plt.savefig(line_chart_path, dpi=300, bbox_inches='tight')
plt.close()
