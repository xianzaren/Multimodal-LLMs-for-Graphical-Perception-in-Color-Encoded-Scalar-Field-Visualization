import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import csv
import os

def save_colormap_rgb_range(cmap, save_path, min_value, max_value):
    if cmap == 'blueyellow':
        cmap = blueyellow
    elif cmap == 'spectral':
        cmap = plt.get_cmap('nipy_spectral')
    elif cmap == 'Blues':
        cmap = plt.get_cmap('Blues_r')
    else:
        cmap = plt.get_cmap(cmap)
    # 保存 RGB 数据到 CSV 文件
    with open(save_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Value", "R", "G", "B"])  # 写入表头
        # 遍历 min_value 到 max_value 的所有值
        for value in range(min_value, max_value + 1):
            # 归一化值到 [0, 1] 范围
            normalized_value = value / max_value
            # 获取 RGB 值（cmap 返回 RGBA，取前 3 个值）
            rgba = cmap(normalized_value)
            rgb = tuple(int(c * 255) for c in rgba[:3])  # 转换为 0-255 的整数值
            csv_writer.writerow([value, rgb[0], rgb[1], rgb[2]])  # 写入 RGB 数据

    print(f"Colormap RGB 数据已保存到 {save_path}")

def create_blueyellow_colormap():
    blue = (13 / 255.0, 0 / 255.0, 252 / 255.0)
    yellow = (252 / 255.0, 252 / 255.0, 0 / 255.0)
    return mcolors.LinearSegmentedColormap.from_list("blueyellow", [blue, yellow], N=256)

blueyellow = create_blueyellow_colormap()

# 定义 colormap 列表
colormap_list = ['gray', 'Blues', 'hot', 'cubehelix', 'magma', 'coolwarm', 'rainbow', 'spectral', 'blueyellow']

# 为每个 colormap 保存对应的 RGB 数据到 CSV 文件
for colormap in colormap_list:
    savepath = f'{colormap}_rgb.csv'
    save_colormap_rgb_range(colormap, savepath, 0, 1000)
