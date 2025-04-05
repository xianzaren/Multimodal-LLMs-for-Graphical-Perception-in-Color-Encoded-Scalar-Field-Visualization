import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap, Normalize


def load_noise_from_csv(filepath):
    return np.loadtxt(filepath, delimiter=",", skiprows=1)


def plot_scalar_field_with_density_map(shape, noise, noise2, colormap, filepath, marker_value, black_star_coords,
                                       dcmap):
    dpi = 100
    figsize_height = shape[0] / dpi
    figsize_width = 767 / dpi
    cmap = plt.get_cmap('Blues_r')

    # Step 1: Draw the noise image and save as p1.png (base layer)
    fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
    cax = ax.imshow(noise, cmap=cmap, origin='upper', vmin=0, vmax=1000, zorder=1)
    cbar = plt.colorbar(cax, fraction=0.030, pad=0.05)
    cbar.ax.set_facecolor('white')  # Set colorbar background to white
    cbar.outline.set_edgecolor('black')
    tick_positions = np.linspace(0, 1000, 9)
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels([f"{t:.0f}" for t in tick_positions])
    plt.axis('off')
    plt.savefig('../output/p1.png', dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Step 2: Reload p1.png and overlay additional layers
    fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
    ax.imshow(plt.imread('../output/p1.png'), zorder=1)
    black_overlay = np.zeros((*noise.shape, 4))
    black_overlay[..., 3] = 0.4  # Set alpha transparency
    ax.imshow(black_overlay, extent=[0, noise.shape[1], noise.shape[0], 0], zorder=2)

    # Step 3: Plot marker points (middle layer)
    marker_positions = np.argwhere(noise2 == marker_value)
    ax.scatter(
        marker_positions[:, 1],
        marker_positions[:, 0],
        color=(1.0, 1.0, 0.6),
        s=20,
        marker='o',
        alpha=0.8,
        linewidth=0,
        zorder=3
    )

    # Step 4: Plot density map (top layer)
    shape = (630, 987)
    black_star_coords = np.array(black_star_coords)
    kde = gaussian_kde(black_star_coords.T, bw_method=5)
    x_grid, y_grid = np.meshgrid(np.linspace(0, shape[1] - 1, shape[1]), np.linspace(0, shape[0] - 1, shape[0]))
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    density = np.reshape(kde(positions).T, x_grid.shape)

    intervals = np.arange(0, 1.0, 0.2)
    for i in range(len(intervals) - 1):
        masked_density = np.ma.masked_outside(density, intervals[i], intervals[i + 1])
        ax.imshow(
            masked_density,
            cmap=dcmap,
            alpha=min(1, 0.6 + i / 2),
            origin='upper',
            extent=[0, shape[1], shape[0], 0],
            zorder=3
        )

    plt.axis('off')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()


def save_noise_to_csv(noise, filepath):
    np.savetxt(filepath, noise.astype(np.float64), delimiter=",", fmt="%.16f", header="Normalized Elevation Data",
               comments="")
    print(f"CSV file saved: {filepath}")


def main():
    shape = (630, 820)
    cmap = "coolwarm"
    marker_value = 750

    colormap_list = ['gray']
    newcolormap_list = []
    for colormap in colormap_list:
        viridis_cmap = plt.get_cmap(colormap)
        colors = viridis_cmap(np.linspace(0, 1, 256))
        custom_cmap = LinearSegmentedColormap.from_list(f'{colormap}_with_transparency', colors)
        newcolormap_list.append(custom_cmap)
    # frequency:1 value:750
    points = [
        # GPT cot
        (530, 220), (494, 248), (515, 99), (474, 208), (530, 145),(521, 341), (501, 134), (529, 226), (585, 322), (470, 80)
        # GPT baseline remove 3 x [0, 0]
        # (611, 564), (380, 450), (839, 379), (727, 443), (553, 378), (50, 500), (0, 550)
        # 8B baseline
        # (350, 250),(510, 370),(650, 100),(500, 500),(810, 620),(780, 575),(700, 550),(750, 60),(1, 1)
        # 40B baseline
        # (750, 0),(450, 500),(590, 240),(330, 400),(596, 459),(390, 310),(540, 200),(600, 450),(365, 125),(720, 605)
        # gemini baseline
        # (200, 100),(110, 150),(130, 150),(150, 100),(200, 150),(170, 120),(130, 150),(200, 150),(130, 120),(140, 100)
        # 8b cot
        # (520, 510),(780, 540),(780, 220),(600, 600),(300, 120),(810, 620),(740, 590),(750, 150)
        # 40b cot
        # (520, 230),(350, 500),(120, 130),(300, 160),(500, 100),(410, 310),(670, 290),(270, 350),(450, 350),(700, 350)
        # gemini cot
        # (120, 100),(100, 100),(110, 110),(120, 10),(110, 120),(110, 10),(100, 100),(110, 10),(112, 123),(120, 100)
        # finetune basline
        # (3, 19), (1, 93), (1, 93), (1, 93), (1, 19), (1, 93), (1, 93), (2, 360), (2, 360), (1, 93),
        # finetune cot
        # (2, 360), (2, 360), (2, 14), (2, 360), (119, 0), (1, 93), (2, 360), (2, 360), (2, 360), (2, 360)
    ]
    model = 'gpt' # model
    prompt = 'CoT' # prompt/task
    csv_filepath = "../output/Noise_(630, 820)_(1, 1)_5.csv"  # 修改为实际的 CSV 文件路径
    noise = load_noise_from_csv(csv_filepath)
    noise2 = load_noise_from_csv("../output/987.csv")
    for i, colormap in enumerate(newcolormap_list):
        scalar_field_png_name_with_density = f"{model}_{prompt}_density_overlay.png"
        scalar_field_png_path_with_density = os.path.join("../output", scalar_field_png_name_with_density)
        plot_scalar_field_with_density_map(
            shape, noise, noise2, cmap, scalar_field_png_path_with_density, marker_value, points, colormap
        )

if __name__ == "__main__":
    main()