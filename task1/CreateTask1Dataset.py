import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from scipy.fftpack import fft2, fftshift
import os


def save_colorbar_rgb(cmap, tick_positions, save_path):
    """Save RGB values of colormap ticks to file"""
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    with open(save_path, 'w') as f:
        f.write("Tick Value, R, G, B\n")
        for tick in tick_positions:
            rgb = cmap(tick / 1000.0)[:3]
            f.write(f"{tick:.0f}, {rgb[0]:.6f}, {rgb[1]:.6f}, {rgb[2]:.6f}\n")


def generate_seamless_perlin_noise_2d(shape, res, octaves=5, persistence=0.5, seed=42):
    """Generate seamless Perlin noise with given parameters"""

    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    def gradient_grid(res):
        np.random.seed(seed)
        angles = 2 * np.pi * np.random.rand(res[0], res[1])
        return np.dstack((np.cos(angles), np.sin(angles)))

    def perlin(x, y, gradients):
        x0 = np.floor(x).astype(int) % gradients.shape[0]
        y0 = np.floor(y).astype(int) % gradients.shape[1]
        x1 = (x0 + 1) % gradients.shape[0]
        y1 = (y0 + 1) % gradients.shape[1]

        dx, dy = x - x0, y - y0
        sx, sy = f(dx), f(dy)

        n00 = gradients[x0, y0, 0] * dx + gradients[x0, y0, 1] * dy
        n10 = gradients[x1, y0, 0] * (dx - 1) + gradients[x1, y0, 1] * dy
        n01 = gradients[x0, y1, 0] * dx + gradients[x0, y1, 1] * (dy - 1)
        n11 = gradients[x1, y1, 0] * (dx - 1) + gradients[x1, y1, 1] * (dy - 1)

        nx0 = (1 - sx) * n00 + sx * n10
        nx1 = (1 - sx) * n01 + sx * n11
        return (1 - sy) * nx0 + sy * nx1

    noise = np.zeros(shape)
    max_amplitude = (1 - persistence ** octaves) / (1 - persistence)
    frequency, amplitude = 1, 1

    for _ in range(octaves):
        res_scaled = (int(res[0] * frequency), int(res[1] * frequency)
                      gradients = gradient_grid(res_scaled)

        grid_x, grid_y = np.meshgrid(
            np.arange(shape[1]) * (res_scaled[1] / shape[1]),
            np.arange(shape[0]) * (res_scaled[0] / shape[0])
        )

        noise += amplitude * perlin(grid_x, grid_y, gradients)
        amplitude *= persistence
        frequency *= 2

        noise = noise / max_amplitude
        noise = np.rint((noise - noise.min()) / (noise.max() - noise.min()) * 1000)
    return np.clip(noise, 0, 1000).astype(int)


def compute_power_spectrum(noise):
    """Compute power spectrum of noise"""
    return np.log1p(np.abs(fftshift(fft2(noise)) ** 2)


def plot_scalar_field(shape, noise, colormap, filepath):
    """Plot scalar field without axes"""
    plt.figure(figsize=(shape[1] / 100, shape[0] / 100), dpi=100)
    plt.imshow(noise, cmap=colormap, vmin=0, vmax=1000)
    plt.axis('off')
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()


def create_blueyellow_colormap():
    """Create custom blue-yellow colormap"""
    return LinearSegmentedColormap.from_list(
        "blueyellow",
        [(13 / 255, 0, 252 / 255), (252 / 255, 252 / 255, 0)],
        N=256
    )


def main():
    shape = (630, 820)
    octaves = 5
    res_list = [(1, 1), (3, 3), (5, 5), (7, 7), (9, 9)]

    blueyellow = create_blueyellow_colormap()
    colormaps = {
        'gray': 'gray',
        'Blues': 'Blues_r',
        'hot': 'hot',
        'cubehelix': 'cubehelix',
        'magma': 'magma',
        'coolwarm': 'coolwarm',
        'rainbow': 'rainbow',
        'spectral': 'nipy_spectral',
        'blueyellow': blueyellow
    }

    # Create directories
    base_dirs = {
        'uncolor': '../../data/uncolor',
        'colored': '../../data/colored',
        'color_maps': {name: f'../../color/exp1/{name}' for name in colormaps}
    }

    for dir_path in [*base_dirs.values(), *base_dirs['color_maps'].values()]:
        os.makedirs(dir_path, exist_ok=True)

    for res in res_list:
        noise = generate_seamless_perlin_noise_2d(shape, res, octaves)

        # Save raw data
        base_name = f"Noise_{shape}_{res}_{octaves}"
        np.savetxt(
            f"{base_dirs['uncolor']}/{base_name}.txt",
            noise,
            fmt="%.16f"
        )
        np.savetxt(
            f"{base_dirs['uncolor']}/{base_name}.csv",
            noise,
            delimiter=",",
            fmt="%.16f",
            header="Normalized Elevation Data",
            comments=""
        )

        # Generate visualizations for each colormap
        for name, cmap in colormaps.items():
            output_dir = base_dirs['color_maps'][name]

            # Colorbar image
            fig, ax = plt.subplots(figsize=(7.67, 6.3))
            img = ax.imshow(noise, cmap=cmap, vmin=0, vmax=1000)
            cbar = plt.colorbar(img, fraction=0.03, pad=0.05)
            ticks = np.linspace(0, 1000, 9)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{t:.0f}" for t in ticks])
            plt.axis('off')
            plt.savefig(
                f"{output_dir}/Colorbar_{name}_{shape}_{res}_{octaves}.png",
                dpi=150,
                bbox_inches='tight'
            )
            plt.close()

            # Save RGB values
            save_colorbar_rgb(
                cmap,
                ticks,
                f"{output_dir}/Colorbar_RGB_{name}_{shape}_{res}_{octaves}.txt"
            )

            # Scalar field
            plot_scalar_field(
                shape,
                noise,
                cmap,
                f"{output_dir}/exps/ScalarField_{name}_{shape}_{res}_{octaves}.png"
            )


if __name__ == "__main__":
    main()