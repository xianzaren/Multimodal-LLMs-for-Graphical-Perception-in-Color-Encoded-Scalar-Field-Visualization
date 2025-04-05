import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import random
from PIL import Image
import matplotlib.colors as mcolors


def create_blueyellow_colormap():
    """Create custom blue-yellow colormap"""
    blue = (13 / 255.0, 0 / 255.0, 252 / 255.0)
    yellow = (252 / 255.0, 252 / 255.0, 0 / 255.0)
    return mcolors.LinearSegmentedColormap.from_list("blueyellow", [blue, yellow], N=256)


def save_colorbar_rgb(cmap, tick_positions, save_path):
    """Save colorbar RGB values to file"""
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    with open(save_path, 'w') as f:
        f.write("Tick Value, R, G, B\n")
        for tick in tick_positions:
            normalized_value = tick / 1000.0
            rgb = cmap(normalized_value)[:3]
            f.write(f"{tick:.0f}, {rgb[0]:.6f}, {rgb[1]:.6f}, {rgb[2]:.6f}\n")


def compute_gradient_average(box):
    """Calculate average gradient magnitude in a box"""
    gradient_x, gradient_y = np.gradient(box)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    return np.mean(gradient_magnitude)


def find_matching_box(noise, x1, y1, target_ratio, tolerance, box_size=175, max_iterations=50000):
    """Find matching box with target gradient ratio"""
    height, width = noise.shape
    iterations = 0

    while True:
        x2 = random.randint(30, width - box_size - 30)
        y2 = random.randint(30, height - box_size - 30)
        iterations += 1

        if abs(x1 - x2) > box_size or abs(y1 - y2) > box_size:
            box1 = noise[y1:y1 + box_size, x1:x1 + box_size]
            box2 = noise[y2:y2 + box_size, x2:x2 + box_size]
            avg1 = compute_gradient_average(box1)
            avg2 = compute_gradient_average(box2)
            flatter, steeper = sorted([avg1, avg2])
            ratio = flatter / steeper if steeper != 0 else 0
            if abs(ratio - target_ratio) <= tolerance:
                return x2, y2, avg1, avg2

        if iterations % max_iterations == 0:
            tolerance += 0.01
        if iterations > 500000:
            return None


def get_box_colors(colormap):
    """Get box colors based on colormap type"""
    cmap_str = colormap if isinstance(colormap, str) else getattr(colormap, 'name', '')

    color_mapping = {
        "hot": ('purple', 'green', "Purple Box Avg", "Green Box Avg"),
        "rainbow": ('black', 'white', "Black Box Avg", "White Box Avg"),
        "gray": ('red', 'blue', "Red Box Avg", "Blue Box Avg"),
        "Blues_r": ('yellow', 'green', "Yellow Box Avg", "Green Box Avg"),
        "coolwarm": ('purple', 'green', "Purple Box Avg", "Green Box Avg"),
        "cubehelix": ('red', 'blue', "Red Box Avg", "Blue Box Avg"),
        "magma": ('green', 'blue', "Green Box Avg", "Blue Box Avg"),
        "nipy_spectral": ('black', 'white', "Black Box Avg", "White Box Avg"),
        "blueyellow": ('Purple', 'Green', "Purple Box Avg", "Green Box Avg")
    }

    return color_mapping.get(cmap_str, ('red', 'blue', "Red Box Avg", "Blue Box Avg"))


def generate_image_with_boxes_and_compare(noise, colormap, output_filepath, result_filepath, target_ratios,
                                          tolerance=0.05):
    """Generate image with boxes and compare gradient ratios"""
    height, width = noise.shape
    box_size = 175
    random.seed(42)

    first_color_default, second_color_default, first_text_default, second_text_default = get_box_colors(colormap)

    with open(result_filepath, "a") as result_file:
        for i, target_ratio in enumerate(target_ratios):
            while True:
                first_is_red = random.choice([True, False])
                if first_is_red:
                    first_color, second_color = first_color_default, second_color_default
                    first_text, second_text = first_text_default, second_text_default
                else:
                    first_color, second_color = second_color_default, first_color_default
                    first_text, second_text = second_text_default, first_text_default

                x1 = random.randint(30, width - box_size - 30)
                y1 = random.randint(30, height - box_size - 30)

                match = find_matching_box(noise, x1, y1, target_ratio, tolerance, box_size)
                if match:
                    x2, y2, avg1, avg2 = match
                    flatter, steeper = sorted([avg1, avg2])
                    ratio = flatter / steeper if steeper != 0 else 0

                    dpi = 100
                    fig, ax = plt.subplots(figsize=(767 / dpi, 630 / dpi))
                    im = ax.imshow(noise, cmap=colormap, origin='upper', vmin=0, vmax=1000)

                    cbar = plt.colorbar(im, label='Elevation', fraction=0.030, pad=0.05)
                    tick_positions = np.linspace(0, 1000, 9)
                    cbar.set_ticks(tick_positions)
                    cbar.set_ticklabels([f"{t:.0f}" for t in tick_positions])

                    plt.axis('off')

                    rect1 = plt.Rectangle((x1, y1), box_size, box_size, edgecolor=first_color,
                                          facecolor='none', linewidth=2)
                    rect2 = plt.Rectangle((x2, y2), box_size, box_size, edgecolor=second_color,
                                          facecolor='none', linewidth=2)
                    ax.add_patch(rect1)
                    ax.add_patch(rect2)

                    output_filepath_with_index = output_filepath.replace(".png", f"_{i + 1}.png")
                    steeper_text = first_text if avg1 > avg2 else second_text
                    result_file.write(f"{os.path.basename(output_filepath_with_index)}: {steeper_text}\n")

                    plt.savefig(output_filepath_with_index, dpi=150, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    break


def main():
    """Main function to process images with different colormaps"""
    csv_dir = r"F:\submit data\test pic\exp1\data"
    target_ratios = [0.8, 0.83, 0.86, 0.9]
    blueyellow = create_blueyellow_colormap()

    colormap_list = [
        'gray', 'Blues', 'hot', 'cubehelix', 'magma',
        'coolwarm', 'rainbow', 'spectral', 'blueyellow'
    ]

    for colormap in colormap_list:
        cmap_name = colormap
        folder = 'exp2'
        output_dir = fr"..\..\color\{folder}\{cmap_name}"
        result_filepath = os.path.join(output_dir, "result.txt")

        os.makedirs(output_dir, exist_ok=True)
        with open(result_filepath, "w") as result_file:
            result_file.write("")

        for csv_filename in os.listdir(csv_dir):
            if csv_filename.endswith(".csv"):
                csv_filepath = os.path.join(csv_dir, csv_filename)
                noise = np.loadtxt(csv_filepath, delimiter=",", skiprows=1)

                # Normalize to [0,1000] range
                data_min, data_max = noise.min(), noise.max()
                noise = (noise - data_min) / (data_max - data_min) * 1000

                output_filepath = os.path.join(
                    output_dir,
                    f"ScalarField_WithBoxes_{os.path.splitext(csv_filename)[0]}_{cmap_name}.png"
                )

                # Select appropriate colormap
                colormap_used = {
                    'Blues': 'Blues_r',
                    'blueyellow': blueyellow,
                    'spectral': 'nipy_spectral'
                }.get(colormap, colormap)

                generate_image_with_boxes_and_compare(
                    noise, colormap_used, output_filepath, result_filepath, target_ratios
                )


if __name__ == "__main__":
    main()