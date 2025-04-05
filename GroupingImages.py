import os
import shutil

def copy_png_files(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for filename in os.listdir(source_folder):
        if filename.endswith(".png"):
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, filename)
            shutil.copy2(source_file, destination_file)
            print(f"Copied: {filename}")

# Example usage
source = "exp2"
target = "exp2"

# Define the source folder
source_folder = fr"E:\桌面\Final project\color\oringinal test\{source}"

# Get all subfolder paths
all_subfolders = [
    os.path.join(source_folder, subfolder)
    for subfolder in os.listdir(source_folder)
    if os.path.isdir(os.path.join(source_folder, subfolder))
]

# Copy PNG files to each destination folder (1 to 10)
for i in range(1, 11):
    destination_folder = fr"E:\桌面\Final project\result\{target}\{i}"
    for subfolder in all_subfolders:
        copy_png_files(subfolder, destination_folder)
