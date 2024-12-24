import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import random

# Function to load GeoTIFF as a NumPy array using Rasterio
def load_geotiff(path):
    with rasterio.open(path) as dataset:
        # Read the image data
        array = dataset.read()

        # Replace NoData values with zeros
        nodata_value = dataset.nodata
        if nodata_value is not None:
            array = np.where(array == nodata_value, 0, array)

        # Replace any NaN values with zeros
        array = np.nan_to_num(array, nan=0.0)

        return array


# Stretch RGB values based on percentile
def stretch_rgb(image):
    stretched_image = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[2]):  # Loop through R, G, B bands
        band = image[:, :, i]
        p2, p98 = np.percentile(band, (2, 98))  # Get 2nd and 98th percentiles
        print(f"Band {i + 1} Percentiles: p2={p2}, p98={p98}")  # Debugging output
        stretched_band = np.clip((band - p2) / (p98 - p2), 0, 1)  # Normalize and clip
        stretched_image[:, :, i] = stretched_band
    return stretched_image

# Function to visualize RGB and DEM pairs in a grid-like layout
def visualize_rgb_dem_pairs(input_folder, output_path, grid_size=(4, 4)):
    """
    Visualizes RGB and DEM pairs in a grid layout and saves the output image.

    Parameters:
        input_folder (str): Path to the folder containing the RGB and DEM files.
        output_path (str): Path to save the generated grid image.
        grid_size (tuple): Number of rows and columns in the grid.
    """
    # Get sorted lists of RGB and DEM files
    rgb_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.startswith('rgb_image_') and f.endswith('.tif')])
    dem_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.startswith('dem_image_') and f.endswith('.tif')])

    # Ensure we have equal numbers of RGB and DEM files
    if len(rgb_files) != len(dem_files):
        print("Mismatch in the number of RGB and DEM files. Ensure they are balanced.")
        return

    # Randomly select image pairs
    total_pairs = min(len(rgb_files), grid_size[0] * grid_size[1] // 2)
    selected_indices = random.sample(range(len(rgb_files)), total_pairs)

    # Log the selected file paths
    print("\nSelected RGB and DEM file paths:")
    for idx in selected_indices:
        print(f"RGB: {rgb_files[idx]} | DEM: {dem_files[idx]}")

    # Create a figure for the grid
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(16, 8))
    axes = axes.flatten()

    for idx, i in enumerate(selected_indices):
        # Load RGB and DEM images
        rgb_image = load_geotiff(rgb_files[i])
        print(f"Before Cleaning RGB Image {os.path.basename(rgb_files[i])}: Min={np.nanmin(rgb_image)}, Max={np.nanmax(rgb_image)}")
        dem_image = load_geotiff(dem_files[i])[0]  # DEM is single-band

        # Prepare RGB for display (selecting R, G, B bands)
        rgb_image = np.transpose(rgb_image[[3, 2, 1], :, :], (1, 2, 0))  # Convert to HWC format
        print(f"RGB Image {os.path.basename(rgb_files[i])} Min: {rgb_image.min()}, Max: {rgb_image.max()}")

        # Stretch RGB for visualization
        rgb_image = stretch_rgb(rgb_image)

        print(f"After Stretching RGB Image {os.path.basename(rgb_files[i])}: Min={np.nanmin(rgb_image)}, Max={np.nanmax(rgb_image)}")

        # Add RGB to the grid
        axes[2 * idx].imshow(rgb_image)
        axes[2 * idx].set_title(f"RGB {os.path.basename(rgb_files[i])}")
        axes[2 * idx].axis('off')

        # Add DEM to the grid
        axes[2 * idx + 1].imshow(dem_image, cmap='terrain')
        axes[2 * idx + 1].set_title(f"DEM {os.path.basename(dem_files[i])}")
        axes[2 * idx + 1].axis('off')

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved preview grid to {output_path}")

# Example usage
if __name__ == "__main__":
    input_folder = "/workspace/images"  # Update with your folder path
    output_path = "preview_raw_grid_debug.png"
    visualize_rgb_dem_pairs(input_folder, output_path, grid_size=(4, 4))
