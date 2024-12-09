import os
import numpy as np
import rasterio  # To read GeoTIFF files
from scipy.ndimage import zoom  # For resizing


# Function to load GeoTIFF as a NumPy array using Rasterio
def load_geotiff(path):
    with rasterio.open(path) as dataset:
        array = dataset.read()
        return array


# Function to process a single DEM and RGB image pair
def process_images(dem_path, rgb_path, output_folder, global_min, global_max):
    # Loading the RGB image and extracting the R, G, B bands
    # For Sentinel-2 imagery, bands 4 (Red), 3 (Green), 2 (Blue) correspond to indices 3, 2, 1 (zero-based indexing)
    rgb_image_raw = load_geotiff(rgb_path)
    rgb_image = rgb_image_raw[[3, 2, 1], :, :]

    # Reordering to (height, width, 3)
    rgb_image = np.transpose(rgb_image, (1, 2, 0))

    # Replace NaN values in RGB images with zeros
    rgb_image = np.nan_to_num(rgb_image, nan=0.0)

    # Scale the RGB data from 0-10000 to 0-1 (Sentinel-2 reflectance values)
    rgb_image_resized = rgb_image / 10000.0

    # Clip values to [0, 1]
    rgb_image_resized = np.clip(rgb_image_resized, 0.0, 1.0)

    # Load the DEM image and accessing the first band for DEM
    dem_image = load_geotiff(dem_path)[0]

    # Check for NaN or Inf in dem_image before resizing
    if np.isnan(dem_image).any() or np.isinf(dem_image).any():
        print(f"Warning: dem_image contains NaN or Inf values before resizing! File: {dem_path}")
        dem_image = np.nan_to_num(dem_image, nan=0.0)

    target_size = (256, 256)
    rgb_image_resized = zoom(rgb_image_resized, (target_size[0] / rgb_image.shape[0],
                                                 target_size[1] / rgb_image.shape[1], 1))
    dem_image_resized = zoom(dem_image, (target_size[0] / dem_image.shape[0],
                                         target_size[1] / dem_image.shape[1]))

    # Check for NaN or Inf in dem_image_resized after resizing
    if np.isnan(dem_image_resized).any() or np.isinf(dem_image_resized).any():
        print(f"Warning: dem_image_resized contains NaN or Inf values after resizing! File: {dem_path}")

    # Normalize the DEM data to [0, 1] using global min and max
    epsilon = 1e-8  # Small constant to prevent division by zero
    dem_image_resized = (dem_image_resized - global_min) / (global_max - global_min + epsilon)
    dem_image_resized = np.clip(dem_image_resized, 0.0, 1.0)

    # Ensuring consistent data types to float32
    rgb_image_resized = rgb_image_resized.astype(np.float32)
    dem_image_resized = dem_image_resized.astype(np.float32)

    # Stack RGB and DEM to create a 4-channel image (height, width, 4)
    combined_image = np.dstack((rgb_image_resized, dem_image_resized))

    print("Combined image shape:", combined_image.shape)

    # Save the combined image as a .npy file
    output_filename = os.path.join(output_folder, os.path.basename(rgb_path).replace('rgb_image_', 'combined_image_').replace('.tif', '.npy'))
    np.save(output_filename, combined_image)
    print(f"Saved combined image: {output_filename}")

    # Check for NaN or Inf in combined_image
    if np.isnan(combined_image).any() or np.isinf(combined_image).any():
        print(f"Warning: combined_image contains NaN or Inf values! File: {output_filename}")

# Function to find and process all image pairs in a folder
def process_all_images_in_folder(input_folder_path, output_folder):
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Get lists of DEM and RGB files based on naming convention
    dem_files = sorted([f for f in os.listdir(input_folder_path) if f.startswith('dem_image_') and f.endswith('.tif')])
    rgb_files = sorted([f for f in os.listdir(input_folder_path) if f.startswith('rgb_image_') and f.endswith('.tif')])

    # Ensure we have equal numbers of DEM and RGB images
    if len(dem_files) != len(rgb_files):
        print("Error: Mismatch between number of DEM and RGB files.")
        return

    # **Compute Global Min and Max for DEM Data**
    global_min = float('inf')
    global_max = float('-inf')
    for dem_file in dem_files:
        dem_path = os.path.join(input_folder_path, dem_file)
        dem_image = load_geotiff(dem_path)[0]
        dem_min = np.nanmin(dem_image)
        dem_max = np.nanmax(dem_image)
        if dem_min < global_min:
            global_min = dem_min
        if dem_max > global_max:
            global_max = dem_max
    print(f"Global DEM Min: {global_min}, Global DEM Max: {global_max}")

    # Process each pair of images
    for dem_file, rgb_file in zip(dem_files, rgb_files):
        dem_path = os.path.join(input_folder_path, dem_file)
        rgb_path = os.path.join(input_folder_path, rgb_file)
        process_images(dem_path, rgb_path, output_folder, global_min, global_max)

input_folder_path = "/workspace/images"
output_folder_path = "/workspace/preprocessed_data"
process_all_images_in_folder(input_folder_path, output_folder_path)
