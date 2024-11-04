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
def process_images(dem_path, rgb_path, output_folder):
    # Loading the RGB image and extracting only the first 3 bands (R, G, B)
    rgb_image = load_geotiff(rgb_path)[:3, :, :]

    # Reordering to (height, width, 3)
    rgb_image = np.transpose(rgb_image, (1, 2, 0))

    # Load the DEM image and accessing the first band for DEM
    dem_image = load_geotiff(dem_path)[0]

    # Calculating the zoom factors for resizing the DEM to match the RGB image dimensions
    zoom_factors = (
        rgb_image.shape[0] / dem_image.shape[0],  # Height factor
        rgb_image.shape[1] / dem_image.shape[1]  # Width factor
    )

    # Resizing the DEM using scipy.ndimage.zoom to match RGB dimensions
    dem_image_resized = zoom(dem_image, zoom_factors)

    # Normalizing the RGB data (scale pixel values to the range [0, 1])
    rgb_image = rgb_image / 255.0

    # Normalizing the DEM data based on the max elevation value
    dem_image_resized = dem_image_resized / np.max(dem_image_resized)

    # Stack RGB and DEM to create a 4-channel image (height, width, 4)
    combined_image = np.dstack((rgb_image, dem_image_resized))

    print("Combined image shape before resizing:", combined_image.shape)

    # Define target dimensions for resizing
    target_size = (128, 128)
    height_factor = target_size[0] / combined_image.shape[0]
    width_factor = target_size[1] / combined_image.shape[1]

    # Resize to (128, 128, 4) using zoom factors for height and width
    combined_image_resized = zoom(combined_image, (height_factor, width_factor, 1))

    print("Combined image shape after resizing:", combined_image_resized.shape)

    # Save the combined image as a .npy file
    output_filename = os.path.join(output_folder, os.path.basename(rgb_path).replace('rgb_image_', 'combined_image_').replace('.tif', '.npy'))
    np.save(output_filename, combined_image_resized)
    print(f"Saved combined image: {output_filename}")

    # Print the shapes of the resized images
    print(f"Processed {os.path.basename(rgb_path)} and {os.path.basename(dem_path)}")
    print("Resized RGB Image Shape:", rgb_image.shape)  # Should be (height, width, 3) for RGB
    print("Resized DEM Image Shape:", dem_image_resized.shape)  # Should match RGB dimensions

    return rgb_image, dem_image_resized


# Function to find and process all image pairs in a folder
def process_all_images_in_folder(input_folder_path , output_folder):
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Get lists of DEM and RGB files based on naming convention
    dem_files = sorted([f for f in os.listdir(input_folder_path ) if f.startswith('dem_image_') and f.endswith('.tif')])
    rgb_files = sorted([f for f in os.listdir(input_folder_path ) if f.startswith('rgb_image_') and f.endswith('.tif')])

    # Ensure we have equal numbers of DEM and RGB images
    if len(dem_files) != len(rgb_files):
        print("Error: Mismatch between number of DEM and RGB files.")
        return

    # Process each pair of images
    for dem_file, rgb_file in zip(dem_files, rgb_files):
        dem_path = os.path.join(input_folder_path , dem_file)
        rgb_path = os.path.join(input_folder_path , rgb_file)
        process_images(dem_path, rgb_path, output_folder)


# Specify the folder containing the images and for output
input_folder_path  = "/workspace/images"
output_folder_path = "/workspace/preprocessed_data"
process_all_images_in_folder(input_folder_path, output_folder_path)