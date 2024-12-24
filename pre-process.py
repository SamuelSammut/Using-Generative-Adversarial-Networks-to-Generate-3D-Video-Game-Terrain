import os
import numpy as np
import rasterio  # To read GeoTIFF files
from scipy.ndimage import zoom  # For resizing

# Function to load GeoTIFF as a NumPy array using Rasterio
def load_geotiff(path):
    with rasterio.open(path) as dataset:
        array = dataset.read()
        return array


def stretch_rgb(image):
    stretched_image = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[2]):  # Loop through R, G, B bands
        band = image[:, :, i]
        p2, p98 = np.percentile(band, (2, 98))  # 2nd and 98th percentiles
        if p98 > p2:  # Avoid division by zero
            stretched_band = np.clip((band - p2) / (p98 - p2), 0, 1)
        else:
            stretched_band = np.zeros_like(band, dtype=np.float32)  # All zeros if percentiles are invalid
        stretched_image[:, :, i] = stretched_band
    return stretched_image


# Function to process a single DEM and RGB image pair
def process_images(dem_path, rgb_path, output_folder, global_min, global_max):
    rgb_image_raw = load_geotiff(rgb_path)
    dem_image = load_geotiff(dem_path)[0]

    # Extract and stretch RGB
    rgb_image = np.transpose(rgb_image_raw[[3, 2, 1], :, :], (1, 2, 0))
    rgb_image = np.nan_to_num(rgb_image, nan=0.0)
    rgb_image = stretch_rgb(rgb_image)

    # Normalize DEM
    dem_image = np.nan_to_num(dem_image, nan=0.0)
    dem_image_resized = zoom(dem_image, (256 / dem_image.shape[0], 256 / dem_image.shape[1]))
    dem_image_resized = np.clip((dem_image_resized - global_min) / (global_max - global_min + 1e-8), 0, 1)

    # Resize RGB
    rgb_image_resized = zoom(rgb_image, (256 / rgb_image.shape[0], 256 / rgb_image.shape[1], 1))

    # Combine and save
    combined_image = np.dstack((rgb_image_resized, dem_image_resized))
    output_filename = os.path.join(output_folder, os.path.basename(rgb_path).replace('rgb_image_', 'combined_image_').replace('.tif', '.npy'))
    np.save(output_filename, combined_image)
    print(f"Saved combined image: {output_filename}")

def balance_files(input_folder, delete_unmatched=False):
    dem_files = sorted([f for f in os.listdir(input_folder) if f.startswith('dem_image_') and f.endswith('.tif')])
    rgb_files = sorted([f for f in os.listdir(input_folder) if f.startswith('rgb_image_') and f.endswith('.tif')])

    # Extract numeric IDs for matching
    dem_ids = set(int(f.replace('dem_image_', '').replace('.tif', '')) for f in dem_files)
    rgb_ids = set(int(f.replace('rgb_image_', '').replace('.tif', '')) for f in rgb_files)

    # Identify unmatched files
    unmatched_dems = dem_ids - rgb_ids
    unmatched_rgbs = rgb_ids - dem_ids

    print("==== Mismatch Report ====")
    print(f"Total DEM files without matching RGB: {len(unmatched_dems)}")
    print(f"Total RGB files without matching DEM: {len(unmatched_rgbs)}")

    if unmatched_dems:
        print("\nDEM files without matching RGB files:")
        for dem_id in unmatched_dems:
            print(f"  - dem_image_{dem_id}.tif")

    if unmatched_rgbs:
        print("\nRGB files without matching DEM files:")
        for rgb_id in unmatched_rgbs:
            print(f"  - rgb_image_{rgb_id}.tif")

    # Optional deletion
    if delete_unmatched:
        print("\nBalancing files: Removing unmatched files...")
        for dem_id in unmatched_dems:
            file_path = os.path.join(input_folder, f"dem_image_{dem_id}.tif")
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted unmatched DEM file: {file_path}")

        for rgb_id in unmatched_rgbs:
            file_path = os.path.join(input_folder, f"rgb_image_{rgb_id}.tif")
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted unmatched RGB file: {file_path}")
    else:
        print("\nNo files were deleted. To delete unmatched files, set 'delete_unmatched=True'.")

def process_all_images_in_folder(input_folder_path, output_folder, delete_unmatched=False):
    # Balance files (optional)
    balance_files(input_folder_path, delete_unmatched=delete_unmatched)

    # Get updated file lists after balancing
    dem_files = sorted([f for f in os.listdir(input_folder_path) if f.startswith('dem_image_') and f.endswith('.tif')])
    rgb_files = sorted([f for f in os.listdir(input_folder_path) if f.startswith('rgb_image_') and f.endswith('.tif')])

    # Ensure we have equal numbers of DEM and RGB images
    if len(dem_files) != len(rgb_files):
        print("Error: Mismatch still exists after balancing.")
        return

    # Compute global min and max for DEM data
    global_min = float('inf')
    global_max = float('-inf')
    for dem_file in dem_files:
        dem_path = os.path.join(input_folder_path, dem_file)
        dem_image = load_geotiff(dem_path)[0]
        global_min = min(global_min, np.nanmin(dem_image))
        global_max = max(global_max, np.nanmax(dem_image))

    print(f"Global DEM Min: {global_min}, Global DEM Max: {global_max}")

    # Process image pairs, skipping already processed ones
    for dem_file, rgb_file in zip(dem_files, rgb_files):
        output_filename = os.path.join(output_folder, rgb_file.replace('rgb_image_', 'combined_image_').replace('.tif', '.npy'))
        if os.path.exists(output_filename):
            print(f"Skipping already processed file: {output_filename}")
            continue  # Skip already processed files

        dem_path = os.path.join(input_folder_path, dem_file)
        rgb_path = os.path.join(input_folder_path, rgb_file)
        process_images(dem_path, rgb_path, output_folder, global_min, global_max)

# Input and output folder paths
input_folder_path = "/workspace/images"
output_folder_path = "/workspace/preprocessed_data"
delete_unmatched_files = True  # Set to True to delete unpaired files

# Run the processing pipeline
process_all_images_in_folder(input_folder_path, output_folder_path, delete_unmatched=delete_unmatched_files)
