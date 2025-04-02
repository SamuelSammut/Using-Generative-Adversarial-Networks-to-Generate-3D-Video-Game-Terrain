import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_preprocessed_data(input_folder, output_folder, num_samples=100):
    """
    Saves preprocessed RGB+DEM arrays as separate 256x256 images in distinct folders.

    Parameters:
        input_folder (str): Path to the folder containing .npy files.
        output_folder (str): Path to save output images (separate folders for RGB & DEM).
        num_samples (int): Number of images to visualize.
    """
    rgb_folder = os.path.join(output_folder, "rgb")
    dem_folder = os.path.join(output_folder, "dem")

    os.makedirs(rgb_folder, exist_ok=True)
    os.makedirs(dem_folder, exist_ok=True)

    npy_files = sorted(
        [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.npy')]
    )

    # Select a subset of files to visualize
    num_samples = min(len(npy_files), num_samples)
    selected_files = np.random.choice(npy_files, num_samples, replace=False)

    for file_path in selected_files:
        data = np.load(file_path)
        rgb_image = data[:, :, :3]
        dem_image = data[:, :, 3]

        if rgb_image.dtype != np.uint8:
            min_val, max_val = np.min(rgb_image), np.max(rgb_image)
            if max_val > min_val:
                rgb_image = (rgb_image - min_val) / (max_val - min_val)
            rgb_image = (rgb_image * 255).astype(np.uint8)

        if dem_image.dtype != np.float32:
            dem_image = dem_image.astype(np.float32)
            min_val, max_val = np.min(dem_image), np.max(dem_image)
            if max_val > min_val:
                dem_image = (dem_image - min_val) / (max_val - min_val)

        # Define save paths
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        rgb_out = os.path.join(rgb_folder, f"{base_name}_rgb.png")
        dem_out = os.path.join(dem_folder, f"{base_name}_dem.png")

        # Save images
        plt.imsave(rgb_out, rgb_image)
        plt.imsave(dem_out, dem_image, cmap='gray')

        print(f"Saved: {rgb_out}")
        print(f"Saved: {dem_out}")

    print(f"Processed {len(selected_files)} images.")

if __name__ == "__main__":
    input_folder = "/workspace/preprocessed_data"
    output_folder = "../preprocessed_data_images"
    num_samples = 200

    visualize_preprocessed_data(input_folder, output_folder, num_samples)
