import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_preprocessed_data(input_folder, output_path, grid_size=(4, 4)):
    """
    Visualizes preprocessed RGB+DEM data in a grid layout and saves the output image.

    Parameters:
        input_folder (str): Path to the folder containing .npy files.
        output_path (str): Path to save the generated grid image.
        grid_size (tuple): Number of rows and columns in the grid.
    """
    npy_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.npy')])

    # Randomly select files to visualize
    total_files = min(len(npy_files), grid_size[0] * grid_size[1] // 2)
    selected_files = np.random.choice(npy_files, total_files, replace=False)

    # Create a grid for visualization
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(16, 8))
    axes = axes.flatten()

    for idx, file_path in enumerate(selected_files):
        data = np.load(file_path)
        rgb_image = data[:, :, :3]
        dem_image = data[:, :, 3]

        # Add RGB to the grid
        axes[2 * idx].imshow(rgb_image)
        axes[2 * idx].set_title(f"RGB {os.path.basename(file_path)}")
        axes[2 * idx].axis('off')

        # Add DEM to the grid
        axes[2 * idx + 1].imshow(dem_image, cmap='terrain')
        axes[2 * idx + 1].set_title(f"DEM {os.path.basename(file_path)}")
        axes[2 * idx + 1].axis('off')

    # Save the grid
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved preprocessed data preview to {output_path}")

if __name__ == "__main__":
    input_folder = "/workspace/preprocessed_data_resized"  # Path to preprocessed .npy files
    output_path = "/workspace/preprocessed_data_preview.png"
    visualize_preprocessed_data(input_folder, output_path, grid_size=(4, 4))
