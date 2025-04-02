import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, laplace

def load_dem(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if np.max(arr) - np.min(arr) > 1e-8:
            arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        else:
            arr = np.zeros_like(arr)
    return arr

def compute_slope(dem):
    dx = sobel(dem, axis=1)
    dy = sobel(dem, axis=0)
    return np.sqrt(dx**2 + dy**2)

def compute_curvature(dem):
    return np.abs(laplace(dem))

def visualize_dem_comparison(real_path, fake_path, output_dir="debug_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    real_name = os.path.basename(real_path).replace(".tif", "")
    fake_name = os.path.basename(fake_path).replace(".tif", "")

    real_dem = load_dem(real_path)
    fake_dem = load_dem(fake_path)

    real_slope = compute_slope(real_dem)
    fake_slope = compute_slope(fake_dem)

    real_curv = compute_curvature(real_dem)
    fake_curv = compute_curvature(fake_dem)

    fig, axes = plt.subplots(4, 2, figsize=(12, 10))

    axes[0, 0].imshow(real_dem, cmap='terrain')
    axes[0, 0].set_title("Real DEM")
    axes[0, 1].imshow(fake_dem, cmap='terrain')
    axes[0, 1].set_title("Fake DEM")

    axes[1, 0].imshow(real_slope, cmap='viridis')
    axes[1, 0].set_title("Real Slope")
    axes[1, 1].imshow(fake_slope, cmap='viridis')
    axes[1, 1].set_title("Fake Slope")

    axes[2, 0].imshow(real_curv, cmap='magma')
    axes[2, 0].set_title("Real Curvature")
    axes[2, 1].imshow(fake_curv, cmap='magma')
    axes[2, 1].set_title("Fake Curvature")

    axes[3, 0].hist(real_dem.ravel(), bins=50, color='blue', alpha=0.7)
    axes[3, 0].set_title("Real DEM Histogram")
    axes[3, 1].hist(fake_dem.ravel(), bins=50, color='green', alpha=0.7)
    axes[3, 1].set_title("Fake DEM Histogram")

    for ax in axes.flat:
        ax.axis('off' if ax.get_subplotspec().rowspan.stop < 3 else 'on')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"compare_{real_name}_vs_{fake_name}.png")
    plt.savefig(save_path)
    print(f"Saved comparison to: {save_path}")

# Example usage
if __name__ == "__main__":
    real_example = "real_dem_for_grass/rgb_2_116_flip_vertical_dem.tif"
    fake_example = "fake_dem_for_grass/terrain_0001_height.tif"
    visualize_dem_comparison(real_example, fake_example)
