import os
import numpy as np
import rasterio
from scipy.ndimage import sobel, laplace
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import random

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

def compute_roughness(dem, patch_size=3):
    """Compute local standard deviation (roughness)"""
    from scipy.ndimage import generic_filter
    def local_std(x):
        return np.std(x)
    return generic_filter(dem, local_std, size=patch_size)

def evaluate_metrics(real_dem, fake_dem):
    metrics = {}

    try:
        slope_real = compute_slope(real_dem)
        slope_fake = compute_slope(fake_dem)
        metrics["slope_ssim"] = ssim(slope_real, slope_fake, data_range=1.0)
    except:
        metrics["slope_ssim"] = np.nan

    try:
        curv_real = compute_curvature(real_dem)
        curv_fake = compute_curvature(fake_dem)
        metrics["curvature_ssim"] = ssim(curv_real, curv_fake, data_range=1.0)
    except:
        metrics["curvature_ssim"] = np.nan

    try:
        rough_real = compute_roughness(real_dem)
        rough_fake = compute_roughness(fake_dem)
        metrics["roughness_mse"] = np.mean((rough_real - rough_fake)**2)
    except:
        metrics["roughness_mse"] = np.nan

    return metrics

def evaluate_sets(real_dir, fake_dir, num_samples=50):
    real_files = sorted([f for f in os.listdir(real_dir) if f.endswith('.tif')])
    fake_files = sorted([f for f in os.listdir(fake_dir) if f.endswith('.tif')])

    real_paths = [os.path.join(real_dir, f) for f in real_files]
    fake_paths = [os.path.join(fake_dir, f) for f in fake_files]

    all_scores = {
        "slope_ssim": [],
        "curvature_ssim": [],
        "roughness_mse": []
    }

    for _ in tqdm(range(min(num_samples, len(real_paths), len(fake_paths))), desc="Evaluating"):
        real_path = random.choice(real_paths)
        fake_path = random.choice(fake_paths)

        real_dem = load_dem(real_path)
        fake_dem = load_dem(fake_path)

        metrics = evaluate_metrics(real_dem, fake_dem)
        for key, val in metrics.items():
            if not np.isnan(val):
                all_scores[key].append(val)

    print("\n=== Evaluation Results ===")
    for metric, values in all_scores.items():
        if values:
            print(f"{metric}: {np.mean(values):.4f} (avg over {len(values)} samples)")
        else:
            print(f"{metric}: no valid comparisons")

# Example usage
if __name__ == "__main__":
    evaluate_sets(
        real_dir="real_dem_for_grass",
        fake_dir="fake_dem_for_grass",
        num_samples=50
    )
