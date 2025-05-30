import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error as mse
from scipy.ndimage import gaussian_laplace

# === CONFIGURATION ===
real_folder = "./real_dems_for_compare"
fake_folder = "./fake_dems_for_compare"
num_samples = 200
output_csv = "dem_evaluation_results.csv"
output_summary = "dem_evaluation_summary.csv"
output_plot_folder = "dem_evaluation_plots"

os.makedirs(output_plot_folder, exist_ok=True)

# terrain geometry functions
def compute_slope(dem):
    dx, dy = np.gradient(dem)
    return np.sqrt(dx**2 + dy**2)

def compute_curvature(dem):
    return gaussian_laplace(dem, sigma=1)

def compute_roughness(dem):
    window = 3
    padded = np.pad(dem, pad_width=1, mode='reflect')
    local_mean = np.zeros_like(dem)
    for i in range(dem.shape[0]):
        for j in range(dem.shape[1]):
            local_window = padded[i:i+window, j:j+window]
            local_mean[i, j] = np.mean(local_window)
    return np.abs(dem - local_mean)

# load dem from tif
def load_dem(path):
    with rasterio.open(path) as src:
        return src.read(1)

# evaluation
real_files = sorted([f for f in os.listdir(real_folder) if f.endswith(".tif")])[:num_samples]
fake_files = sorted([f for f in os.listdir(fake_folder) if f.endswith(".tif")])[:num_samples]

results = []

print(f"Evaluating {num_samples} DEM pairs...")

for i in range(num_samples):
    real_path = os.path.join(real_folder, real_files[i])
    fake_path = os.path.join(fake_folder, fake_files[i])

    real_dem = load_dem(real_path)
    fake_dem = load_dem(fake_path)

    # Geometry
    slope_r = compute_slope(real_dem)
    slope_f = compute_slope(fake_dem)
    curv_r = compute_curvature(real_dem)
    curv_f = compute_curvature(fake_dem)
    rough_r = compute_roughness(real_dem)
    rough_f = compute_roughness(fake_dem)

    # Metrics
    ssim_slope = ssim(slope_r, slope_f, data_range=slope_f.max() - slope_f.min())
    ssim_curv = ssim(curv_r, curv_f, data_range=curv_f.max() - curv_f.min())
    mse_rough = mse(rough_r.flatten(), rough_f.flatten())

    results.append({
        "real_file": real_files[i],
        "fake_file": fake_files[i],
        "ssim_slope": ssim_slope,
        "ssim_curvature": ssim_curv,
        "mse_roughness": mse_rough
    })

    print(f"[{i+1}/{num_samples}] {real_files[i]} vs {fake_files[i]} — SSIM(slope): {ssim_slope:.4f}, SSIM(curv): {ssim_curv:.4f}, MSE(rough): {mse_rough:.2f}")

# Save results
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
summary = df.describe()
summary.to_csv(output_summary)

# Plot histograms
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(df["ssim_slope"], bins=20, color='skyblue')
plt.title("SSIM on Slope")

plt.subplot(1, 3, 2)
plt.hist(df["ssim_curvature"], bins=20, color='salmon')
plt.title("SSIM on Curvature")

plt.subplot(1, 3, 3)
plt.hist(df["mse_roughness"], bins=20, color='lightgreen')
plt.title("MSE on Roughness")

plt.tight_layout()
plt.savefig(os.path.join(output_plot_folder, "metric_histograms.png"))
plt.show()

print(f"\n✅ Evaluation complete!")
print(f"📄 Results saved to: {output_csv}")
print(f"📊 Summary saved to: {output_summary}")
print(f"📈 Plots saved to: {output_plot_folder}/metric_histograms.png")
