import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from scipy.ndimage import laplace
from skimage.metrics import structural_similarity as ssim

# Paths
real_folder = "./real_dems_for_compare"
fake_folder = "./fake_dems_for_compare"

# Pick a matching real and fake file
real_paths = sorted([os.path.join(real_folder, f) for f in os.listdir(real_folder) if f.endswith(".tif")])
fake_paths = sorted([os.path.join(fake_folder, f) for f in os.listdir(fake_folder) if f.endswith(".tif")])

# Load sample
sample_index = 0
real = rasterio.open(real_paths[sample_index]).read(1).astype(np.float32)
fake = rasterio.open(fake_paths[sample_index]).read(1).astype(np.float32)

# Normalize for SSIM
real_norm = (real - np.min(real)) / (np.max(real) - np.min(real))
fake_norm = (fake - np.min(fake)) / (np.max(fake) - np.min(fake))

# Compute terrain metrics
def slope(dem): return np.sqrt(np.square(np.gradient(dem)[0]) + np.square(np.gradient(dem)[1]))
def roughness(dem): return np.std(dem)
def curvature(dem): return laplace(dem)

real_slope = slope(real)
fake_slope = slope(fake)
real_curv = curvature(real)
fake_curv = curvature(fake)

# Compute SSIM and histograms
ssim_slope = ssim(real_slope, fake_slope)
ssim_curv = ssim(real_curv, fake_curv)
mse_rough = np.mean(np.square(real - fake))

print(f"SSIM (Slope): {ssim_slope:.4f}")
print(f"SSIM (Curvature): {ssim_curv:.4f}")
print(f"MSE (Roughness): {mse_rough:.4f}")

# Plot comparison
fig, axes = plt.subplots(4, 2, figsize=(12, 10))
axes[0, 0].imshow(real, cmap='terrain'); axes[0, 0].set_title("Real DEM")
axes[0, 1].imshow(fake, cmap='terrain'); axes[0, 1].set_title("Fake DEM")
axes[1, 0].imshow(real_slope); axes[1, 0].set_title("Real Slope")
axes[1, 1].imshow(fake_slope); axes[1, 1].set_title("Fake Slope")
axes[2, 0].imshow(real_curv); axes[2, 0].set_title("Real Curvature")
axes[2, 1].imshow(fake_curv); axes[2, 1].set_title("Fake Curvature")
axes[3, 0].hist(real.flatten(), bins=50); axes[3, 0].set_title("Real Elevation Histogram")
axes[3, 1].hist(fake.flatten(), bins=50); axes[3, 1].set_title("Fake Elevation Histogram")

for ax in axes.flatten(): ax.axis('off')
axes[3, 0].axis('on'); axes[3, 1].axis('on')  # keep histograms visible

plt.tight_layout()
plt.savefig("real_vs_fake_detailed.png")
plt.show()
