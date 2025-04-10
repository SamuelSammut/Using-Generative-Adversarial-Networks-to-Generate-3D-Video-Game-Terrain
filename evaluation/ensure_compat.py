import os
import rasterio
import numpy as np

# === CONFIGURATION ===

real_folder = "./real_dems_for_compare" # or wherever your real DEMs are
fake_folder = "./fake_dems_for_compare"
num_samples = 10  # How many real/fake pairs to check

# === HELPER FUNCTION ===
def load_dem(path):
    with rasterio.open(path) as src:
        dem = src.read(1)
        profile = src.profile
    return dem, profile

# === COLLECT FILES ===
real_files = sorted([f for f in os.listdir(real_folder) if f.endswith(".tif")])
fake_files = sorted([f for f in os.listdir(fake_folder) if f.endswith(".tif")])

num_samples = min(num_samples, len(real_files), len(fake_files))

print(f"Comparing {num_samples} real/fake DEM pairs...\n")

for i in range(num_samples):
    real_path = os.path.join(real_folder, real_files[i])
    fake_path = os.path.join(fake_folder, fake_files[i])

    real_dem, real_meta = load_dem(real_path)
    fake_dem, fake_meta = load_dem(fake_path)

    print(f"--- Pair {i + 1}: {real_files[i]} vs {fake_files[i]} ---")

    # Shape
    print(f"  Shape: Real {real_dem.shape}, Fake {fake_dem.shape}")

    # Dtype
    print(f"  Dtype: Real {real_dem.dtype}, Fake {fake_dem.dtype}")

    # Range
    print(f"  Value Range: Real [{np.min(real_dem):.4f}, {np.max(real_dem):.4f}], "
          f"Fake [{np.min(fake_dem):.4f}, {np.max(fake_dem):.4f}]")

    # Check for NaNs
    print(f"  NaNs present? Real: {np.isnan(real_dem).any()}, Fake: {np.isnan(fake_dem).any()}")

    # CRS & Transform
    print(f"  CRS match? {real_meta['crs'] == fake_meta['crs']}")
    print(f"  Resolution match? {real_meta['transform'] == fake_meta['transform']}")
    print()

print("Comparison complete.")
