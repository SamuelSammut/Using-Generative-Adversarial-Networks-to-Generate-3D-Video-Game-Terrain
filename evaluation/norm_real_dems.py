import os
import numpy as np
import rasterio
from rasterio.enums import Resampling

# config
input_folder = "./real_dems_for_compare"  # path to real dems
output_folder = "./real_dems_for_compare"
target_range = (0, 1000)  # Match the fake DEM value range

os.makedirs(output_folder, exist_ok=True)

def normalize_dem(dem, new_min=0, new_max=1000):
    old_min = np.min(dem)
    old_max = np.max(dem)
    if old_max == old_min:
        return np.zeros_like(dem, dtype=np.float32)
    norm = (dem - old_min) / (old_max - old_min)
    scaled = norm * (new_max - new_min) + new_min
    return scaled.astype(np.float32)

# process files
input_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".tif")])

print(f"Found {len(input_files)} real DEMs to normalize...")

for i, filename in enumerate(input_files):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    with rasterio.open(input_path) as src:
        dem = src.read(1)
        profile = src.profile

        dem_normalized = normalize_dem(dem, *target_range)

        # Update profile
        profile.update(dtype=rasterio.float32, count=1)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(dem_normalized, 1)

    print(f"[{i+1}/{len(input_files)}] Saved normalized DEM to {output_path}")

print("\nAll real DEMs normalized to float32 and saved in:", output_folder)
