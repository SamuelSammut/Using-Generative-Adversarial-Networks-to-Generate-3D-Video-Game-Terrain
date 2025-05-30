import os
import numpy as np
from scipy.ndimage import zoom

# config
input_folder = "./preprocessed_data"
output_folder = "./preprocessed_data_resized"
target_size = (256, 256)

os.makedirs(output_folder, exist_ok=True)

files = [f for f in os.listdir(input_folder) if f.endswith(".npy")]
count = 0

for filename in files:
    path = os.path.join(input_folder, filename)
    arr = np.load(path)

    if arr.shape[:2] == target_size:
        print(f"Skipping {filename} (already 256x256)")
        continue

    zoom_factors = (
        target_size[0] / arr.shape[0],
        target_size[1] / arr.shape[1],
        1  # keep channel count (RGB+DEM)
    )
    resized = zoom(arr, zoom_factors, order=1)

    np.save(os.path.join(output_folder, filename), resized)
    count += 1
    if count % 25 == 0:
        print(f"[{count}] Resized: {filename}")

print(f"✅ Resized {count} files to {target_size} and saved to: {output_folder}")
