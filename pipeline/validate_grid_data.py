import os
from os.path import join

# Define directories
output_dir = "../output_tiles"
rgb_dir = join(output_dir, "rgb")
dem_dir = join(output_dir, "dem")

# List all RGB and DEM files
rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".tif")])
dem_files = sorted([f for f in os.listdir(dem_dir) if f.endswith(".tif")])

# Check if the counts match
if len(rgb_files) != len(dem_files):
    print(f"Mismatch in tile counts! RGB: {len(rgb_files)}, DEM: {len(dem_files)}")
else:
    print(f"Tile counts match: {len(rgb_files)} each.")

# Ensure file pairs are aligned
mismatched_pairs = []
for rgb_file, dem_file in zip(rgb_files, dem_files):
    # Extract grid positions from file names
    rgb_position = rgb_file.replace("rgb_", "").replace(".tif", "")
    dem_position = dem_file.replace("dem_", "").replace(".tif", "")

    if rgb_position != dem_position:
        mismatched_pairs.append((rgb_file, dem_file))

# Output results
if mismatched_pairs:
    print("Mismatched tile pairs found:")
    for rgb_file, dem_file in mismatched_pairs:
        print(f"  RGB: {rgb_file}  DEM: {dem_file}")
else:
    print("All tile pairs are aligned properly.")
