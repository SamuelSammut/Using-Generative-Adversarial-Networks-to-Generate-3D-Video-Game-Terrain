import rasterio
import numpy as np
import os
from rasterio.windows import Window

# Parameters
GRID_SIZE = 512
OUTPUT_DIR = "../output_tiles"

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/rgb", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/dem", exist_ok=True)

# Open RGB and DEM files
with rasterio.open('../big_rgb.tif') as rgb, rasterio.open('../big_dem.tif') as dem:
    # Read RGB and DEM data
    rgb_data = rgb.read()  # Read all RGB bands
    dem_data = dem.read(1)  # DEM is single-band

    # Create masks for NoData
    rgb_nodata = rgb.nodata or 0
    rgb_mask = np.all(rgb_data != rgb_nodata, axis=0)

    dem_nodata = dem.nodata or 0
    dem_mask = dem_data != dem_nodata

    # Combine masks
    valid_mask = np.logical_and(rgb_mask, dem_mask)

    # Save the valid mask
    with rasterio.open(
            os.path.join(OUTPUT_DIR, "valid_mask.tif"),
            "w",
            driver="GTiff",
            height=valid_mask.shape[0],
            width=valid_mask.shape[1],
            count=1,
            dtype=rasterio.uint8,
            crs=rgb.crs,
            transform=rgb.transform,
    ) as mask_dst:
        mask_dst.write(valid_mask.astype(np.uint8), 1)

    # Split into grid tiles
    grid_rows = rgb.height // GRID_SIZE
    grid_cols = rgb.width // GRID_SIZE

    for row in range(grid_rows):
        for col in range(grid_cols):
            # Calculate window for each tile
            window = Window(
                col_off=col * GRID_SIZE,
                row_off=row * GRID_SIZE,
                width=GRID_SIZE,
                height=GRID_SIZE,
            )

            # Read RGB and DEM for the current window
            rgb_tile = rgb.read(window=window)
            dem_tile = dem.read(1, window=window)
            mask_tile = valid_mask[
                        row * GRID_SIZE:(row + 1) * GRID_SIZE,
                        col * GRID_SIZE:(col + 1) * GRID_SIZE,
                        ]

            # Skip tiles with no valid data
            if np.sum(mask_tile) == 0:
                continue

            # Save the RGB tile
            with rasterio.open(
                    f"{OUTPUT_DIR}/rgb/rgb_{row}_{col}.tif",
                    "w",
                    driver="GTiff",
                    height=GRID_SIZE,
                    width=GRID_SIZE,
                    count=rgb.count,
                    dtype=rgb.dtypes[0],
                    crs=rgb.crs,
                    transform=rasterio.windows.transform(window, rgb.transform),
            ) as rgb_dst:
                rgb_dst.write(rgb_tile)

            # Save the DEM tile
            with rasterio.open(
                    f"{OUTPUT_DIR}/dem/dem_{row}_{col}.tif",
                    "w",
                    driver="GTiff",
                    height=GRID_SIZE,
                    width=GRID_SIZE,
                    count=1,
                    dtype=dem.dtypes[0],
                    crs=dem.crs,
                    transform=rasterio.windows.transform(window, dem.transform),
            ) as dem_dst:
                dem_dst.write(dem_tile, 1)
