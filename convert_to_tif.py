import os
import glob
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin


def convert_png_to_tif_v2(input_folder, output_folder, resize=False, target_size=(512, 512)):
    """
    Converts all PNG images in a folder to GeoTIFFs.

    Parameters:
    - input_folder: Path to the folder containing input PNG images.
    - output_folder: Path to the folder where output TIFFs will be saved.
    - resize: Boolean indicating whether to resize images.
    - target_size: Tuple specifying target size if resizing (width, height).
    """
    os.makedirs(output_folder, exist_ok=True)
    png_files = glob.glob(os.path.join(input_folder, "*.png"))

    for png_path in png_files:
        # Load and convert image
        img = Image.open(png_path).convert("L")  # Convert to grayscale

        if resize:
            img = img.resize(target_size, Image.BICUBIC)

        dem_array = np.array(img).astype(np.float32)
        dem_array_scaled = (dem_array / 255.0) * 1000  # Elevation scaling: 0–1000m

        # Save as GeoTIFF
        filename = os.path.splitext(os.path.basename(png_path))[0] + '.tif'
        tif_path = os.path.join(output_folder, filename)

        height, width = dem_array_scaled.shape
        transform = from_origin(0, 0, 1, 1)  # Dummy geotransform

        with rasterio.open(
            tif_path, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype='float32',
            crs='+proj=latlong',
            transform=transform
        ) as dst:
            dst.write(dem_array_scaled, 1)

    return f"Converted {len(png_files)} files from '{input_folder}' to '{output_folder}'"


# Example call for illustration (not executed here):
convert_png_to_tif_v2('terrain_outputs/dem', 'fake_dem_for_grass', resize=False)

