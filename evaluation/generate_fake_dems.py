import os
import numpy as np
import tensorflow as tf
import rasterio
from rasterio.transform import from_origin

# Config
generator_path = "../saved_models/generator_model.h5"
output_folder = "fake_dems_for_compare"
noise_dim = 100
num_to_generate = 200
image_size = (512, 512)
batch_size = 32  # Adjust based on VRAM

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the trained generator
generator = tf.keras.models.load_model(generator_path, compile=False)

# Calculate number of batches
num_batches = int(np.ceil(num_to_generate / batch_size))
generated_count = 0

print("Starting generation of fake DEMs...")

for batch_idx in range(num_batches):
    current_batch_size = min(batch_size, num_to_generate - generated_count)
    noise = tf.random.normal([current_batch_size, noise_dim])
    generated_images = generator(noise, training=False).numpy()

    for i in range(current_batch_size):
        dem = generated_images[i, :, :, 3]  # Extract DEM channel
        dem = (dem + 1) / 2.0  # Scale from [-1, 1] to [0, 1]

        # Save as .tif using rasterio
        filepath = os.path.join(output_folder, f"fake_dem_{generated_count + i}.tif")

        with rasterio.open(
            filepath,
            'w',
            driver='GTiff',
            height=image_size[0],
            width=image_size[1],
            count=1,
            dtype=dem.dtype,
            crs='+proj=latlong',
            transform=from_origin(0, 0, 1, 1)  # Dummy transform
        ) as dst:
            dst.write(dem, 1)

    generated_count += current_batch_size
    print(f"Saved {generated_count}/{num_to_generate} fake DEMs")

print("✅ All fake DEMs saved to:", output_folder)
