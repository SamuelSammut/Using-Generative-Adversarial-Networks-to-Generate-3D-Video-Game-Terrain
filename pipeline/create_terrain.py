import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import rasterio
from rasterio.transform import from_origin
from PIL import Image
import re

def blend_block_into(big_rgb, big_dem, block_rgb, block_dem, top_left, overlap):
    """
    Blends a single (block_rgb, block_dem) into the large (big_rgb, big_dem)
    arrays at position `top_left` using an overlap-pixel-wide linear blend.
    """
    H_block, W_block, _ = block_rgb.shape
    y0, x0 = top_left

    y1 = y0 + H_block
    x1 = x0 + W_block

    # Existing slices in the big arrays
    big_slice_rgb = big_rgb[y0:y1, x0:x1, :]
    big_slice_dem = big_dem[y0:y1, x0:x1]

    # Create a mask for blending
    mask = np.ones((H_block, W_block), dtype=np.float32)

    # Vertical overlap fade (top edge)
    if y0 > 0:
        for r in range(overlap):
            alpha = r / float(overlap)
            mask[r, :] = alpha

    # Horizontal overlap fade (left edge)
    if x0 > 0:
        for c in range(overlap):
            alpha = c / float(overlap)
            mask[:, c] = np.maximum(mask[:, c], alpha)

    # Apply the blend: final = (1 - mask)*existing + mask*new_block
    mask_rgb = np.expand_dims(mask, axis=-1)
    big_rgb[y0:y1, x0:x1, :] = (1.0 - mask_rgb) * big_slice_rgb + mask_rgb * block_rgb
    big_dem[y0:y1, x0:x1] = (1.0 - mask) * big_slice_dem + mask * block_dem


def generate_single_block(generator, noise_dim=100, out_height=256, out_width=256):
    """
    Generates one (RGB, DEM) block from the GAN. Returns float32 arrays [0,1]
    of shape (out_height, out_width, 3) and (out_height, out_width).
    """
    # 1) Make random noise
    noise = tf.random.normal([1, noise_dim])

    # 2) Generate image [H, W, 4] from the GAN
    generated_image = generator(noise, training=False).numpy()[0]  # [-1, 1]

    # 3) Scale to [0,1]
    generated_image = (generated_image + 1.0) / 2.0
    generated_image = np.clip(generated_image, 0.0, 1.0)

    # 4) Separate out RGB & DEM
    rgb_raw = generated_image[..., :3]  # shape e.g. (H_in, W_in, 3)
    dem_raw = generated_image[..., 3]   # shape e.g. (H_in, W_in)

    # 5) Resize them to (out_height, out_width) with TensorFlow
    #    so each block has a consistent size for final stitching
    rgb_resized = tf.image.resize(rgb_raw, [out_height, out_width]).numpy()
    dem_resized = tf.image.resize(dem_raw[..., None], [out_height, out_width]).numpy()[..., 0]

    # Return float32 in [0,1]
    return rgb_resized.astype(np.float32), dem_resized.astype(np.float32)


def compute_edge_mismatch(block_dem, final_dem, top_left, overlap):
    """
    Computes how well 'block_dem' matches the existing 'final_dem' in the overlap edges.
    Lower cost => better match.
    """
    H_block, W_block = block_dem.shape
    y0, x0 = top_left
    cost = 0.0

    # Top overlap region
    if y0 > 0:
        final_slice_top = final_dem[y0:y0+overlap, x0:x0+W_block]
        block_slice_top = block_dem[0:overlap, :]
        if final_slice_top.shape == block_slice_top.shape:
            cost_top = np.sum(np.abs(final_slice_top - block_slice_top))
            cost += cost_top

    # Left overlap region
    if x0 > 0:
        final_slice_left = final_dem[y0:y0+H_block, x0:x0+overlap]
        block_slice_left = block_dem[:, 0:overlap]
        if final_slice_left.shape == block_slice_left.shape:
            cost_left = np.sum(np.abs(final_slice_left - block_slice_left))
            cost += cost_left

    return cost


def generate_block_with_edge_match(generator, final_dem, top_left, overlap, tries, noise_dim,
                                   out_height, out_width):
    """
    Generate multiple candidate blocks, pick the one with the lowest mismatch cost
    for better seamless edges.
    """
    best_rgb = None
    best_dem = None
    best_cost = float('inf')

    for _ in range(tries):
        block_rgb, block_dem = generate_single_block(
            generator,
            noise_dim=noise_dim,
            out_height=out_height,
            out_width=out_width
        )
        cost = compute_edge_mismatch(block_dem, final_dem, top_left, overlap)

        if cost < best_cost:
            best_cost = cost
            best_rgb = block_rgb
            best_dem = block_dem

    return best_rgb, best_dem


def generate_big_terrain(generator,
                         grid_rows=2,
                         grid_cols=2,
                         overlap=16,
                         noise_dim=100,
                         edge_match_tries=5,
                         target_final_size=1025):
    """
    Generates one large terrain by tiling grid_rows x grid_cols blocks.
    The final dimension is about `target_final_size x target_final_size`.

    Steps:
    1) Compute block sizes so the final image is ~ target_final_size in both width & height.
    2) Generate each block with `generate_block_with_edge_match()` so edges match better.
    3) Blend them together with `blend_block_into()`.
    """

    # Calculate how big each tile should be in height & width
    # so that the final dimension ~ target_final_size x target_final_size.
    # final_height = H_block + (grid_rows-1)*(H_block-overlap)
    # => H_block ~ (target_final_size + (grid_rows-1)*overlap) / grid_rows
    H_block = (target_final_size + (grid_rows - 1) * overlap) // grid_rows
    W_block = (target_final_size + (grid_cols - 1) * overlap) // grid_cols

    # Create the final arrays
    H_big = H_block + (grid_rows - 1) * (H_block - overlap)
    W_big = W_block + (grid_cols - 1) * (W_block - overlap)

    final_rgb = np.zeros((H_big, W_big, 3), dtype=np.float32)
    final_dem = np.zeros((H_big, W_big), dtype=np.float32)

    # Generate the first block and place it at (0,0)
    first_rgb, first_dem = generate_single_block(
        generator,
        noise_dim=noise_dim,
        out_height=H_block,
        out_width=W_block
    )
    blend_block_into(final_rgb, final_dem, first_rgb, first_dem, (0, 0), overlap)

    # Fill the rest of the blocks
    for r in range(grid_rows):
        for c in range(grid_cols):
            if r == 0 and c == 0:
                continue

            y0 = r * (H_block - overlap)
            x0 = c * (W_block - overlap)

            best_rgb, best_dem = generate_block_with_edge_match(
                generator,
                final_dem,
                top_left=(y0, x0),
                overlap=overlap,
                tries=edge_match_tries,
                noise_dim=noise_dim,
                out_height=H_block,
                out_width=W_block
            )
            blend_block_into(final_rgb, final_dem, best_rgb, best_dem, (y0, x0), overlap)

    return final_rgb, final_dem


def save_rgb_dem(rgb, dem, rgb_filename, dem_filename):
    """
    Save the RGB and DEM arrays to PNG files:
      - rgb in 8-bit
      - dem in 16-bit
    """
    rgb_8bit = (rgb * 255).astype(np.uint8)
    Image.fromarray(rgb_8bit).save(rgb_filename, "PNG")

    dem_16bit = (dem * 65535).astype(np.uint16)
    Image.fromarray(dem_16bit, mode="I;16").save(dem_filename, "PNG")


def save_dem_as_tif(dem, dem_filename):
    """
    Saves a DEM (2D float32 numpy array) as a GeoTIFF (.tif).
    """
    height, width = dem.shape
    transform = from_origin(0, 0, 1, 1)  # dummy geotransform

    with rasterio.open(
        dem_filename, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype='float32',
        crs='+proj=latlong',
        transform=transform
    ) as dst:
        dst.write(dem.astype(np.float32), 1)


def generate_terrain_blocks(generator_path,
                            num_terrains=1,
                            noise_dim=100,
                            output_folder="terrain_outputs",
                            big_grid=None,
                            overlap=16,
                            edge_match_tries=5,
                            target_final_size=1025):
    """
    Generates terrain data using a pre-trained GAN generator model.

    Modes:
    1) If big_grid is None:
       - Generate 'num_terrains' individual blocks (each resized to target_final_size x target_final_size).

    2) If big_grid is set (e.g. "4x4"):
       - Create one large terrain with 'grid_rows x grid_cols' blocks, aiming
         for a final resolution ~ target_final_size x target_final_size.
       - Try multiple candidates per block to minimize edge mismatch, then blend them.

    Args:
        generator_path (str): Path to the pre-trained generator (.h5 file).
        num_terrains (int): Number of terrain blocks (only if big_grid is None).
        noise_dim (int): Dimension of the generator's noise vector.
        output_folder (str): Folder to store output images.
        big_grid (str): e.g. "2x3" => 2 rows, 3 cols. If given, only 1 large terrain is produced.
        overlap (int): Overlap in pixels for blending adjacent blocks.
        edge_match_tries (int): Times to re-generate a block to find best edge match.
        target_final_size (int): Desired final dimension in height & width (approx) for large terrain,
                                 or the size of single blocks if not using big_grid.
    """
    os.makedirs(output_folder, exist_ok=True)

    print(f"Loading generator from: {generator_path}")
    generator = load_model(generator_path, compile=False)
    print("Generator loaded successfully.")

    if big_grid is None:
        # Mode 1: Generate individual blocks
        for i in range(num_terrains):
            # Generate a single block at target_final_size x target_final_size
            rgb, dem = generate_single_block(
                generator,
                noise_dim=noise_dim,
                out_height=target_final_size,
                out_width=target_final_size
            )

            rgb_filename = os.path.join(output_folder, f"terrain_{i:04d}_rgb.png")
            dem_filename = os.path.join(output_folder, f"terrain_{i:04d}_height.tif")

            save_dem_as_tif(dem, dem_filename)
            print(f"Saved terrain {i+1}/{num_terrains}:")
            print(f"  RGB -> {rgb_filename}")
            print(f"  DEM -> {dem_filename}")

        print("Generation complete (individual blocks).")

    else:
        # Mode 2: Generate one large terrain
        match = re.match(r"^(\d+)x(\d+)$", big_grid.strip().lower())
        if not match:
            raise ValueError("Invalid format for --big-grid. Expected something like '4x4' or '10x5'.")

        grid_rows = int(match.group(1))
        grid_cols = int(match.group(2))

        print(f"Generating one large terrain of size {grid_rows} x {grid_cols} blocks...")
        print(f"Target final size ~ {target_final_size} x {target_final_size}")

        final_rgb, final_dem = generate_big_terrain(
            generator,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            overlap=overlap,
            noise_dim=noise_dim,
            edge_match_tries=edge_match_tries,
            target_final_size=target_final_size
        )

        rgb_filename = os.path.join(output_folder, "large_terrain_rgb.png")
        dem_filename = os.path.join(output_folder, "large_terrain_height.tif"
                                                   "")

        save_dem_as_tif(final_dem, dem_filename)

        print("Saved single large terrain:")
        print(f"  RGB -> {rgb_filename}")
        print(f"  DEM -> {dem_filename}")
        print("Generation complete (large seamless terrain).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate terrain blocks (RGB & 16-bit heightmaps) "
                    "with optional large seamless combination."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained generator (.h5 file)."
    )
    parser.add_argument(
        "--num-terrains",
        type=int,
        default=1,
        help="Number of terrain blocks to generate (if not creating a large terrain)."
    )
    parser.add_argument(
        "--noise-dim",
        type=int,
        default=100,
        help="Dimension of the noise vector for the generator."
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="terrain_outputs",
        help="Folder to store generated terrain images."
    )
    parser.add_argument(
        "--big-grid",
        type=str,
        default=None,
        help="If provided (e.g. '2x3'), create one large terrain by tiling blocks in a grid."
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=16,
        help="Number of pixels to blend between adjacent blocks in the large terrain."
    )
    parser.add_argument(
        "--edge-match-tries",
        type=int,
        default=5,
        help="Number of times to re-generate a block to find a better edge match."
    )
    parser.add_argument(
        "--target-final-size",
        type=int,
        default=1025,
        help="Desired final dimension for the large terrain or single block (in px). "
             "E.g. 1025, 2049, 4033, etc. For big_grid, the final resolution is approx."
    )

    args = parser.parse_args()

    generate_terrain_blocks(
        generator_path=args.model_path,
        num_terrains=args.num_terrains,
        noise_dim=args.noise_dim,
        output_folder=args.output_folder,
        big_grid=args.big_grid,
        overlap=args.overlap,
        edge_match_tries=args.edge_match_tries,
        target_final_size=args.target_final_size
    )
