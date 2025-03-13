import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image  # For saving 16-bit PNG
import re

def blend_block_into(big_rgb, big_dem, block_rgb, block_dem, top_left, overlap):
    """
    Blends a single (block_rgb, block_dem) into the large (big_rgb, big_dem)
    arrays at position `top_left` using an `overlap`-pixel-wide linear blend
    with any already-placed content.

    big_rgb, block_rgb are float32 arrays of shape (H_big, W_big, 3) and (H_block, W_block, 3).
    big_dem, block_dem are float32 arrays of shape (H_big, W_big) and (H_block, W_block).
    top_left = (y, x) indicates where block[0,0] maps in the big arrays.
    overlap is the size (in pixels) for linear blending on the edges:
       - horizontally near x-boundary
       - vertically near y-boundary
    """
    H_block, W_block, _ = block_rgb.shape
    y0, x0 = top_left

    y1 = y0 + H_block
    x1 = x0 + W_block

    big_slice_rgb = big_rgb[y0:y1, x0:x1, :]
    big_slice_dem = big_dem[y0:y1, x0:x1]

    mask = np.ones((H_block, W_block), dtype=np.float32)

    if y0 > 0:
        for r in range(overlap):
            alpha = r / float(overlap)
            mask[r, :] = alpha

    if x0 > 0:
        for c in range(overlap):
            alpha = c / float(overlap)
            mask[:, c] = np.maximum(mask[:, c], alpha)

    #   final = (1 - mask)*existing + mask*new_block
    mask_rgb = np.expand_dims(mask, axis=-1)
    big_rgb[y0:y1, x0:x1, :] = (1.0 - mask_rgb) * big_slice_rgb + mask_rgb * block_rgb
    big_dem[y0:y1, x0:x1] = (1.0 - mask) * big_slice_dem + mask * block_dem


def generate_single_block(generator, noise_dim=100):
    """
    Generates a single (RGB, DEM) block using the generator. Returns
    two NumPy arrays in float32, scaled to [0,1].
    """
    # 1. Create random noise vector
    noise = tf.random.normal([1, noise_dim])

    # 2. Generate image [H,W,4]
    generated_image = generator(noise, training=False).numpy()[0]
    # Range is [-1,1]. Scale to [0,1]
    generated_image = (generated_image + 1.0) / 2.0
    generated_image = np.clip(generated_image, 0.0, 1.0)

    # 3. Separate RGB and DEM
    rgb = generated_image[..., :3].astype(np.float32)  # (H, W, 3)
    dem = generated_image[..., 3].astype(np.float32)   # (H, W)

    return rgb, dem


def compute_edge_mismatch(block_dem, final_dem, top_left, overlap):
    """
    Computes how well 'block_dem' matches the existing 'final_dem' in the overlap edges.

    block_dem shape: (H_block, W_block)
    final_dem shape: (H_big, W_big)
    top_left = (y0, x0) in final_dem where the top-left of block_dem is placed.
    overlap (int) = number of pixels that overlap.

    Returns a single scalar 'cost' (the lower, the better match).
    """
    H_block, W_block = block_dem.shape
    y0, x0 = top_left

    cost = 0.0

    # If there's a block above -> overlap in top
    if y0 > 0:
        final_slice_top = final_dem[y0: y0 + overlap, x0: x0 + W_block]
        block_slice_top = block_dem[0: overlap, :]

        if final_slice_top.shape == block_slice_top.shape:
            cost_top = np.sum(np.abs(final_slice_top - block_slice_top))
            cost += cost_top

    # If there's a block to the left -> overlap in left
    if x0 > 0:
        final_slice_left = final_dem[y0: y0 + H_block, x0: x0 + overlap]
        block_slice_left = block_dem[:, 0: overlap]

        if final_slice_left.shape == block_slice_left.shape:
            cost_left = np.sum(np.abs(final_slice_left - block_slice_left))
            cost += cost_left

    return cost


def generate_block_with_edge_match(generator, final_dem, top_left, overlap, tries, noise_dim):
    """
    Generate multiple candidate blocks, compute edge mismatch vs 'final_dem',
    and return the block (rgb, dem) with the smallest mismatch cost.

    Args:
        generator: the loaded GAN generator
        final_dem: big DEM array so far (float32, shape [H_big, W_big])
        top_left: (y0, x0) in final_dem where block is to be placed
        overlap (int): overlap size in pixels
        tries (int): how many times to generate a candidate block
        noise_dim (int): dimension of the noise vector

    Returns:
        (best_rgb, best_dem): float32 arrays in [0,1]
    """
    best_rgb = None
    best_dem = None
    best_cost = float('inf')

    for _ in range(tries):
        block_rgb, block_dem = generate_single_block(generator, noise_dim=noise_dim)
        cost = compute_edge_mismatch(block_dem, final_dem, top_left, overlap)

        if cost < best_cost:
            best_cost = cost
            best_rgb = block_rgb
            best_dem = block_dem

    return best_rgb, best_dem


def generate_big_terrain(
        generator,
        grid_rows=2,
        grid_cols=2,
        overlap=16,
        noise_dim=100,
        edge_match_tries=5
):
    """
    Generates one large terrain by tiling grid_rows x grid_cols blocks.
    Blends edges so that seams are less obvious. Also tries to match
    the DEM edges by regenerating blocks if the mismatch is too big.

    Returns:
       final_rgb (H_big, W_big, 3) in [0,1] float32
       final_dem (H_big, W_big) in [0,1] float32
    """
    block_rgb, block_dem = generate_single_block(generator, noise_dim=noise_dim)
    H_block, W_block, _ = block_rgb.shape

    H_big = H_block + (grid_rows - 1) * (H_block - overlap)
    W_big = W_block + (grid_cols - 1) * (W_block - overlap)

    final_rgb = np.zeros((H_big, W_big, 3), dtype=np.float32)
    final_dem = np.zeros((H_big, W_big), dtype=np.float32)

    blend_block_into(final_rgb, final_dem, block_rgb, block_dem, (0, 0), overlap)

    # Fill the rest
    for r in range(grid_rows):
        for c in range(grid_cols):
            if r == 0 and c == 0:
                continue  # already placed the first block

            y0 = r * (H_block - overlap)
            x0 = c * (W_block - overlap)

            # Generate multiple candidates; pick one with the best edge match
            best_rgb, best_dem = generate_block_with_edge_match(
                generator,
                final_dem,
                top_left=(y0, x0),
                overlap=overlap,
                tries=edge_match_tries,
                noise_dim=noise_dim
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


def generate_terrain_blocks(
        generator_path,
        num_terrains=1,
        noise_dim=100,
        output_folder="terrain_outputs",
        big_grid=None,
        overlap=16,
        edge_match_tries=5
):
    """
    Generates terrain data using a pre-trained GAN generator model.

    Two modes:

    1) If big_grid is None:
       - Generate 'num_terrains' blocks, each with its own RGB (8-bit) and DEM (16-bit).

    2) If big_grid is NOT None:
       - Parse big_grid as something like "4x4" => grid_rows=4, grid_cols=4
       - Generate (4*4) blocks internally and:
           (a) try multiple times for each block,
           (b) pick the candidate whose DEM edges best match the existing big DEM,
           (c) blend them all into one large seamless terrain.
         Produce one large RGB and one large DEM as output.

    Args:
        generator_path (str): Path to the pre-trained generator (.h5 file).
        num_terrains (int): Number of terrain blocks to generate (only if big_grid is None).
        noise_dim (int): Dimension of the generator's noise vector.
        output_folder (str): Folder to store output images.
        big_grid (str): e.g. "2x3" => 2 rows, 3 cols. If given, only 1 large terrain is produced.
        overlap (int): Overlap in pixels for blending adjacent blocks in the large terrain.
        edge_match_tries (int): Number of times to regenerate a block for better edge matching.
    """
    os.makedirs(output_folder, exist_ok=True)

    print(f"Loading generator from: {generator_path}")
    generator = load_model(generator_path, compile=False)
    print("Generator loaded successfully.")

    if big_grid is None:
        # -- Mode 1: Generate individual blocks
        for i in range(num_terrains):
            rgb, dem = generate_single_block(generator, noise_dim=noise_dim)

            rgb_filename = os.path.join(output_folder, f"terrain_{i:04d}_rgb.png")
            dem_filename = os.path.join(output_folder, f"terrain_{i:04d}_height.png")

            save_rgb_dem(rgb, dem, rgb_filename, dem_filename)

            print(f"Saved terrain {i + 1}/{num_terrains}:")
            print(f"  RGB -> {rgb_filename}")
            print(f"  DEM -> {dem_filename}")

        print("Generation complete (individual blocks).")

    else:
        # -- Mode 2: Create one large terrain
        match = re.match(r"^(\d+)x(\d+)$", big_grid.strip().lower())
        if not match:
            raise ValueError("Invalid format for --big-grid. Expected something like '4x4'.")

        grid_rows = int(match.group(1))
        grid_cols = int(match.group(2))

        print(f"Generating one large terrain of size {grid_rows} x {grid_cols} blocks...")

        final_rgb, final_dem = generate_big_terrain(
            generator,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            overlap=overlap,
            noise_dim=noise_dim,
            edge_match_tries=edge_match_tries
        )

        rgb_filename = os.path.join(output_folder, "large_terrain_rgb.png")
        dem_filename = os.path.join(output_folder, "large_terrain_height.png")

        save_rgb_dem(final_rgb, final_dem, rgb_filename, dem_filename)

        print("Saved single large terrain:")
        print(f"  RGB -> {rgb_filename}")
        print(f"  DEM -> {dem_filename}")

        print("Generation complete (large seamless terrain).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate terrain blocks (RGB & 16-bit heightmaps) with optional large, seamless combination."
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

    args = parser.parse_args()

    generate_terrain_blocks(
        generator_path=args.model_path,
        num_terrains=args.num_terrains,
        noise_dim=args.noise_dim,
        output_folder=args.output_folder,
        big_grid=args.big_grid,
        overlap=args.overlap,
        edge_match_tries=args.edge_match_tries
    )
