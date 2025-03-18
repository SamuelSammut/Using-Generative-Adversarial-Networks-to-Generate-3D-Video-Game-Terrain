import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # for saving RGB as PNG

SCALE_FACTOR = 1.0
Z_SCALE = 10.0

def shift_dem_to_match_edge(dem_a, dem_b, overlap='right'):
    """
    Shift dem_b's elevations so that its shared edge with dem_a is closer in value.
    """
    if overlap == 'right':
        edge_a = dem_a[:, -1]
        edge_b = dem_b[:, 0]
    elif overlap == 'bottom':
        edge_a = dem_a[-1, :]
        edge_b = dem_b[0, :]
    else:
        raise ValueError("Unsupported overlap direction.")

    diff = edge_a - edge_b
    shift_val = np.mean(diff)

    dem_b_shifted = dem_b + shift_val
    return dem_b_shifted, shift_val

def blend_edge(rgb_a, dem_a, rgb_b, dem_b, blend_width=15, overlap='right'):
    """
    Blend edges of two adjacent tiles (both RGB and DEM) in the overlapping boundary region.
    """
    if overlap == 'right':
        alpha = np.linspace(0, 1, blend_width).reshape(1, -1, 1)

        # Blend RGB
        rgb_a[:, -blend_width:, :] = (1 - alpha) * rgb_a[:, -blend_width:, :] + alpha * rgb_b[:, :blend_width, :]
        rgb_b[:, :blend_width, :] = alpha * rgb_b[:, :blend_width, :] + (1 - alpha) * rgb_a[:, -blend_width:, :]

        # Blend DEM
        alpha_dem = alpha.squeeze(-1)
        dem_a[:, -blend_width:] = (1 - alpha_dem) * dem_a[:, -blend_width:] + alpha_dem * dem_b[:, :blend_width]
        dem_b[:, :blend_width] = alpha_dem * dem_b[:, :blend_width] + (1 - alpha_dem) * dem_a[:, -blend_width:]

    return rgb_a, dem_a, rgb_b, dem_b

def combine_tiles_in_row(rgb_list, dem_list, do_edge_matching=True, blend_width=15):
    """
    Combine multiple tiles side-by-side in a single row.
    """
    if len(rgb_list) == 0:
        return None, None

    final_rgb = rgb_list[0].copy()
    final_dem = dem_list[0].copy()

    for i in range(1, len(rgb_list)):
        current_rgb = rgb_list[i].copy()
        current_dem = dem_list[i].copy()

        if do_edge_matching:
            current_dem, shift_val = shift_dem_to_match_edge(final_dem, current_dem, overlap='right')

        if blend_width > 0:
            final_rgb, final_dem, current_rgb, current_dem = blend_edge(
                final_rgb, final_dem, current_rgb, current_dem, blend_width=blend_width, overlap='right'
            )

        final_rgb = np.concatenate([final_rgb, current_rgb], axis=1)
        final_dem = np.concatenate([final_dem, current_dem], axis=1)

    return final_rgb, final_dem

def visualize_3d(rgb, dem, output_path=None):
    """
    Creates a 3D surface plot from the combined tile.
    """
    h, w, _ = rgb.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    dem_norm = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, dem_norm, rstride=1, cstride=1, facecolors=rgb, shade=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Normalized Elevation')
    ax.view_init(45, -30)

    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Saved combined 3D terrain to {output_path}")
    else:
        plt.show()

def save_as_obj_with_texture(rgb, dem, obj_filename, png_filename, mtl_filename):
    """
    Saves the combined terrain as a Wavefront .obj with a texture.
    """
    # Save texture as PNG
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(rgb_uint8).save(png_filename)
    print(f"Saved texture to {png_filename}")

    H, W = dem.shape
    with open(obj_filename, 'w') as fobj:
        fobj.write("# Exported Terrain OBJ\n")
        fobj.write(f"mtllib {os.path.basename(mtl_filename)}\n")

        print(f"Exporting OBJ: {obj_filename}, Size: {H} x {W}")

        # Write vertices
        for row in range(H):
            for col in range(W):
                x = col * SCALE_FACTOR
                y = row * SCALE_FACTOR
                z = dem[row, col] * Z_SCALE
                u = col / (W - 1) if W > 1 else 0
                v = 1.0 - (row / (H - 1) if H > 1 else 0)
                fobj.write(f"v {x} {y} {z}\n")
                fobj.write(f"vt {u} {v}\n")

        # Write faces
        def idx(r, c):
            return r * W + c + 1  # 1-based indexing in OBJ

        for r in range(H - 1):
            for c in range(W - 1):
                fobj.write(f"f {idx(r,c)} {idx(r,c+1)} {idx(r+1,c)}\n")
                fobj.write(f"f {idx(r+1,c)} {idx(r,c+1)} {idx(r+1,c+1)}\n")

    print(f"Saved mesh to {obj_filename}")

    # Write MTL file
    with open(mtl_filename, 'w') as fmtl:
        fmtl.write("newmtl terrain_material\n")
        fmtl.write("Ka 1.0 1.0 1.0\n")
        fmtl.write("Kd 1.0 1.0 1.0\n")
        fmtl.write("Ks 0.0 0.0 0.0\n")
        fmtl.write(f"map_Kd {os.path.basename(png_filename)}\n")

    print(f"Saved material to {mtl_filename}")

def load_tiles(folder, prefix, count):
    """
    Loads multiple (rgb, dem) pairs.
    """
    rgb_list = []
    dem_list = []
    for i in range(count):
        rgb_file = os.path.join(folder, f"{prefix}_{i}_rgb.npy")
        dem_file = os.path.join(folder, f"{prefix}_{i}_dem.npy")

        if not os.path.isfile(rgb_file) or not os.path.isfile(dem_file):
            print(f"Missing tile: {i}")
            continue

        rgb = np.load(rgb_file)
        dem = np.load(dem_file)

        if rgb.shape[:2] != dem.shape:
            raise ValueError(f"Shape mismatch at {i}!")

        rgb_list.append(np.clip(rgb, 0, 1))
        dem_list.append(dem)

    return rgb_list, dem_list

if __name__ == "__main__":
    folder = "inference_outputs"
    prefix = "output"
    num_tiles = 10

    rgb_list, dem_list = load_tiles(folder, prefix, num_tiles)
    combined_rgb, combined_dem = combine_tiles_in_row(rgb_list, dem_list, blend_width=15)

    np.save(os.path.join(folder, "combined_rgb.npy"), combined_rgb)
    np.save(os.path.join(folder, "combined_dem.npy"), combined_dem)

    visualize_3d(combined_rgb, combined_dem, output_path=os.path.join(folder, "combined_terrain_3d.png"))

    save_as_obj_with_texture(
        combined_rgb,
        combined_dem,
        os.path.join(folder, "combined_terrain.obj"),
        os.path.join(folder, "combined_terrain.png"),
        os.path.join(folder, "combined_terrain.mtl"),
    )
