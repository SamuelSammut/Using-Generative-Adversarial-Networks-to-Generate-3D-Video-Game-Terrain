import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

def median_smooth_dem(dem, size=3):
    """
    Smooths the DEM using a median filter.

    Parameters:
        dem (np.array): The DEM data to smooth.
        size (int): Size of the filtering window.

    Returns:
        np.array: Smoothed DEM.
    """
    return median_filter(dem, size=size)

def create_3d_terrain(rgb_file, dem_file, output_filename=None, smooth=False, smooth_window=3, elev=30, azim=-60, downsample_factor=1):
    """
    Generates a 3D terrain model using RGB and DEM data with optional DEM smoothing.

    Parameters:
        rgb_file (str): Path to the RGB .npy file.
        dem_file (str): Path to the DEM .npy file.
        output_filename (str, optional): Path to save the 3D terrain plot. If None, displays the plot.
        smooth (bool): Whether to apply smoothing to the DEM.
        smooth_window (int): Window size for median filter if smoothing is enabled.
        elev (float): Elevation angle in degrees.
        azim (float): Azimuthal angle in degrees.
        downsample_factor (int): Factor by which to downsample the dataset for faster rendering.
    """
    # Load RGB and DEM
    rgb = np.load(rgb_file)
    dem = np.load(dem_file)

    if smooth:
        print(f"Applying median filter with window size {smooth_window}...")
        dem = median_smooth_dem(dem, size=smooth_window)

    rgb = rgb[::downsample_factor, ::downsample_factor, :]
    dem = dem[::downsample_factor, ::downsample_factor]

    dem_normalized = (dem - dem.min()) / (dem.max() - dem.min())

    height, width = dem.shape
    x = np.arange(0, width)
    y = np.arange(0, height)
    x, y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(
        x, y, dem_normalized,
        rstride=5, cstride=5,
        facecolors=rgb,
        shade=False
    )

    ax.view_init(elev=elev, azim=azim)

    if output_filename:
        plt.savefig(output_filename)
        print(f"Saved 3D terrain to {output_filename}")
    else:
        plt.show()

create_3d_terrain(
    rgb_file="inference_outputs/final_inference_output_rgb.npy",
    dem_file="inference_outputs/final_inference_output_dem.npy",
    output_filename="3d_terrain.png",
    smooth=True,
    smooth_window=25,
    elev=45,
    azim=90,
    downsample_factor=2
)
