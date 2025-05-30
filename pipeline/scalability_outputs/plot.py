import matplotlib.pyplot as plt


def main():
    # Data tuples: (label, pixel_count, total_file_size_in_KB)
    data = [
        ("single_128", 2 * (128 * 128), 31 + 26),
        ("single_256", 2 * (256 * 256), 113 + 96),
        ("single_512", 2 * (512 * 512), 381 + 431),
        ("single_1024", 2 * (1024 * 1024), 1108 + 1499),
        ("2x2_grid", 2 * ((2 * 1024) * (2 * 1024)), 1411 + 1538),
        ("4x4_grid", 2 * ((4 * 1024) * (4 * 1024)), 5559 + 1538),
        ("8x8_grid", 2 * ((8 * 1024) * (8 * 1024)), 21570 + 24869),
    ]

    # Separate the data by type
    single_x, single_y = [], []
    grid_x, grid_y = [], []

    for label, px, size in data:
        if "single" in label:
            single_x.append(px)
            single_y.append(size)
        else:
            grid_x.append(px)
            grid_y.append(size)

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Plot singles (blue circles)
    plt.scatter(single_x, single_y, color='blue', marker='o', label='single')

    # Plot big grids (crimson rectangles)
    plt.scatter(grid_x, grid_y, color='crimson', marker='s', label='big_grid')

    # Labels and legend
    plt.xlabel("Pixel Count")
    plt.ylabel("Total File Size (KB)")
    plt.title("File Size vs Pixel Count")
    plt.legend(loc="upper left")

    # Save the figure
    plt.tight_layout()
    plt.savefig("filesize_vs_pixelcount.png", dpi=300)
    print("Plot saved as 'filesize_vs_pixelcount.png'")


if __name__ == "__main__":
    main()
