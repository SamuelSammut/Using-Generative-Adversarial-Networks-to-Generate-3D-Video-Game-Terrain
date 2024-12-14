import random
import ee

# Authenticating and initializing Earth Engine
ee.Initialize(project='earth-engine-project-gan')


def round_coordinates(coordinates, precision=5):
    """Rounds coordinates to the given precision."""
    return [[round(point[0], precision), round(point[1], precision)] for point in coordinates]


def clean_region(region):
    """Cleans up region geometry for better readability."""
    return ee.Geometry.Polygon(round_coordinates(region.getInfo()['coordinates'][0]))


def get_grassy_hill_areas():
    areas = [
        ee.Geometry.Rectangle([5.0, 45.0, 10.0, 47.0]),  # Alps region
        ee.Geometry.Rectangle([-1.5, 42.0, 3.0, 43.5]),  # Pyrenees region
        ee.Geometry.Rectangle([14.0, 49.0, 18.0, 50.5]),  # Czech hills
        ee.Geometry.Rectangle([-111.5, 42.5, -108.5, 45.0]),  # Wyoming
        ee.Geometry.Rectangle([-71.5, -35.0, -69.0, -31.0]),  # Chile
        ee.Geometry.Rectangle([88.0, 46.0, 96.0, 50.0]),  # Mongolia
        ee.Geometry.Rectangle([-64.0, -36.0, -58.0, -30.0]),  # Argentina
        ee.Geometry.Rectangle([-104.0, 35.0, -95.0, 45.0]),  # Great plains
    ]
    return areas


def get_rgb_images(region):
    rgb_images = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    rgb_images = rgb_images.filterBounds(region) \
        .filterDate('2022-01-01', '2022-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))  # Filtering images with low cloud cover
    return rgb_images.median().clip(region)  # Clipping the image to the region


def get_dem_image(region):
    return ee.Image("USGS/SRTMGL1_003").clip(region)


def generate_grid_regions(bounds, box_width, box_height):
    x_min = bounds[0][0]
    y_min = bounds[0][1]
    x_max = bounds[2][0]
    y_max = bounds[2][1]

    regions = []
    x = x_min
    while x + box_width <= x_max:
        y = y_min
        while y + box_height <= y_max:
            region = ee.Geometry.Rectangle([x, y, x + box_width, y + box_height])
            # Clean the region by rounding coordinates
            region = clean_region(region)
            regions.append(region)
            y += box_height
        x += box_width
    return regions


def export_rgb_and_dem(region, num):
    try:
        print(f"Preparing export for region {num}: {region.getInfo()}")

        rgb_image = get_rgb_images(region)
        dem_image = get_dem_image(region)

        # Adjusting scale for smaller file size to avoid tiling
        scale = 20

        task_rgb = ee.batch.Export.image.toDrive(
            image=rgb_image,
            description=f'rgb_image_{num}',
            folder='earth_engine_data',
            scale=scale,
            region=region.getInfo()['coordinates'],
            fileFormat='GeoTIFF',
            maxPixels=1e13  # Allows larger regions to export as a single file
        )
        task_rgb.start()
        print(f"Started RGB export task {num}")

        task_dem = ee.batch.Export.image.toDrive(
            image=dem_image,
            description=f'dem_image_{num}',
            folder='earth_engine_data',
            scale=scale,
            region=region.getInfo()['coordinates'],
            fileFormat='GeoTIFF',
            maxPixels=1e13
        )
        task_dem.start()
        print(f"Started DEM export task {num}")

    except Exception as e:
        print(f"Error exporting region {num}: {e}")
        raise


# Main script logic
try:
    print("Script started.")
    areas = get_grassy_hill_areas()
    box_width = 0.5
    box_height = 0.5

    all_regions = []
    for area in areas:
        bounds = area.bounds().getInfo()['coordinates'][0]
        grid_regions = generate_grid_regions(bounds, box_width, box_height)
        cleaned_bounds = round_coordinates(bounds)
        print(f"Generated {len(grid_regions)} grid regions for area: {cleaned_bounds}")
        bounds = round_coordinates(cleaned_bounds)
        print(f"Generated {len(grid_regions)} grid regions for area: {bounds}")
        all_regions.extend(grid_regions)

    random.shuffle(all_regions)
    num_regions = 1000
    selected_regions = all_regions[:num_regions]

    print(f"Total selected regions: {len(selected_regions)}")

    for num, region in enumerate(selected_regions):
        print(f"Exporting region {num}/{len(selected_regions)}")
        export_rgb_and_dem(region, num)

    print("Script completed. Check Earth Engine Task Manager for task progress.")

except Exception as e:
    print(f"Script stopped due to an error: {e}")
