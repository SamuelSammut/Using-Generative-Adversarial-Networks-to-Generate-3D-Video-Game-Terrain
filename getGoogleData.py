import random
import ee

# Authenticating and initializing Earth Engine
ee.Initialize(project='earth-engine-project-gan')


def get_grassy_hill_areas():
    areas = [
        ee.Geometry.Rectangle([5.0, 45.0, 10.0, 47.0]),  # Alps region
        ee.Geometry.Rectangle([-1.5, 42.0, 3.0, 43.5]),  # Pyrenees region
        ee.Geometry.Rectangle([14.0, 49.0, 18.0, 50.5]),  # Czech hills
    ]
    return areas


def get_rgb_images(region):
    rgb_images = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    rgb_images = rgb_images.filterBounds(region) \
        .filterDate('2022-01-01', '2022-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))  # Filtering images with low cloud cover
    return rgb_images.median().clip(region)  # Clipping the image to the region

# Fetching DEM image from the SRTM dataset
def get_dem_image(region):
    return ee.Image("USGS/SRTMGL1_003").clip(region)


def generate_grid_regions(bounds, box_width, box_height):
    # Extract bounding box values from the bounds (list of coordinates)
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
            regions.append(region)
            y += box_height
        x += box_width
    return regions


def export_rgb_and_dem(region, num):
    rgb_image = get_rgb_images(region)
    dem_image = get_dem_image(region)

    print(f"Region for clipping RGB and DEM data: {region.getInfo()}")

    task_rgb = ee.batch.Export.image.toDrive(
        image=rgb_image,
        description=f'rgb_image_{num}',
        folder='earth_engine_data',
        scale=10,  # meters per pixel
        region=region.getInfo()['coordinates'],
        fileFormat='GeoTIFF'
    )

    task_dem = ee.batch.Export.image.toDrive(
        image=dem_image,
        description=f'dem_image_{num}',
        folder='earth_engine_data',
        scale=10,  # meters per pixel
        region=region.getInfo()['coordinates'],
        fileFormat='GeoTIFF'
    )

    task_rgb.start()
    task_dem.start()
    print(f"Exporting task for RGB image {num} and DEM started.")


# Main script logic
areas = get_grassy_hill_areas()
box_width = 0.5  # Increased to reduce duplicates
box_height = 0.5

all_regions = []
for area in areas:
    bounds = area.bounds().getInfo()['coordinates'][0]  # Get the bounding box
    all_regions.extend(generate_grid_regions(bounds, box_width, box_height))

# Shuffle regions to randomize export
random.shuffle(all_regions)
num_regions = 1000
selected_regions = all_regions[:num_regions]

for num, region in enumerate(selected_regions):
    export_rgb_and_dem(region, num)

print("Export tasks started. Check Google Earth Engine Task Manager for progress.")
