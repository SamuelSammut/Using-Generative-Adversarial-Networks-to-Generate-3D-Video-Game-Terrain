import random
import ee

# Authenticating and initializing Earth Engine
ee.Initialize(project='earth-engine-project-gan')

# Identifying mountainous areas using slope
def get_mountainous_areas():
    # SRTM DEM dataset
    dem = ee.Image("USGS/SRTMGL1_003")
    # Calculating slope from DEM
    slope = ee.Terrain.slope(dem)
    # Defining mountainous terrain with slope > 30 degrees
    mountainous_areas = slope.gt(30)
    return mountainous_areas

# Defining a region of interest
region = ee.Geometry.Rectangle([86.5, 27.5, 87.5, 28.5])

# Fetching RGB images from the Sentinel-2 Harmonized dataset
def get_rgb_images(region):
    # Updated Sentinel-2 Harmonized data, non-harmonized variant was deprecated and not working
    rgb_images = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    rgb_images = rgb_images.filterBounds(region) \
                           .filterDate('2022-01-01', '2022-12-31') \
                           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))  # Filtering images with low cloud cover
    return rgb_images.median().clip(region)  # Clipping the image to the region

# Fetching DEM image from the SRTM dataset
def get_dem_image(region):
    # Clipping DEM to the region
    dem_image = ee.Image("USGS/SRTMGL1_003").clip(region)
    return dem_image

def generate_grid_regions(bounds, box_width, box_height):
    x_min, y_min, x_max, y_max = bounds
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


# Exporting both RGB and DEM data to Google Drive
def export_rgb_and_dem(region, num):
    # Fetching RGB and DEM data
    rgb_image = get_rgb_images(region)
    dem_image = get_dem_image(region)

    # Log the geometry for debugging
    print(f"Region for clipping RGB and DEM data: {region.getInfo()}")

    # Exporting RGB to Google Drive
    task_rgb = ee.batch.Export.image.toDrive(
        image=rgb_image,
        description=f'rgb_image_{num}',
        folder='earth_engine_data',
        scale=50,  # meters per pixel
        region=region.getInfo()['coordinates'],
        fileFormat='GeoTIFF'
    )

    # Exporting DEM to Google Drive
    task_dem = ee.batch.Export.image.toDrive(
        image=dem_image,
        description=f'dem_image_{num}',
        folder='earth_engine_data',
        scale=50,  # meters per pixel
        region=region.getInfo()['coordinates'],
        fileFormat='GeoTIFF'
    )

    # Starting the export tasks
    task_rgb.start()
    task_dem.start()

    print(f"Exporting task for RGB image {num} and DEM started.")


base_bounds = [75.0, 25.0, 100.0, 40.0]
box_width = 0.25  # in degrees
box_height = 0.25

grid_regions = generate_grid_regions(base_bounds, box_width, box_height)

# Shuffle regions and select wanted number
random.shuffle(grid_regions)
num_regions = 1000  # Change
selected_regions = grid_regions[:num_regions]

# Automating the export process
for num, region in enumerate(selected_regions):
    export_rgb_and_dem(region, num)

print("Export tasks started. Check Google Earth Engine Task Manager for progress.")
