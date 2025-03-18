// Load Sentinel-2 dataset (RGB)
var s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    .filterDate('2023-05-20', '2023-08-30')  // Spring/early summer
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10));

var region = ee.Geometry.Polygon([
    [[5.0, 45.0], [10.0, 45.0], [10.0, 47.0], [5.0, 47.0], [5.0, 45.0]]
]);  // Alps example

var s2_median = s2.median().clip(region);

var dem = ee.Image("USGS/SRTMGL1_003").clip(region);

// Display in Google Earth Engine UI
Map.centerObject(region, 7);
Map.addLayer(s2_median, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 'RGB');
Map.addLayer(dem, {min: 0, max: 3000, palette: ['white', 'black']}, 'DEM');

// Export RGB Image to Google Drive
Export.image.toDrive({
    image: s2_median,
    description: 'big_rgb',
    scale: 20,
    region: region,
    fileFormat: 'GeoTIFF',
    maxPixels: 1e13
});

// Export DEM Image to Google Drive
Export.image.toDrive({
    image: dem,
    description: 'big_dem',
    scale: 20,
    region: region,
    fileFormat: 'GeoTIFF',
    maxPixels: 1e13
});
