import matplotlib.pyplot as plt
import rasterio
import rasterio.warp
from pyrosm import OSM, get_data
import geopandas as gpd
from pyproj import CRS
from rasterio.plot import show



equal_earth_crs = CRS.from_proj4("+proj=eqearth +wktext")

def plot_buildings(image_path, buildings_path):
    with rasterio.open(image_path) as src:
        image = src.read([1, 2, 3])
        transform = src.transform
    
    buildings = gpd.read_file(buildings_path).to_crs(equal_earth_crs)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    show(image, transform=transform, ax=ax)
    buildings.plot(ax=ax, facecolor='none', edgecolor='red')
    plt.title("Buildings Overlaid on RGB Image")
    plt.show()

# download OSM data for Berlin
osm = OSM(get_data("Berlin"))

# get buildings
buildings = osm.get_buildings()

buildings_file_path = "buildings.geojson"
buildings.to_file(buildings_file_path, driver='GeoJSON')

# reproject buildings to equal earth CRS
buildings = buildings.to_crs(equal_earth_crs)

output_path = "data_acquisition/equal_earth/openEO_2024-02-07Z_image_equal_earth.tif"

# plot buildings on satellite image
plot_buildings(output_path, buildings_file_path)