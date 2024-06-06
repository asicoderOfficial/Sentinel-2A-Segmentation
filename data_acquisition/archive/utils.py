from __future__ import annotations
import rasterio
from pyproj import CRS
import rasterio.warp
import matplotlib.pyplot as plt
import numpy as np
from rasterio.plot import show
import geopandas as gpd

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# equal_earth_crs = CRS.from_proj4("+proj=eqearth +wktext")

                

# def reproject_to_equal_earth(input_path, output_path):
#     with rasterio.open(input_path) as src:
#         transform, width, height = rasterio.warp.calculate_default_transform(
#             src.crs, equal_earth_crs, src.width, src.height, *src.bounds)
#         kwargs = src.meta.copy()
#         kwargs.update({
#             'crs': equal_earth_crs,
#             'transform': transform,
#             'width': width,
#             'height': height
#         })
#         with rasterio.open(output_path, 'w', **kwargs) as dst:
#             for i in range(1, src.count + 1):
#                 rasterio.warp.reproject(
#                     source=rasterio.band(src, i),
#                     destination=rasterio.band(dst, i),
#                     src_transform=src.transform,
#                     src_crs=src.crs,
#                     dst_transform=transform,
#                     dst_crs=equal_earth_crs,
#                     resampling=rasterio.warp.Resampling.nearest)
            

# input_path = "data_acquisition/satellite_images/openEO_2024-02-07Z.tif"
# output_path = "data_acquisition/equal_earth/openEO_2024-02-07Z_image_equal_earth.tif"
# reproject_to_equal_earth(input_path, output_path)


# # Plotting functions
# def plot_single_band(image_path, band_index):
#     with rasterio.open(image_path) as src:
#         band = src.read(band_index)
#         plt.figure(figsize=(10, 10))
#         plt.title(f"Band {band_index}")
#         plt.imshow(band, cmap='gray')
#         plt.colorbar()
#         plt.show()
                
# def plot_rgb(image_path):
#     with rasterio.open(image_path) as src:
#         red = src.read(1)
#         green = src.read(2)
#         blue = src.read(3)
#         rgb = np.dstack((red, green, blue))
#         plt.figure(figsize=(10, 10))
#         plt.title("RGB Image")
#         plt.imshow(rgb)
#         plt.show()

# def plot_irb(image_path):
#     with rasterio.open(image_path) as src:
#         infrared = src.read(4)
#         red = src.read(1)
#         blue = src.read(3)
#         irb = np.dstack((infrared, red, blue))
#         plt.figure(figsize=(10, 10))
#         plt.title("IRB Image")
#         plt.imshow(irb)
#         plt.show()

# def plot_buildings(image_path, buildings_path):
#     with rasterio.open(image_path) as src:
#         image = src.read([1, 2, 3])
#         transform = src.transform
    
#     buildings = gpd.read_file(buildings_path).to_crs(equal_earth_crs)
    
#     fig, ax = plt.subplots(figsize=(10, 10))
#     show(image, transform=transform, ax=ax)
#     buildings.plot(ax=ax, facecolor='none', edgecolor='red')
#     plt.title("Buildings Overlaid on RGB Image")
#     plt.show()


# plot_single_band(output_path, 1)
# plot_rgb(output_path)
# plot_irb(output_path)





def plot_image(
    image: np.ndarray, factor: float = 1.0, clip_range: tuple[float, float] | None = None, **kwargs: Any
) -> None:
    """Utility function for plotting RGB images."""
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])