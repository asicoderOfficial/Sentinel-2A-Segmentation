import os

import osmnx as ox
import geopandas as gpd
import numpy as np
from PIL import Image



class OSM:
    """ Class to interact with OpenStreetMaps data."""

    def __init__(self, imgs_tmp_folder:str, imgs_tmp_extension:str) -> None:
        self.imgs_tmp_folder = imgs_tmp_folder
        self.imgs_tmp_extension = imgs_tmp_extension
        self.fp = f"./{self.imgs_tmp_folder}/osm_buildings.{self.imgs_tmp_extension}"


    def _create_buildings_raw_img(self, bbox: tuple) -> None:
        """ Private method to obtain the nodes as a geopandas dataframe and plot them as an image in a temporary file.

        Args:
            bbox (tuple): Bounding box of coordinates

        Returns:
            None
        """        
        if not os.path.exists(self.fp):
            osm_feats:gpd.GeoDataFrame = ox.features_from_bbox(bbox=bbox, tags={'building':True})
            proj:gpd.GeoDataFrame = ox.projection.project_gdf(osm_feats, to_latlong=True) # Very important to specify to_latlong=True, if not the image gets slightly rotated
            ox.plot_footprints(proj, filepath=self.fp, dpi=200, save=True, show=False, close=True, color='white')


    def _load_raw_img(self, dimensions:tuple=()) -> Image:
        img = Image.open(self.fp)

        if dimensions:
            img = img.resize(dimensions)
        
        return img


    def buildings(self, bbox: tuple, dimensions:tuple=()) -> np.array:
        """ Get the buildings in a given bounding box of coordinates as a black and white 2D array.

        Args:
            bbox (tuple): Bounding box of coordinates
            dimensions (tuple, optional): The dimensions of the image to be saved. Defaults to () (in which case the image will be saved with the original dimensions). 
                If specified, must be a tuple of two integers (width, height), and the image will be resized to these dimensions before saving and displaying.
                Usually it is needed, as there is a slight difference between SentinelHub and OpenStreetMaps dimensions, but this resizing has little to no impact on the image quality.

        Returns:
            np.array: Array of the buildings in the given bounding box, with 255 for buildings and 0 for others.
        """        
        self._create_buildings_raw_img(bbox=bbox)

        buildings_img = self._load_raw_img(dimensions=dimensions)

        if buildings_img.mode == 'RGBA':
            # Remove the alpha channel (transparency)
            buildings_img = buildings_img.convert('RGB')

        buildings_img = np.array(buildings_img)

        buildings_img[buildings_img < 50] = 0
        buildings_img[buildings_img >= 50] = 255

        # Remove the temporary image, no longer needed
        os.remove(self.fp)

        return buildings_img


    def visualize_buildings(self, bbox: tuple, img_name:str='', dimensions:tuple=()) -> None:
        """ Visualize the buildings in a given bounding box of coordinates in binary (black for buildings, white for others).

        Args:
            bbox (tuple): Bounding box of coordinates.
            image_name (str, optional): The name with which to save the image. Defaults to '' (in which case the image will not be saved).
            dimensions (tuple, optional): The dimensions of the image to be saved. Defaults to () (in which case the image will be saved with the original dimensions). 
                If specified, must be a tuple of two integers (width, height), and the image will be resized to these dimensions before saving and displaying.
                Usually it is needed, as there is a slight difference between SentinelHub and OpenStreetMaps dimensions, but this resizing has little to no impact on the image quality.

        Returns:
            None
        """        
        self._create_buildings_raw_img(bbox=bbox)

        img = self._load_raw_img(dimensions=dimensions)

        img.show()

        os.remove(self.fp)
        if img_name:
            img.save(f'./{self.imgs_tmp_folder}/{img_name}.{self.imgs_tmp_extension}')
