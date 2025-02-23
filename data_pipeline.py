import os
import logging
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
import numpy as np
from sentinelhub import (
    CRS,
    BBox,
    bbox_to_dimensions,
    SHConfig,
)

from src.data_acquisition.sentinel_hub_image_downloader import SentinelHubImageDownloader
from src.data_acquisition.osm import OSM
from src.data_acquisition.constants import EVALSCRIPTS
from src.data_processing.utils import pytorch_convention_array, binarize, merge_patches
from src.data_processing.patches import store_patches


# Load configurations and credentials
main_dir = os.getcwd()
sentinel_config = SHConfig()

yml_file = input("Enter the path to the yml file with the configuration for the data pipeline (defaults to data/config/data_pipeline.yml): ")
if not yml_file:
    yml_file = f'{main_dir}/config/data_pipeline.yml'
while not os.path.exists(yml_file):
    logging.warning(f"Warning! The file {yml_file} does not exist. Please provide a valid path.")
    yml_file = input("Enter the path to the yml file with the configuration for the data pipeline (defaults to data/config/data_pipeline.yml): ")
    if not yml_file:
        yml_file = f'{main_dir}/config/data_pipeline.yml'

with open(yml_file, 'r') as file:
    data_pipeline_config = yaml.load(file, Loader=yaml.FullLoader)

load_dotenv(f'{main_dir}/{data_pipeline_config["env_file"]}')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


sentinel_config.sh_client_id = os.getenv("SH_CLIENT_ID")
sentinel_config.sh_client_secret = os.getenv("SH_CLIENT_SECRET")

if not sentinel_config.sh_client_id or not sentinel_config.sh_client_secret:
    logging.warning(f"Warning! To use Process API, please provide the credentials (OAuth client ID and client secret). They must be located in the {main_dir}/config/.env.local file, and be named SH_CLIENT_ID and SH_CLIENT_SECRET.")
    exit()

# Set parameters from the yml file
merge = data_pipeline_config['merge']
download = data_pipeline_config['download']
cities = data_pipeline_config['cities']
patches_save_dir = data_pipeline_config['patches_save_dir']
if download:
    resolution = data_pipeline_config['sentinel']['resolution']
    max_cloud_coverage = data_pipeline_config['sentinel']['max_cloud_coverage']
    evalscript = EVALSCRIPTS[data_pipeline_config['sentinel']['evalscript']]
    patches = data_pipeline_config['patches']
    verbose = data_pipeline_config['verbose']

    directories = [patches_save_dir, 
                    f'{patches_save_dir}/original_images', 
                    f'{patches_save_dir}/train', 
                    f'{patches_save_dir}/test']

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    # Get the corresponding coordinates bounding boxes for each of the specified cities
    osm = OSM(imgs_tmp_folder='tmp', imgs_tmp_extension='png')
    for city in cities:
        city_name = city['name']
        # Bounding boxes
        north = city['north']
        south = city['south']
        east = city['east']
        west = city['west']
        # Train or test data
        data_set = city['set']
        if verbose:
            logging.info(f"Downloading data for city {city['name']} with coordinates, north: {city['north']}, south: {city['south']}, east: {city['east']}, west: {city['west']}")
        osmnx_bbox = (north, south, east, west)
        sentinel_bbox = BBox(bbox=(west, south, east, north), crs=CRS.WGS84)

        size = bbox_to_dimensions(sentinel_bbox, resolution=resolution)
        if size[0] > 2500 or size[1] > 2500:
            logging.warning(f"Warning! The size of the image is too big ({size[0]}x{size[1]} pixels) for city {city}. The maximum size allowed is 2500x2500 pixels for SentinelHub API in free version.")
            exit()
        # Date range
        end_date = datetime.now()
        start_date = end_date.replace(month=end_date.month-3)
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")

        # Download the Sentinel-2A image data RGB + CLM
        downloader = SentinelHubImageDownloader(sentinel_bbox, size, sentinel_config, max_cloud_coverage=max_cloud_coverage)
        downloader.time_interval = (start_date, end_date)
        downloader.evalscript = evalscript
        city_data = downloader.download_image()
        city_data = city_data[0]
        if verbose:
            logging.info(f"Downloaded the city data for {city_name} from SentinelHub with shape {city_data.shape}")

        # Download the buildings data from OpenStreetMaps (target labels)
        buildings_data = osm.buildings(osmnx_bbox, dimensions=size)
        if verbose:
            logging.info(f"Downloaded the buildings data for {city_name} from OpenStreetMaps with shape {buildings_data.shape}")

        # Reshape them to have PyTorch like convention (C, H, W), instead of (W, H, C)
        city_data = pytorch_convention_array(city_data)
        buildings_data = pytorch_convention_array(buildings_data)
        buildings_data = binarize(buildings_data)

        # Save the original images, so that if later different patches are needed, they can be extracted from the original images, avoiding a new download
        city_path = f'{patches_save_dir}/original_images/{city_name}_city.npy'
        buildings_path = f'{patches_save_dir}/original_images/{city_name}_buildings.npy'
        np.save(city_path, city_data)
        np.save(buildings_path, buildings_data)
        if verbose:
            logging.info(f"Saved the original images for {city_name} in {city_path} and {buildings_path}")

        # Extract patches from the images and save them
        if verbose:
            logging.info(f"Extracting patches from the images for {city_name}")
        patches_save_dir_set = f'{patches_save_dir}/{data_set}'
        for patch in patches:
            patch_side_size = patch['size']
            patch_stride = patch['stride']
            patch_augmentations = patch['augmentations']
            store_patches(city_data, buildings_data, patches_save_dir_set, city_name, patch_side_size, stride=patch_stride, augmentations=patch_augmentations)
        if verbose:
            logging.info(f"Finished extracting patches from the images for {city_name}. Find them in {patches_save_dir_set}/, each city has a subdirectory, and all patches are merged together in a file for each patch size, called buildings.npy or city.npy + _patchsize.")

    if os.path.exists('tmp'):
        os.rmdir('tmp')

if merge:
    # Merge all downloaded patches together into the same npy files. One for the cities and one for the buildings, for each patch side size
    train_cities = [city for city in cities if city['set'] == 'train']
    test_cities = [city for city in cities if city['set'] == 'test']
    
    merge_patches(train_cities, patches_save_dir)
    merge_patches(test_cities, patches_save_dir)
