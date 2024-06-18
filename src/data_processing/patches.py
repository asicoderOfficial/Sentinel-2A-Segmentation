from typing import Union

import numpy as np

from src.data_processing.constants import PATCHES_DEFAULT_SIZES



def _pad(city:np.array, buildings:np.array, side_size:int, stride:int):
    width = city.shape[0]
    height = city.shape[1]
    
    # Calculate the padding needed to make dimensions multiples of side_size
    width_pad = (side_size - (width % side_size)) % side_size
    height_pad = (side_size - (height % side_size)) % side_size

    # Calculate padding for width and height
    pad_width_before = width_pad // 2
    pad_width_after = width_pad - pad_width_before
    pad_height_before = height_pad // 2
    pad_height_after = height_pad - pad_height_before

    # Pad the city and buildings arrays
    city_padded = np.pad(city, ((pad_width_before, pad_width_after), (pad_height_before, pad_height_after), (0, 0)), mode='constant')
    buildings_padded = np.pad(buildings, ((pad_width_before, pad_width_after), (pad_height_before, pad_height_after)), mode='constant')

    return city_padded, buildings_padded



def store_patches(city:np.array, buildings:np.array, dir:str, side_sizes:list=PATCHES_DEFAULT_SIZES, stride:Union[int, list]=1):
    """ Given a city and its buildings, it extracts patches as square parts, with a specified stride and side size, and stores them in a directory, together with the corresponding labels (building or not building).

    Args:
        city (np.array): Features, representing the city satellite image in RGB (3D array).
        buildings (np.array): Labels, representing the city satellite image with building annotations (2D array).
        dir (str): Directory where the patches will be stored, together with the labels. 
            A subdirectory will be created for each patch size, and each name of file will specify the coordinates of the patch and if it is a feature file or a label file.
        side_sizes (list, optional): How many pixels the squared patches have. Defaults to PATCHES_DEFAULT_SIZES.
        stride (int, optional): How many pixels the sliding window moves at each step. Defaults to 1. At max, it will move the size of the patch.

    Raises:

    Returns:
        None
    """    
    # Check input
    # City and buildings shapes
    if city.shape[:2] != buildings.shape[:2]: raise ValueError("City and buildings must have the same shape.")
    # Side sizes
    if any(s < 1 for s in side_sizes): raise ValueError("Patch sizes must be at least 1.")
    # Stride
    if stride < 1: raise ValueError("Stride must be at least 1.")
    if stride > min(city.shape[:2]): raise ValueError("Stride must be smaller than the smallest dimension of the city.")
    if stride > min(side_sizes): raise ValueError("Stride must be smaller than the smallest patch size.")

    # Add a border to the city and buildings, to make all patches equally big, before selecting them
    # The border won't be removed afterwards, as this way the patches stay of the same size, and it acts as a form of padding already
    # This border will be black for the city and white for the buildings arrays respectively
    for side_size in side_sizes:
        city_padded, buildings_padded = _pad(city, buildings, side_size, stride)
