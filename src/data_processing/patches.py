import os

import numpy as np
import torch

from src.data_processing.constants import AUGMENTATIONS_DECODER


def _pad(city: np.array, buildings: np.array, side_size: int, stride: int):
    # Extract dimensions
    _, height_city, width_city = city.shape
    height_buildings, width_buildings = buildings.shape

    # Calculate padding needed to make dimensions multiples of side_size and account for stride
    def calculate_padding(dim_size, side_size, stride):
        pad_needed = (side_size - (dim_size % side_size)) % side_size
        if (dim_size + pad_needed) % stride != 0:
            pad_needed += stride - ((dim_size + pad_needed) % stride)
        pad_before = pad_needed // 2
        pad_after = pad_needed - pad_before
        return pad_before, pad_after

    # Calculate padding for width and height for city
    pad_height_before_city, pad_height_after_city = calculate_padding(height_city, side_size, stride)
    pad_width_before_city, pad_width_after_city = calculate_padding(width_city, side_size, stride)

    # Calculate padding for width and height for buildings
    pad_height_before_buildings, pad_height_after_buildings = calculate_padding(height_buildings, side_size, stride)
    pad_width_before_buildings, pad_width_after_buildings = calculate_padding(width_buildings, side_size, stride)

    # Create the padding tuple for the city array
    padding_city = [(0, 0), (pad_height_before_city, pad_height_after_city), (pad_width_before_city, pad_width_after_city)]
    
    # Create the padding tuple for the buildings array
    padding_buildings = [(pad_height_before_buildings, pad_height_after_buildings), (pad_width_before_buildings, pad_width_after_buildings)]

    # Pad the city and buildings arrays
    city_padded = np.pad(city, padding_city, mode='constant')
    buildings_padded = np.pad(buildings, padding_buildings, mode='constant')

    return city_padded, buildings_padded


def store_patches(city:np.array, buildings:np.array, dir:str, city_id:str, side_size:int, stride:int=1, augmentations:list=[]):
    """ Given a city and its buildings, it extracts patches as square parts, with a specified stride and side size, and stores them in a directory, together with the corresponding labels (building or not building).

    Important: it assumes that city is 3D (ndim = 3) and buildings is 2D (ndim = 2) and they have a [C, H, W] and [H, W] shape respectively, 
    where C is the number of channels, H is the height and W is the width (compatible with PyTorch convention).

    Args:
        city (np.array): Features, representing the city satellite image in RGB or multispectral format (both 3D array).
        buildings (np.array): Labels, representing the city satellite image with building annotations (2D array, with only 0s and 1s).
        dir (str): Directory where the patches will be stored, together with the labels. 
            A subdirectory will be created for each patch size, and each name of file will specify the coordinates of the patch and if it is a feature file or a label file.
        city_id (str): The id of the city, to be used in the name of the files.
        side_size (int, optional): How many pixels the squared patch will have.
        stride (int, optional): How many pixels the sliding window moves at each step. Defaults to 1. At max, it will move the size of the patch.

    Raises:

    Returns:
        None
    """    
    # Check input
    # City and buildings shapes
    if city.shape[1:] != buildings.shape: raise ValueError("City and buildings (features and labels, respectively) must have the same shape.")
    # Side sizes
    if side_size < 1: raise ValueError("Patch sizes must be at least 1 pixel by 1 pixel.")
    # Stride
    if stride < 1: raise ValueError("Stride must be at least 1.")
    if stride > min(city.shape[:2]): raise ValueError("Stride must be smaller than the smallest dimension of the city.")
    if stride > side_size: raise ValueError("Stride must be smaller than the patch size.")

    city_patches = []
    buildings_patches = []
    if not os.path.exists(f'{dir}/{city_id}'): os.makedirs(f'{dir}/{city_id}')
    # Add a border to the city and buildings, to make all patches equally big, before selecting them
    # The border won't be removed afterwards, as this way the patches stay of the same size, and it acts as a form of padding already
    # This border will be black for the city and white for the buildings arrays respectively
    city_padded, buildings_padded = _pad(city, buildings, side_size, stride)
    # Iterate over the city and buildings arrays, selecting patches
    # convert city_padded to torch tensor
    city_padded = torch.from_numpy(city_padded)
    buildings_padded = torch.from_numpy(buildings_padded)
    for i in range(0, city_padded.shape[2] - side_size + 1, stride):
        for j in range(0, city_padded.shape[1] - side_size + 1, stride):
            # Select the patch
            city_patch = city_padded[:, j:j + side_size, i:i + side_size]
            buildings_patch = buildings_padded[j:j + side_size, i:i + side_size]
            # Detect clouds. If any, discard the patch
            # Check if there is any 1 in the 13th channel of the city patch
            if not city_patch[12, :, :].any():
                city_patches.append(city_patch)
                buildings_patches.append(buildings_patch)
                # Apply the transformations to augment the dataset
                for augmentation in augmentations:
                    augmentation_name = augmentation['name']
                    if augmentation_name in AUGMENTATIONS_DECODER:
                        if augmentation_name == 'compose':
                            # Compose multiple augmentations
                            curr_aug = AUGMENTATIONS_DECODER[augmentation_name]([AUGMENTATIONS_DECODER[aug['name']](**aug['parameters']) for aug in augmentation['augmentations']])
                        else:
                            curr_aug = AUGMENTATIONS_DECODER[augmentation_name](**augmentation['parameters'])
                        # Apply it
                        augmented_city_patch = curr_aug(city_patch)
                        city_patches.append(augmented_city_patch)
                        buildings_patches.append(buildings_patch)
                    else:
                        raise ValueError(f"Augmentation {augmentation['name']} not found in the decoder dictionary. Check the src.data_processing.constants.AUGMENTATIONS_DECODER dictionary for the available augmentations.")
    
    # Save all patches together
    city_patches = np.array(city_patches)
    buildings_patches = np.array(buildings_patches)
    np.save(f'{dir}/{city_id}/city.npy', city_patches)
    np.save(f'{dir}/{city_id}/buildings.npy', buildings_patches)
