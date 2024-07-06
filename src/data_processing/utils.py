import os

import numpy as np


def pytorch_convention_array(arr:np.array) -> np.array:
    """ Transpose the array to the PyTorch convention of channels first.
    From (width, height, channels) to (channels, height, width).

    Args:
        arr (np.array): The array to be transposed.

    Returns:
        np.array: The transposed array following the PyTorch convention for Computer Vision.
    """    
    if arr.ndim == 3:
        # For cities data
        return np.transpose(arr, (2, 0, 1))
    elif arr.ndim == 2:
        # For buildings data (labels)
        return np.transpose(arr, (0, 1))


def binarize(arr:np.array) -> np.array:
    """ Binarize the array, by setting all values above 0 to 1.
    Used for buildings array, which represents the labels.

    Args:
        arr (np.array): The array to be binarized.

    Returns:
        np.array: The binarized array.
    """    
    arr[arr > 0] = 1
    return arr


def merge_patches(cities:list, patches_save_dir:str) -> None:
    all_cities_patches = {}
    all_buildings_patches = {}
    for city in cities:
        city_name = city['name']
        data_set = city['set']
        patches_save_dir_set = f'{patches_save_dir}/{data_set}'
        npy_files = [f for f in os.listdir(f'{patches_save_dir_set}/{city_name}')]
        city_files = [f for f in npy_files if 'city' in f]
        buildings_files = [f for f in npy_files if 'buildings' in f]
        for city_file in city_files:
            patch_size = city_file.split('_')[1].split('.')[0]
            if patch_size not in all_cities_patches:
                all_cities_patches[city_file.split('_')[1].split('.')[0]] = []
            city_patches = np.load(f'{patches_save_dir_set}/{city_name}/{city_file}', mmap_mode='r')
            all_cities_patches[patch_size].append(city_patches)
        for buildings_file in buildings_files:
            patch_size = buildings_file.split('_')[1].split('.')[0]
            if patch_size not in all_buildings_patches:
                all_buildings_patches[buildings_file.split('_')[1].split('.')[0]] = []
            buildings_patches = np.load(f'{patches_save_dir_set}/{city_name}/{buildings_file}', mmap_mode='r')
            all_buildings_patches[patch_size].append(buildings_patches)

    for patch_size, city_patches in all_cities_patches.items():
        np.save(f'{patches_save_dir_set}/city_{patch_size}.npy', np.concatenate(city_patches))
    for patch_size, building_patches in all_buildings_patches.items():
        np.save(f'{patches_save_dir_set}/buildings_{patch_size}.npy', np.concatenate(building_patches))
