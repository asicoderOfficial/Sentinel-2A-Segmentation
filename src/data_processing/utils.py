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
