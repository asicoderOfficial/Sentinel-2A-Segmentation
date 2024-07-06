import numpy as np
import torch
from torch.utils.data import Dataset


class SentinelDataset(Dataset):

    def __init__(self, dir:str='', patch_size:int=None, features:np.ndarray=None, labels:np.ndarray=None, precision:np.dtype=np.float32) -> None:
        """ Dataset class for the Sentinel dataset.
        It assumes that, as requested in the task description, features and labels are stored in the same directory, each in a big .npy file.

        Args:
            dir (str): Directory where the features and labels are stored.
            patch_size (int): The size of the patches to load.
            features (np.ndarray, optional): The features data. Defaults to None. Used when splitting the data by density.
            labels (np.ndarray, optional): The labels data. Defaults to None. Used when splitting the data by density.
            precision (np.dtype, optional): Precision of the data. Defaults to np.float32. Allowing for later reduction of memory usage.
        
        Raises:
            ValueError: If neither a directory nor features and labels are provided.

        Returns:
            None
        """        
        if dir and patch_size:
            self.features = np.load(f'{dir}/city_{patch_size}.npy', mmap_mode='r').astype(precision)
            self.labels = np.load(f'{dir}/buildings_{patch_size}.npy', mmap_mode='r').astype(precision)
        elif features is not None and labels is not None:
            self.features = features
            self.labels = labels
        else:
            raise ValueError('You need to provide either a directory or features and labels.')


    def __len__(self) -> int:
        return len(self.features)
    

    def __getitem__(self, index: int) -> tuple:
        return torch.from_numpy(self.features[index]), torch.from_numpy(self.labels[index])
