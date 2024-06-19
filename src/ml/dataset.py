import numpy as np
import torch
from torch.utils.data import Dataset


class SentinelDataset(Dataset):

    def __init__(self, dir, transforms: list=[], precision=np.float32) -> None:
        """ Dataset class for the Sentinel dataset.
        It assumes that, as requested in the task description, features and labels are stored in the same directory, each in a big .npy file.

        Args:
            dir (str): Directory where the features and labels are stored.
            transforms (list, optional): Which transforms to perform. Defaults to [].
            precision (np.dtype, optional): Precision of the data. Defaults to np.float32. Allowing for later reduction of memory usage.
        
        Returns:
            None
        """        
        self.dir = dir
        self.features = np.load(f'{dir}/city.npy', mmap_mode='r').astype(precision)
        self.labels = np.load(f'{dir}/buildings.npy', mmap_mode='r').astype(precision)


    def __len__(self) -> int:
        return len(self.features)
    

    def __getitem__(self, index: int) -> tuple:
        return torch.from_numpy(self.features[index]), torch.from_numpy(self.labels[index])
