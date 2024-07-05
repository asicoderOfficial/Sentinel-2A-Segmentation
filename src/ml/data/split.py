import numpy as np
import torch
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split

from src.ml.data.dataset import SentinelDataset
from src.ml.constants import RANDOM_SEED


def random_split_by_percentage(train_dataset:SentinelDataset, train_percentage:float) -> tuple:
    """ Called when splitting the dataset into train and validation sets randomly by a given percentage.

    Args:
        train_dataset (SentinelDataset): The training dataset.
        train_percentage (float): How much to allocate to train, in the inverval [0, 1].

    Returns:
        tuple: The training and validation datasets.
    """    
    train_size = int(train_percentage * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_SEED))

    return train_dataset, val_dataset


def density_split(dataset: SentinelDataset, train_percentage: float) -> tuple:
    """ Called when splitting by 'density', i.e. stratified split based on the percentage of buildings in each patch 
        (train and validation sets have similar distributions of building percentages).

    Args:
        dataset (SentinelDataset): The dataset to split.
        train_percentage (float): How much to allocate to train, in the inverval [0, 1].

    Returns:
        tuple: The training and validation datasets.
    """    
    # Calculate the percentage of buildings in each patch
    building_percentages = np.mean(dataset.labels, axis=(1, 2))

    # Bin the building percentages to create classes for stratification
    num_bins = 10  # Adjust the number of bins as needed
    bins = np.linspace(0, 1, num_bins + 1)
    binned_percentages = np.digitize(building_percentages, bins) - 1

    # Perform stratified split based on binned percentages
    train_indices, val_indices = train_test_split(
        np.arange(len(building_percentages)),
        train_size=train_percentage,
        stratify=binned_percentages
    )

    # Create the feature and label subsets for train and validation sets
    train_features = dataset.features[train_indices]
    train_labels = dataset.labels[train_indices]
    val_features = dataset.features[val_indices]
    val_labels = dataset.labels[val_indices]

    # Create new SentinelDataset instances for the train and validation sets
    train_dataset = SentinelDataset(features=train_features, labels=train_labels)
    val_dataset = SentinelDataset(features=val_features, labels=val_labels)

    return train_dataset, val_dataset
