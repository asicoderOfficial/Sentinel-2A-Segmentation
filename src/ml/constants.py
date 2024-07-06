import torch

from src.ml.models.baseline import Baseline
from src.ml.models.unet import UNet
from src.ml.loss_functions.dice import DiceLoss
from src.ml.loss_functions.hausdorff import HausdorffLoss

# Training
EXPERIMENTS_SAVE_DIR = 'runs'
RANDOM_SEED = 42
DEFAULT_CONFIG_PATH = 'config/experiments.yml'

# Data
NPY_DATA_DIR = 'data/'
N_CLASSES = 4
LABELS = [0, 1]

# Parsing yml configuration
# Models
MODELS_DECODER = {
    'baseline': Baseline,
    'unet': UNet
}

# Loss functions
LOSS_FN_DECODER = {
    'bcewithlogits': torch.nn.BCEWithLogitsLoss,
    'dice': DiceLoss,
    'haussdorf': HausdorffLoss
}

# Optimizers
OPTIMIZER_DECODER = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}

