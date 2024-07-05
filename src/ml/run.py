import os
from logging import Logger

import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader

from src.ml.data.dataset import SentinelDataset
from src.ml.data.split import random_split_by_percentage, density_split
from src.ml.constants import EXPERIMENTS_SAVE_DIR
from src.ml.trainer import Trainer
from src.ml.logs.logger import Logger
from src.ml.logs.plots import TrainingPlots



class Run:
    def __init__(self, id: str, train_dataset: SentinelDataset, model: nn.Module, criterion, optimizer: torch, 
                 train_percentage: float=0.7, trainer_hyperparameters: dict={'epochs': 20}, model_hyperparameters: dict={'batch_size': 16, 'learning_rate': 0.001},
                 device:torch.device=torch.device('cuda'), test_dataset: SentinelDataset=None, split_mode:str='density', verbose: bool=False, logger:Logger=None) -> None:
        """ To run an experiment.

        Args:
            id (str): The experiment id.
            train_dataset (SentinelDataset): The training dataset.
            model (nn.Module): The model to train.
            criterion (nn): The loss function.
            optimizer (torch): The optimizer.
            train_percentage (float, optional): How much to allocate to train, in the inverval [0, 1]. Defaults to 0.7.
            trainer_hyperparameters (dict, optional): The trainer hyperparameters to pass to the train() function of Trainer class. Defaults to {'epochs': 20}.
            model_hyperparameters (dict, optional): The model hyperparameters. Defaults to {'batch_size': 16, 'learning_rate': 0.001}.
            device (torch.device, optional): The device to use. Defaults to torch.device('cuda').
            test_dataset (SentinelDataset, optional): The testing dataset (for inference, no labels). Defaults to None.
            split_mode (str, optional): How to split the data. Defaults to 'density'. Modes: 'density' or 'random'.
            verbose (bool, optional): Whether to print information of the run. Defaults to False.
            logger (Logger, optional): The logger to use. Defaults to None.

        Returns:
            None
        """        
        self.id = id
        self.train_dataset = train_dataset
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_percentage = train_percentage
        self.trainer_hyperparameters = trainer_hyperparameters
        self.model_hyperparameters = model_hyperparameters
        self.device = device
        self.test_dataset = test_dataset
        self.verbose = verbose
        self.logger = logger

        self.run_info_dir = f'{EXPERIMENTS_SAVE_DIR}/{self.id}'

        self._dataloaders(split_mode)
        self._directories()

        if self.device.type != 'cuda' and verbose:
            print('WARNING: You are not using CUDA, are you sure you want to continue? Training may be very slow.')

        if self.verbose:
            print('------------------------')
            print(f'Experiment id: {self.id}')
            print(f'Run information directory: {self.run_info_dir}')
            print(f'Training information directory: {self.run_info_dir}/train')
            print(f'Testing information directory: {self.run_info_dir}/test')
            print('------------------------')
            print()

            self.logger.info(
                f'''
                ------------------------
                Experiment id: {self.id}
                Run information directory: {self.run_info_dir}
                Training information directory: {self.run_info_dir}/train
                Testing information directory: {self.run_info_dir}/test
                ------------------------
                '''
                )


    def run(self) -> None:
        """ Runs the experiment.

        Returns:
            None
        """        
        if self.verbose:
            self.logger.info(
                f'''
                Starting experiment training...
                Using {self.device} device
                Model: {self.model.__class__.__name__}, criterion: {self.criterion.__class__.__name__}, optimizer: {self.optimizer.__class__.__name__}
                Number of training samples: {len(self.train_dataset)}, number of validation samples: {len(self.val_loader.dataset)}
                Training hyperparameters: {self.trainer_hyperparameters}
                Model hyperparameters: {self.model_hyperparameters}
                '''
            )
        # Train
        self.trainer = Trainer(self.model, self.criterion, self.optimizer, self.train_loader, self.val_loader, 
                               self.run_info_dir, test_dataloader=self.test_loader, device=self.device, logger=self.logger)
        epoch_losses, train_dice_metrics, val_dice_metrics, train_time_by_epoch = self.trainer.train(**self.trainer_hyperparameters)

        if self.verbose:
            self.logger.info(
                f'''
                Finished training.
                '''
            )

        # Test
        # TODO: Implement testing (if needed)

        if self.verbose:
            self.logger.info(
                f'''
                Starting to store training information...
                '''
            )
        # YML
        Logger.run_yml(self.run_info_dir, self.id, self.train_percentage, len(self.train_dataset), len(self.val_loader.dataset), self.model, self.criterion, self.optimizer, 
                       self.trainer_hyperparameters, self.model_hyperparameters, self.device)
        # Log
        epoch_loss_df = Logger.store_by_epoch_data(epoch_losses, info_dir=f'{self.run_info_dir}/train/logs', file_name='loss')
        train_epoch_dice_df = Logger.store_by_epoch_data(train_dice_metrics, info_dir=f'{self.run_info_dir}/train/logs', file_name='dice_train')
        val_epoch_dice_df = Logger.store_by_epoch_data(val_dice_metrics, info_dir=f'{self.run_info_dir}/train/logs', file_name='dice_val')
        train_time_by_epoch_df = Logger.store_by_epoch_data(train_time_by_epoch, info_dir=f'{self.run_info_dir}/train/logs', file_name='time_by_epoch', prefix='')

        # Plots
        TrainingPlots.loss_by_epoch(epoch_loss_df, f'{self.run_info_dir}/train/plots')
        TrainingPlots.dice_by_epoch(train_epoch_dice_df, val_epoch_dice_df, f'{self.run_info_dir}/train/plots')
        TrainingPlots.time_by_epoch(train_time_by_epoch_df, f'{self.run_info_dir}/train/plots')

        if self.verbose:
            self.logger.info(
                f'''
                Experiment {self.id} finished.
                Check the logs, plots, checkpoints & yml configuration file at {self.run_info_dir}
                '''
            )


    def _dataloaders(self, split_mode:str='density') -> None:
        """ Creates the dataloaders.

        Args:
            split_mode (str, optional): The mode to split the data. Defaults to 'density'. Modes:
                - density: splits the data by the density of the buildings, having a balanced distribution of buildings in the train and validation sets.
                - random: splits the data randomly, without any consideration of the distribution of the buildings, just the percentage of the train set.

        Returns:
            None
        """        
        if split_mode == 'density':
            train_dataset, val_dataset = density_split(self.train_dataset, self.train_percentage)            
        else:
            train_dataset, val_dataset = random_split_by_percentage(self.train_dataset, self.train_percentage)

        self.train_loader = DataLoader(train_dataset, batch_size=self.model_hyperparameters['batch_size'], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.model_hyperparameters['batch_size'], shuffle=False)

        if self.test_dataset:
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.model_hyperparameters['batch_size'], shuffle=False)
        else:
            self.test_loader = None


    def _directories(self) -> None:
        """ Creates the directories where to store the training & testing information. """
        os.makedirs(self.run_info_dir, exist_ok=True)
        # Training information
        os.makedirs(f'{self.run_info_dir}/train/checkpoints', exist_ok=True)
        os.makedirs(f'{self.run_info_dir}/train/logs', exist_ok=True)
        os.makedirs(f'{self.run_info_dir}/train/plots', exist_ok=True)

        # Testing information
        if self.test_dataset:
            os.makedirs(f'{self.run_info_dir}/test/checkpoints', exist_ok=True)
            os.makedirs(f'{self.run_info_dir}/test/logs', exist_ok=True)
            os.makedirs(f'{self.run_info_dir}/test/plots', exist_ok=True)
