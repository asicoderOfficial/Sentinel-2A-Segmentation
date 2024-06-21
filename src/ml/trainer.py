import time
from logging import Logger

import torch.nn as nn
import torch
from torch.utils.data.dataloader import DataLoader


from src.ml.metrics.dice import DiceCoefficient
from src.ml.constants import LABELS


class Trainer:
    def __init__(self, model: nn.Module, criterion: nn, optimizer: torch.optim, train_dataloader: DataLoader, val_dataloader: DataLoader, 
                 info_dir: str, test_dataloader: DataLoader=None, device: str='cuda', logger:Logger=None) -> None:
        """ Trainer class for training and evaluating a model.

        Args:
            model (nn.Module): Model to train.
            criterion (nn.Module): Loss function.
            optimizer (nn.Module): Optimizer.
            train_dataloader (DataLoader): Training dataloader.
            val_dataloader (DataLoader): Validation dataloader.
            info_dir (str): Directory where to store the training information.
            test_dataloader (DataLoader, optional): Test dataloader. Defaults to None.
            device (str, optional): Device to use. Defaults to 'cuda'.
            logger (Logger, optional): Logger to use. Defaults to None.

        Returns:
            None
        """        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.info_dir = info_dir
        self.device = device
        self.logger = logger

        self.model.to(self.device)
        self.criterion.to(self.device)

    
    def train(self, epochs: int, early_stopping:dict={}) -> tuple:
        """ Train the model.

        Args:
            epochs (int): Number of epochs.
            early_stopping (dict, optional): 
                {"delta": When to consider that the loss is converging.
                "patience": Number of epochs to wait before stopping the training, if the loss continues to converge.
                }. Defaults to {}.

        Returns:
            tuple: Losses, metrics & training time by epoch.
        """        
        # Losses & metrics by epoch
        epoch_losses = []
        train_dice_metrics = []
        val_dice_metrics = []
        train_time_by_epoch = []

        # Early stopping
        if early_stopping:
            delta = early_stopping['delta']
            patience = early_stopping['patience']
            prev_loss = 0

        for epoch in range(1, epochs + 1):
            if early_stopping and patience == 0:
                if self.logger:
                    print(f'Early stopping at epoch {epoch}.')
                break
            epoch_start_time = time.time()
            self.model.train() # Set model to train mode, to enable dropout, batch normalization, etc if used
            epoch_loss = 0
            epoch_total_dice = 0
            for batch_idx, (samples, labels) in enumerate(self.train_dataloader):
                # Move data to device, to avoid the overhead of moving it every time inside the loop from CPU to GPU
                samples = samples.to(self.device)
                labels = labels.to(self.device)
                # Reset gradients. 
                # Otherwise, the gradient would be a combination of the old gradient, which has already been used to update the model parameters and the newly-computed gradient. 
                # It would therefore point in some other direction than the intended direction towards the minimum (or maximum, in case of maximization objectives).
                # However, when training some networks such as RNNs, this is behavior is desired (but it is not the case here).
                self.optimizer.zero_grad() 

                # Forward pass
                outputs = self.model(samples)
                outputs = outputs.squeeze(1) # Special case: as in the task it was requested that the labels have no channels, the output should not have them either
                # Compute loss
                loss = self.criterion(outputs, labels)
                # Backward pass
                loss.backward()
                # Update model parameters
                self.optimizer.step()

                # Metrics & loss storage for the current batch
                epoch_loss += loss.item()
                dice = DiceCoefficient.compute_dice(predicted=outputs, target=labels)
                epoch_total_dice += dice
            
                if batch_idx % 10 == 0:
                    if self.logger:
                        self.logger.info(f'Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item()} | DICE: {dice}')
            
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            train_time_by_epoch.append(epoch_time)

            # Average training metrics & loss storage by epoch
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            avg_epoch_dice = epoch_total_dice / len(self.train_dataloader)
            epoch_losses.append(avg_epoch_loss)
            train_dice_metrics.append(avg_epoch_dice)
            
            self.logger.info(f'Epoch: {epoch} | Average loss: {avg_epoch_loss} | Average DICE: {avg_epoch_dice}')

            # Validation
            validation_dice = self.validate()
            val_dice_metrics.append(validation_dice)

            # Early stopping
            if early_stopping:
                if prev_loss - avg_epoch_loss < delta and prev_loss - avg_epoch_loss > 0:
                    patience -= 1
                else:
                    patience = early_stopping['patience']

            # If it is the last epoch, or one every x, save the model
            if epoch == epochs:
                torch.save(self.model.state_dict(), f'{self.info_dir}/model.pth')
            elif epoch % 5 == 0:
                torch.save(self.model.state_dict(), f'{self.info_dir}/train/checkpoints/epoch_{epoch}.pth')

        return epoch_losses, train_dice_metrics, val_dice_metrics, train_time_by_epoch
    

    def validate(self) -> float:
        all_validation_dice = 0
        self.model.eval()
        with torch.no_grad():
            for samples, labels in self.val_dataloader:
                samples = samples.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(samples)
                dice = DiceCoefficient.compute_dice(predicted=outputs, target=labels)
                all_validation_dice += dice

        avg_validation_dice = all_validation_dice / len(self.val_dataloader)

        return avg_validation_dice
