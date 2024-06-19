import yaml
import pandas as pd
import torch
import torch.nn as nn

from src.ml.utils import model_n_parameters



class Logger:
    @staticmethod
    def store_by_epoch_data(data: list, info_dir: str, file_name: str, prefix: str='avg_') -> pd.DataFrame:
        """ Stores the data in a file.

        Args:
            data (list): Data to store.
            info_dir (str): Directory where to store the data.
            file_name (str): Name of the file where to store the data.
            prefix (str, optional): Prefix of the file name. Defaults to 'avg'.

        Returns:
            pd.DataFrame: Dataframe with the data.
        """        
        if not isinstance(data[0], dict):
            data = [{'epoch': epoch, 'avg': value} for epoch, value in enumerate(data, start=1)]
        logs_df = pd.DataFrame(data)
        if len(data[0]) > 2:
            logs_df = logs_df.reset_index().rename(columns={'index': 'epoch'})
            # add 1 to the epoch, to start at 1 instead of 0
            logs_df['epoch'] = logs_df['epoch'] + 1
        logs_df.to_csv(f'{info_dir}/{prefix}{file_name}.csv', index=False)

        return logs_df
    

    @staticmethod
    def run_yml(run_info_dir: str, run_id:str, train_percentage: float, train_samples: int, validation_samples: int, model: nn.Module, criterion, optimizer: torch, 
                 trainer_hyperparameters: dict, model_hyperparameters: dict, device:str) -> None:
        """ Stores the run information in a yml file."""

        run_info = {
            'run_id': run_id,
            'device': device,
            'train_percentage': train_percentage,
            'train_samples': train_samples,
            'validation_samples': validation_samples,
            'model': {
                'name': model.__class__.__name__,
                'n_parameters': model_n_parameters(model),
                'hyperparameters': model_hyperparameters
            },
            'criterion': criterion.__class__.__name__,
            'optimizer': optimizer.__class__.__name__,
            'trainer_hyperparameters': trainer_hyperparameters
        }

        with open(f'{run_info_dir}/run.yml', 'w') as outfile:
            yaml.dump(run_info, outfile, default_flow_style=False, sort_keys=False)


    @staticmethod
    def error(run_info_dir: str, error: str) -> None:
        """ Stores the error in a file.

        Args:
            run_info_dir (str): Directory where to store the error.
            error (str): Error to store.

        Returns:
            None
        """        
        with open(f'{run_info_dir}/error.txt', 'w') as outfile:
            outfile.write(error)
