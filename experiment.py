import os
import traceback
import logging
from datetime import datetime

import torch
import yaml
from sklearn.utils import check_random_state

from src.ml.dataset import SentinelDataset
from src.ml.constants import RANDOM_SEED, DEFAULT_CONFIG_PATH, MODELS_DECODER, LOSS_FN_DECODER, OPTIMIZER_DECODER
from src.ml.run import Run


# Parse configuration from yml file
config_path = input(f'Please enter the path to the configuration YAML file (default: {DEFAULT_CONFIG_PATH}): ')
if config_path == '':
    config_path = DEFAULT_CONFIG_PATH

with open(config_path, 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

logging_dir = config['logging']['dir']
if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)
train_data_dir = config['train']['data_dir']

# Set a fixed random seed for reproducibility
seed = config['seed'] if 'seed' in config else RANDOM_SEED
torch.manual_seed(seed)
check_random_state(seed)

# Set the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if config['device'] == 'auto' else torch.device(config['device'])
# Model
model_name = config['model']['name'].lower()
model_parameters = config['model']['parameters']
# Loss function
criterion_name = config['loss']['name'].lower()
criterion_parameters = config['loss']['parameters']
# Optimizer
optimizer_name = config['optimizer']['name'].lower()
optimizer_parameters = config['optimizer']['parameters']
# Model training hyperparameters
train_percentages = config['train']['mode']['perc']
batch_sizes = config['train']['hyperparameters']['batch_size']
epochs = config['train']['hyperparameters']['epochs']
learning_rates = config['train']['hyperparameters']['lr']
early_stopping = config['train']['early_stopping']
# Data for train-val and for test
train_val_dataset = SentinelDataset(train_data_dir)
#test_dataset = SentinelDataset(config['test']['data_dir'])

verbose = config['verbose']

for curr_train_percentage in train_percentages:
    for curr_batch_size in batch_sizes:
        for curr_epoch in epochs:
            for curr_learning_rate in learning_rates:
                try:
                    curr_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                    experiment_id = f'{curr_time}_{model_name}_trperc_{curr_train_percentage}_batch_{curr_batch_size}_epoch_{curr_epoch}_lr_{curr_learning_rate}'
                    # Set up logging
                    if not os.path.exists(f"runs/{experiment_id}/{logging_dir}/"):
                        os.makedirs(f"runs/{experiment_id}/{logging_dir}/")
                    log_file = f"runs/{experiment_id}/{logging_dir}/training_logs.log"
                    logging.basicConfig(level=logging.INFO,
                                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                        handlers=[
                                            logging.FileHandler(log_file),
                                            logging.StreamHandler()
                                        ])
                    logger = logging.getLogger(__name__)
                    if verbose:
                        logger.info(f'Running experiment with train percentage {curr_train_percentage}, batch size {curr_batch_size}, epoch {curr_epoch} and learning rate {curr_learning_rate}.')
                    # Model hyperparameters
                    model_hyperparameters = {'batch_size': curr_batch_size, 'learning_rate': curr_learning_rate}
                    # Trainer hyperparameters
                    trainer_hyperparameters = {'epochs': curr_epoch, 'early_stopping': early_stopping}

                    # Model
                    model = MODELS_DECODER[model_name](**model_parameters)

                    # Loss function
                    criterion = LOSS_FN_DECODER[criterion_name](**criterion_parameters)

                    # Optimizer
                    optimizer = OPTIMIZER_DECODER[optimizer_name](model.parameters(), **optimizer_parameters)

                    # Execute the experiment! Cross your fingers!
                    r = Run(id=experiment_id, train_dataset=train_val_dataset, model=model, criterion=criterion, optimizer=optimizer,
                            train_percentage=curr_train_percentage, trainer_hyperparameters=trainer_hyperparameters, model_hyperparameters=model_hyperparameters,
                            device=device, verbose=verbose, logger=logger)
                    r.run()
                    exit()

                except KeyboardInterrupt:
                    # Allow keyboard interrupts such as CTRL+C to stop the execution
                    logger.warning(f'Experiment {experiment_id} interrupted via keyboard. Exiting.')
                    # Leave the program
                    exit()
                except:
                    logger.error(f'Experiment {experiment_id} failed. Error:')
                    logger.error(traceback.format_exc())
                    # Continue trying the next experiment
                    exit()
