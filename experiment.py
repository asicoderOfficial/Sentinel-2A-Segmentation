import torch
import traceback
from sklearn.utils import check_random_state

from src.ml.dataset import SentinelDataset
from src.ml.models.unet import UNet
from src.ml.constants import RANDOM_SEED
from src.ml.run import Run
from src.ml.logs.logger import Logger

# Constants
# Set a fixed random seed for reproducibility
torch.manual_seed(RANDOM_SEED)
check_random_state(RANDOM_SEED)

in_channels = 13
out_channels = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
train_percentages = [0.7, 0.8, 0.9]
batch_sizes = [128, 16, 32, 64]
epochs = [15, 40, 100]
learning_rates = [0.001, 0.01, 0.1]
early_stopping = {'delta': 0.001, 'patience': 4}

# Data
dataset = SentinelDataset('data/Berlin')



for curr_train_percentage in train_percentages:
    for curr_batch_size in batch_sizes:
        for curr_epoch in epochs:
            for curr_learning_rate in learning_rates:
                try:
                    print(f'Running experiment with train percentage {curr_train_percentage}, batch size {curr_batch_size}, epoch {curr_epoch} and learning rate {curr_learning_rate}.')
                    experiment_id = f'UNet_trperc_{curr_train_percentage}_batch_{curr_batch_size}_epoch_{curr_epoch}_lr_{curr_learning_rate}'
                    # Model hyperparameters
                    model_hyperparameters = {'batch_size': curr_batch_size, 'learning_rate': curr_learning_rate}
                    # Trainer hyperparameters
                    trainer_hyperparameters = {'epochs': curr_epoch, 'early_stopping': early_stopping}

                    # Model
                    model = UNet(in_channels=in_channels, out_channels=out_channels)

                    # Loss function
                    criterion = torch.nn.BCEWithLogitsLoss()

                    # Optimizer
                    optimizer = torch.optim.Adam(model.parameters(), lr=curr_learning_rate)

                    r = Run(id=experiment_id, train_dataset=dataset, model=model, criterion=criterion, optimizer=optimizer,
                            train_percentage=curr_train_percentage, trainer_hyperparameters=trainer_hyperparameters, model_hyperparameters=model_hyperparameters,
                            device=device, verbose=True)

                    # Execute the experiment! Cross your fingers!
                    r.run()

                # Allow keyboard interrupts such as CTRL+C to stop the execution
                except KeyboardInterrupt:
                    print()
                    print(f'Experiment {experiment_id} interrupted.')
                    # Leave the program
                    exit()
                except:
                    print(f'Experiment {experiment_id} failed.')
                    print(traceback.format_exc())
                    exit()
                    Logger.error(run_info_dir=r.run_info_dir, error=traceback.format_exc())
                    continue
