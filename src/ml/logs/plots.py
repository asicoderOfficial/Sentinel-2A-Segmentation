import pandas as pd
import plotly.graph_objects as go

from src.ml.constants import LABELS


class TrainingPlots:
    def epoch_plot(df: pd.DataFrame, x_title: str, y_title: str, title: str, info_dir: str, file_name: str, name:str, y_axis_range:list=[]) -> None:
        """ Plots information by epoch.

        Args:
            df (pd.DataFrame): Dataframe with the training information by epoch.
            x_title (str): X-axis title.
            y_title (str): Y-axis title.
            title (str): Plot title.
            info_dir (str): Directory where to store the plot.
            file_name (str): Name of the file.

        Returns:
            None
        """        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['avg'], mode='lines+markers', name=name))
        # set x-axis title
        fig.update_xaxes(title_text=x_title)
        # set y-axis title
        fig.update_yaxes(title_text=y_title)
        # Update title and size
        fig.update_layout(title=title, width=800, height=600, hovermode='x unified')
        # Set y axis range between 0 and 1
        if y_axis_range:
            fig.update_yaxes(range=y_axis_range)
        fig.write_image(f'{info_dir}/{file_name}.png')


    def epoch_label_plot(df: pd.DataFrame, x_title: str, y_title: str, title: str, info_dir: str, file_name: str, name:str, y_axis_range:list=[]) -> None:
        """ Plots information by epoch and label.

        Args:
            df (pd.DataFrame): Dataframe with the training information by epoch.
            x_title (str): X-axis title.
            y_title (str): Y-axis title.
            title (str): Plot title.
            info_dir (str): Directory where to store the plot.
            file_name (str): Name of the file.

        Returns:
            None
        """        
        fig = go.Figure()
        for label in LABELS:
            fig.add_trace(go.Scatter(x=df['epoch'], y=df[label], mode='lines+markers', name=label))
        # set x-axis title
        fig.update_xaxes(title_text=x_title)
        # set y-axis title
        fig.update_yaxes(title_text=y_title)
        # Update title and size
        fig.update_layout(title=title, width=800, height=600, hovermode='x unified')
        # Set y axis range between 0 and 1
        if y_axis_range:
            fig.update_yaxes(range=y_axis_range)
        fig.write_image(f'{info_dir}/{file_name}.png')


    @staticmethod
    def loss_by_epoch(epoch_loss_df: pd.DataFrame, info_dir: str) -> None:
        """ Plots the loss by epoch.

        Args:
            epoch_loss_df (pd.DataFrame): Dataframe with the loss by epoch.
            info_dir (str): Directory where to store the plot.

        Returns:
            None
        """        
        TrainingPlots.epoch_plot(epoch_loss_df, 'Epoch', 'Loss', 'Mean loss by epoch', info_dir, 'average_loss_by_epoch', 'Training loss')


    @staticmethod
    def dice_by_epoch(train_epoch_dice_df: pd.DataFrame, val_epoch_dice_df: pd.DataFrame, info_dir: str) -> None:
        """ Plots the DICE by epoch.

        Args:
            train_epoch_dice_df (pd.DataFrame): Dataframe with the DICE by epoch for the training set.
            val_epoch_dice_df (pd.DataFrame): Dataframe with the DICE by epoch for the validation set.
            info_dir (str): Directory where to store the plot.

        Returns:
            None
        """        
        TrainingPlots.epoch_plot(train_epoch_dice_df, 'Epoch', 'DICE', 'Mean DICE by epoch (Training set)', info_dir, 'average_dice_by_epoch_train', 'Training DICE', y_axis_range=[0, 1])
        TrainingPlots.epoch_plot(val_epoch_dice_df, 'Epoch', 'DICE', 'Mean DICE by epoch (Validation set)', info_dir, 'average_dice_by_epoch_val', 'Validation DICE', y_axis_range=[0, 1])
    
    
    @staticmethod
    def dice_label_by_epoch(train_dice_by_label_df: pd.DataFrame, val_dice_by_label_df: pd.DataFrame, info_dir: str) -> None:
        """ Plots the DICE by label by epoch.

        Args:
            train_dice_by_label_df (pd.DataFrame): Dataframe with the DICE by label by epoch for the training set.
            val_dice_by_label_df (pd.DataFrame): Dataframe with the DICE by label by epoch for the validation set.
            info_dir (str): Directory where to store the plot.

        Returns:
            None
        """        
        TrainingPlots.epoch_label_plot(train_dice_by_label_df, 'Epoch', 'DICE', f'DICE by label by epoch (Training set)', info_dir, f'dice_by_label_by_epoch_train', f'Training DICE', y_axis_range=[0, 1])
        TrainingPlots.epoch_label_plot(val_dice_by_label_df, 'Epoch', 'DICE', f'DICE by label by epoch (Validation set)', info_dir, f'dice_by_label_by_epoch_val', f'Validation DICE', y_axis_range=[0, 1])


    @staticmethod
    def time_by_epoch(train_time_by_epoch_df: pd.DataFrame, info_dir: str) -> None:
        """ Plots the time by epoch.

        Args:
            train_time_by_epoch_df (pd.DataFrame): Dataframe with the time by epoch for the training set.
            info_dir (str): Directory where to store the plot.

        Returns:
            None
        """        
        TrainingPlots.epoch_plot(train_time_by_epoch_df, 'Epoch', 'Time', 'Time by epoch (Training set)', info_dir, 'time_by_epoch_train', 'Training time')
