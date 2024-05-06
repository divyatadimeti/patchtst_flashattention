import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index

class ETTDataset(Dataset):
    """
    ETTDataset is a custom PyTorch Dataset for handling the Energy Trading Time-series (ETT) dataset.
    This class is designed to work with time-series data, specifically formatted for forecasting tasks.
    
    Attributes:
        data (list): A list of dictionaries where each dictionary contains 'past_values' and 'future_values'
                     keys with their corresponding time-series data.
    
    Methods:
        __init__(self, data): Initializes the ETTDataset instance with the provided data.
        __len__(self): Returns the number of samples in the dataset.
        __getitem__(self, index): Retrieves the (past_values, future_values) tuple at the specified index.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]['past_values'], self.data[index]['future_values']

def get_ETT_datasets(data_path, context_length, forecast_horizon, resolution):
    """
    Prepares and returns the train, validation, and test datasets for the ETT dataset.

    This function reads the ETT dataset from a CSV file, preprocesses it, and splits it into
    train, validation, and test datasets based on the provided indices. It also applies
    scaling and other preprocessing steps necessary for time-series forecasting.

    Args:
        data_path (str): The path to the CSV file containing the ETT dataset.
        context_length (int): The number of past time steps to include in each sample.
        forecast_horizon (int): The number of future time steps to predict.
        resolution (int): The resolution of the data in terms of time steps.

    Returns:
        tuple: A tuple containing three ForecastDFDataset objects for training, validation, and testing.
    """
    # Read the dataset from the specified CSV file and parse the date column
    data = pd.read_csv(
        data_path,
        parse_dates=["date"],
    )

    # Define the indices for splitting the data into training, validation, and testing sets
    train_start_index = None
    train_end_index = 12 * 30 * 24 * resolution
    valid_start_index = train_end_index - context_length
    valid_end_index = train_end_index + 4 * 30 * 24 * resolution
    test_start_index = valid_end_index - context_length
    test_end_index = valid_end_index + 4 * 30 * 24 * resolution

    # Select the subsets of data for training, validation, and testing
    train_data = select_by_index(data, start_index=train_start_index, end_index=train_end_index)
    valid_data = select_by_index(data, start_index=valid_start_index, end_index=valid_end_index)
    test_data = select_by_index(data, start_index=test_start_index, end_index=test_end_index)

    # Initialize the TimeSeriesPreprocessor with the necessary parameters
    tsp = TimeSeriesPreprocessor(
        timestamp_column="date",
        target_columns=["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
        scaling=True,
    )

    # Fit the preprocessor on the training data and preprocess all datasets
    tsp.train(train_data)
    train_dataset = ForecastDFDataset(tsp.preprocess(train_data), context_length=context_length, prediction_length=forecast_horizon)
    valid_dataset = ForecastDFDataset(tsp.preprocess(valid_data), context_length=context_length, prediction_length=forecast_horizon)
    test_dataset = ForecastDFDataset(tsp.preprocess(test_data), context_length=context_length, prediction_length=forecast_horizon)

    # Return the prepared datasets
    return train_dataset, valid_dataset, test_dataset

def get_ETT_dataloaders(data_config, 
                        context_length, 
                        forecast_horizon,
                        batch_size=32, 
                        num_workers=2):
    """
    Create data loaders for the ETT datasets.

    This function prepares data loaders for the training, validation, and testing datasets. It uses the configuration
    provided to fetch the datasets, wrap them into PyTorch DataLoader objects, and return them for use in training
    and evaluation.

    Args:
        data_config (dict): Configuration dictionary containing data paths and resolution.
        context_length (int): The number of past time steps to include in each sample.
        forecast_horizon (int): The number of future time steps to predict.
        batch_size (int, optional): Number of samples in each batch. Defaults to 32.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 2.

    Returns:
        tuple: A tuple containing three DataLoader objects for training, validation, and testing datasets.
    """
    # Extract the data path and resolution from the configuration
    data_path = data_config["data_path"]
    resolution = data_config["resolution"]

    # Retrieve datasets for training, validation, and testing
    train_dataset, valid_dataset, test_dataset = get_ETT_datasets(data_path, context_length, forecast_horizon, resolution)
    
    # Wrap the datasets with the custom ETTDataset class
    train_dataset = ETTDataset(train_dataset)
    valid_dataset = ETTDataset(valid_dataset)
    test_dataset = ETTDataset(test_dataset)

    # Create DataLoader for each dataset
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    # Return the data loaders
    return train_dataloader, valid_dataloader, test_dataloader
