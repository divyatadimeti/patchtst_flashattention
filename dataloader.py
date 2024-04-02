import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index

class ETTDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]['past_values'], self.data[index]['future_values']

def get_ETT_datasets(data_path, context_length, forecast_horizon):
    timestamp_column = "date"
    id_columns = []
    forecast_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    train_start_index = None
    train_end_index = 12 * 30 * 24

    valid_start_index = 12 * 30 * 24 - context_length
    valid_end_index = 12 * 30 * 24 + 4 * 30 * 24

    test_start_index = 12 * 30 * 24 + 4 * 30 * 24 - context_length
    test_end_index = 12 * 30 * 24 + 8 * 30 * 24

    data = pd.read_csv(
        data_path,
        parse_dates=[timestamp_column],
    )

    train_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=train_start_index,
        end_index=train_end_index,
    )
    valid_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=valid_start_index,
        end_index=valid_end_index,
    )
    test_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=test_start_index,
        end_index=test_end_index,
    )

    tsp = TimeSeriesPreprocessor(
        timestamp_column=timestamp_column,
        id_columns=id_columns,
        target_columns=forecast_columns,
        scaling=True,
    )
    tsp.train(train_data)

    train_dataset = ForecastDFDataset(
        tsp.preprocess(train_data),
        id_columns=id_columns,
        target_columns=forecast_columns,
        context_length=context_length,
        prediction_length=forecast_horizon,
    )
    valid_dataset = ForecastDFDataset(
        tsp.preprocess(valid_data),
        id_columns=id_columns,
        target_columns=forecast_columns,
        context_length=context_length,
        prediction_length=forecast_horizon,
    )
    test_dataset = ForecastDFDataset(
        tsp.preprocess(test_data),
        id_columns=id_columns,
        target_columns=forecast_columns,
        context_length=context_length,
        prediction_length=forecast_horizon,
    )

    return train_dataset, valid_dataset, test_dataset

def get_ETT_dataloaders(data_config, 
                        context_length, 
                        forecast_horizon,
                        batch_size=32, 
                        num_workers=2):
    data_path = data_config["data_path"]
    train_dataset, valid_dataset, test_dataset = get_ETT_datasets(data_path, context_length, forecast_horizon)
    
    train_dataset = ETTDataset(train_dataset)
    valid_dataset = ETTDataset(valid_dataset)
    test_dataset = ETTDataset(test_dataset)

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
    
    return train_dataloader, valid_dataloader, test_dataloader
