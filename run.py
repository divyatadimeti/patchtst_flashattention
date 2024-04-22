import yaml
import argparse
import wandb
import pytorch_lightning as pl

from dataloader import get_ETT_dataloaders
from patchtst_model import PatchTST
from experiment_utils import patch_sizes_experiment, batch_sizes_experiment, dataset_experiment, num_workers_experiment, driver
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor


def main(args):
    # Load the configs for model, training, and logging parameters
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    
    config = config["default_config"]
    model_config = config["model"]
    train_config = config["training"]
    data_config = config["data"]
    log_config = config["logging"]
    
    attn_types = ["vanilla", "flash"]

    for attn in attn_types:
        model_config["attn_type"] = attn
        patch_sizes_experiment(data_config, model_config, train_config, log_config)
        batch_sizes_experiment(data_config, model_config, train_config, log_config)
        dataset_experiment(data_config, model_config, train_config, log_config)
        num_workers_experiment(data_config, model_config, train_config, log_config)


    # driver(model_config, train_config, log_config, data_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training PatchTST vanilla and FlashAttention2 models for benchmarking and profiling")
    parser.add_argument("-c", "--config", type=str, default="./config.yaml")
    args = parser.parse_args()

    main(args)
