import yaml
import argparse
import wandb
import pytorch_lightning as pl
import numpy as np
import time

from dataloader import get_ETT_dataloaders
from patchtst_model import PatchTST
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, Callback


class MetricLogger(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.total_epoch_times = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start_time
        self.total_epoch_times.append(epoch_time)
        self.log('total_time', epoch_time, on_step=False, on_epoch=True, logger=True)

    def on_train_end(self, trainer, pl_module):
        self.avg_total_time = np.mean(self.total_epoch_times)
        trainer.logger.experiment.log({"avg_total_time": self.avg_total_time})

def patch_sizes_experiment(data_config, model_config, train_config, log_config):
    patch_sizes = [12, 24, 48, 96, 192]
    for patch_size in patch_sizes:
        model_config["patch_length"] = patch_size
        attn_type = model_config["attn_type"]
        log_config["wandb_run_name"] = f"patchtst_{attn_type}_patchsize_{patch_size}"
        driver(data_config, model_config, train_config, log_config)

def batch_sizes_experiment(data_config, model_config, train_config, log_config):
    batch_sizes = [32, 64, 128, 256, 512]
    for batch_size in batch_sizes:
        train_config["batch_size"] = batch_size
        attn_type = model_config["attn_type"]
        log_config["wandb_run_name"] = f"patchtst_{attn_type}_batchsize_{batch_size}"
        driver(data_config, model_config, train_config, log_config)

def num_workers_experiment(data_config, model_config, train_config, log_config):
    num_workers = [2, 4, 8, 16]
    for num in num_workers:
        train_config["num_workers"] = num
        attn_type = model_config["attn_type"]
        log_config["wandb_run_name"] = f"patchtst_{attn_type}_numworkers_{num}"
        driver(data_config, model_config, train_config, log_config)

def driver(data_config, model_config, train_config, log_config):
    # Set up wandb logging and PyTorch Lightning logger
    logger = None
    if log_config["use_wandb"]:
        run = wandb.init(project=log_config["wandb_project"], 
                entity=log_config["wandb_entity"],
                tags=["gpu.0.memory"],
                name=log_config["wandb_run_name"])
        assert run is wandb.run

        wandb.config.update(model_config)
        wandb.config.update(train_config)
        wandb.config.update(data_config)
        wandb.config.update(log_config)

        logger = WandbLogger(project=log_config["wandb_project"],
                            entity=log_config["wandb_entity"],
                            name=log_config["wandb_run_name"])

    # Load dataset and dataloaders depending on the dataset chosen for training
    dataset = data_config["dataset"]

    train_dataloader, val_dataloader, test_dataloader = get_ETT_dataloaders(data_config[dataset], 
                                                                        model_config["context_length"], 
                                                                        model_config["forecast_horizon"],
                                                                        train_config["batch_size"],
                                                                        train_config["num_workers"])

    # Load the appropriate model (either Vanilla or FlashAttention2)
    if model_config["attn_type"] == "vanilla":
        model = PatchTST(model_config)
    else:
        model = PatchTST(model_config)

    # Set up callbacks for early stopping, model checkpointing and learning rate scheduling
    callbacks = []
    if train_config["early_stopping"]:
        early_stop_callback = EarlyStopping(monitor="val_mse_loss", patience=train_config["patience"])
        callbacks.append(early_stop_callback)

    checkpoint_callback = ModelCheckpoint(dirpath=log_config["checkpoint_path"], 
                                              monitor="val_mse_loss", 
                                              save_top_k=1, 
                                              mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    metric_logger = MetricLogger()
    callbacks.append(checkpoint_callback)
    callbacks.append(lr_monitor)
    callbacks.append(metric_logger)

    # Set up the trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=train_config["epochs"],
        logger=logger,
        callbacks=callbacks,
        profiler="simple",
    )

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)

    # Log avergage data loading time
    if log_config["use_wandb"]:
        avg_data_loading_time = np.mean(trainer.profiler.recorded_durations["[_TrainingEpochLoop].train_dataloader_next"])
        wandb.log({"avg_data_loading_time": avg_data_loading_time})

    # Test the model
    trainer.test(model, dataloaders=test_dataloader)

    if log_config["use_wandb"]:
        wandb.finish()
