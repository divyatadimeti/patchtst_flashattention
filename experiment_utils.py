import yaml
import argparse
import wandb
import pytorch_lightning as pl
import time

from dataloader import get_ETT_dataloaders
from patchtst_model import PatchTST
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, Callback


class MetricLogger(Callback):
    # Calculate compute time for a single batch (excluding data loading time)
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.batch_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batch_time = time.time() - self.batch_time
        self.log('compute_time', self.batch_time, on_step=True, on_epoch=False, logger=True)
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start_time
        self.log('total_time', epoch_time, on_step=False, on_epoch=True, logger=True)

        data_loading_time = epoch_time - self.batch_time
        self.log('data_loading_time', data_loading_time, on_step=False, on_epoch=True, logger=True)

def patch_sizes_experiment(data_config, model_config, train_config, log_config):
    patch_sizes = [12, 24, 48, 96, 192]
    for patch_size in patch_sizes:
        model_config["patch_length"] = patch_size
        attn_type = model_config["attn_type"]
        log_config["wandb_run_name"] = f"patchtst_{attn_type}_patchsize_{patch_size}"
        driver(data_config, model_config, train_config, log_config)

def batch_sizes_experiment(data_config, model_config, train_config, log_config):
    batch_sizes = [32, 64, 128]
    for batch_size in batch_sizes:
        train_config["batch_size"] = batch_size
        attn_type = model_config["attn_type"]
        log_config["wandb_run_name"] = f"patchtst_{attn_type}_batchsize_{batch_size}"
        driver(data_config, model_config, train_config, log_config)

def dataset_experiment(data_config, model_config, train_config, log_config):
    datasets = ["ETTh1", "ETTm1"]
    for data in datasets:
        data_config["dataset"] = data
        data_config["data_path"] = f"data/{data}.csv"
        data_config["resolution"] = 1 if data == "ETTh1" else 4
        attn_type = model_config["attn_type"]
        log_config["wandb_run_name"] = f"patchtst_{attn_type}_dataset_{data}"
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

    if dataset == "ETTh1" or dataset == "ETTm1":
        train_dataloader, val_dataloader, test_dataloader = get_ETT_dataloaders(data_config, 
                                                                            model_config["context_length"], 
                                                                            model_config["forecast_horizon"],
                                                                            train_config["batch_size"],
                                                                            train_config["num_workers"])
    else:
        pass

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
    
    metrics_callback = MetricLogger()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(checkpoint_callback)
    callbacks.append(lr_monitor)
    callbacks.append(metrics_callback)

    # Set up the trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=train_config["epochs"],
        logger=logger,
        callbacks=callbacks,
    )

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)

    # Test the model
    trainer.test(model, dataloaders=test_dataloader)

    wandb.finish()
