import yaml
import argparse
import wandb
import pytorch_lightning as pl

from dataloader import get_ETT_dataloaders
from patchtst_model import PatchTST
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

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

def dataset_experiment(data_config, model_config, train_config, log_config):
    datasets = ["ETTh1", "ETTm1", "Weather"]
    for data in datasets:
        data_config["dataset"] = data
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
    # Set up wandb logging and PyTorch Lightning logger
    logger = None
    if log_config["use_wandb"]:
        run = wandb.init(project=log_config["wandb_project"], 
                entity=log_config["wandb_entity"],
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
        early_stop_callback = EarlyStopping(monitor="val_loss", patience=train_config["patience"])
        callbacks.append(early_stop_callback)

    checkpoint_callback = ModelCheckpoint(dirpath=log_config["checkpoint_path"], 
                                              monitor="val_loss", 
                                              save_top_k=1, 
                                              mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(checkpoint_callback)
    callbacks.append(lr_monitor)

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
