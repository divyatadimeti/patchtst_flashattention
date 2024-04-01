import yaml
import argparse
import wandb
import pytorch_lightning as pl

from dataloader import get_ETT_dataloaders
from patchtst_vanilla import PatchTSTVanilla
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

def main(args):
    # Load the configs for model, training, and logging parameters
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
        
    model_config = config["model"]
    train_config = config["training"]
    data_config = config["data"]
    log_config = config["logging"]

    # Set up wandb logging and PyTorch Lightning logger
    run = wandb.init(project=log_config["wandb_project"], 
            entity=log_config["wandb_entity"],
            name=log_config["wandb_run_name"])
    assert run is wandb.run

    logger = WandbLogger(project=config["logging"]["wandb_project"],
                         entity=config["logging"]["wandb_entity"],
                         name=config["logging"]["wandb_run_name"])

    # Load dataset and dataloaders depending on the dataset chosen for training
    dataset = train_config["dataset"]

    if dataset == "ETTh1" or dataset == "ETTm1":
        train_dataloader, val_dataloader, test_dataloader = get_ETT_dataloaders(data_config[dataset], 
                                                                            model_config["context_length"], 
                                                                            model_config["forecast_horizon"],
                                                                            train_config["batch_size"],
                                                                            train_config["num_workers"])
    else:
        pass

    # Load the appropriate model (either Vanilla or FlashAttention2)
    if model_config["model_type"] == "vanilla":
        model = PatchTSTVanilla(model_config)
    else:
        pass

    # Set up callbacks for early stopping, model checkpointing and learning rate scheduling
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=train_config["patience"])
    checkpoint_callback = ModelCheckpoint(dirpath=log_config["checkpoint_path"], 
                                              monitor="val_loss", 
                                              save_top_k=1, 
                                              mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Set up the trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        gpus=1,
    )

    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training PatchTST vanilla and FlashAttention2 models for benchmarking and profiling")
    parser.add_argument("-c", "--config", type=str, default="./config.yaml")
    args = parser.parse_args()

    main(args)