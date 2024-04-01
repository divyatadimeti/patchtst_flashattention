import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser
import yaml

from dataloader import get_ETT_dataloaders
from patchtst_vanilla import get_vanilla_patchtst


''' Adapt existing model (PatchTSTForPrediction) to work
    within the LightningModule. Involves defining training, validation steps,
    and configuration of optimizers. Can modify patchtst_vanilla.py or include these changes in trainer.py
    directly under a new class like below.'''

class PatchTSTModel(pl.LightningModule):
    def __init__(self, model_config):
        super().__init__()
        self.model = get_vanilla_patchtst(model_config)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.mse_loss(y_hat, y)
        self.log('val_loss', val_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup Data
    train_dataloader, val_dataloader, _ = get_ETT_dataloaders(
        data_config=config["data"][config["training"]["dataset"]],
        context_length=config["model"]["context_length"],
        forecast_horizon=config["model"]["forecast_horizon"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"]
    )

    # Initialize Model
    model = PatchTSTModel(model_config=config["model"])

    # Setup Wandb Logger
    logger = WandbLogger(project=config["logging"]["wandb_project"],
                         entity=config["logging"]["wandb_entity"],
                         name=config["logging"]["wandb_run_name"])

    # Callbacks
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=3, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gpus=1,  # or adjust according to your setup
    )

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./config.yaml')
    args = parser.parse_args()

    main(args)
