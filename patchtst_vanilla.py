import torch
import time
import pytorch_lightning as pl
import torch.nn.functional as F

from transformers import PatchTSTConfig, PatchTSTForPrediction

class PatchTSTVanilla(pl.LightningModule):
    def __init__(self, model_config):
        super().__init__()
        config = PatchTSTConfig(do_mask_input=False,
                            context_length=model_config["context_length"],
                            patch_length=model_config["patch_length"],
                            num_input_channels=model_config["num_input_channels"],
                            patch_stride=model_config["patch_length"],
                            prediction_length=model_config["prediction_length"],
                            d_model=model_config["d_model"],
                            num_attention_heads=model_config["num_attention_heads"],
                            num_hidden_layers=model_config["num_hidden_layers"],
                            ffn_dim=512,
                            dropout=0.2,
                            head_dropout=0.2,
                            pooling_type=None,
                            channel_attention=False,
                            scaling="std",
                            loss="mse",
                            pre_norm=True,
                            norm_type="batchnorm")
        self.model = PatchTSTForPrediction(config=config)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.mse_loss(y_hat, y)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.mse_loss(y_hat, y)
        self.log('test_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        batch_time = time.time() - self.batch_start_time
        self.log('batch_time', batch_time, on_step=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
