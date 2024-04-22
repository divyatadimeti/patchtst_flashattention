import torch
import time
import pytorch_lightning as pl
import torch.nn.functional as F

from transformers import PatchTSTConfig, PatchTSTForPrediction
from patchtst_flash import PatchTSTFlashConfig, PatchTSTFlashAttention2

class PatchTST(pl.LightningModule):
    def __init__(self, model_config):
        super().__init__()
        if model_config["attn_type"] == "vanilla":
            config = PatchTSTConfig(do_mask_input=False,
                                    context_length=model_config["context_length"],
                                    patch_length=model_config["patch_length"],
                                    num_input_channels=model_config["num_input_channels"],
                                    patch_stride=model_config["patch_length"],
                                    prediction_length=model_config["forecast_horizon"],
                                    d_model=model_config["d_model"],
                                    num_attention_heads=model_config["num_heads"],
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
        else:
            config = PatchTSTFlashConfig(causal=True,
                                    num_key_value_heads=model_config["num_key_value_heads"],
                                    do_mask_input=False,
                                    context_length=model_config["context_length"],
                                    patch_length=model_config["patch_length"],
                                    num_input_channels=model_config["num_input_channels"],
                                    patch_stride=model_config["patch_length"],
                                    prediction_length=model_config["forecast_horizon"],
                                    d_model=model_config["d_model"],
                                    num_attention_heads=model_config["num_heads"],
                                    num_hidden_layers=model_config["num_hidden_layers"],
                                    ffn_dim=512,
                                    dropout=0.2,
                                    head_dropout=0.2,
                                    pooling_type=None,
                                    channel_attention=False,
                                    scaling="std",
                                    loss="mse",
                                    pre_norm=True,
                                    norm_type="batchnorm",
                                    )
            self.model = PatchTSTFlashAttention2(config=config)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        y_hat = outputs.prediction_outputs  # Correctly accessing the prediction tensor
        mse_loss = F.mse_loss(y_hat, y)
        mae_loss = F.l1_loss(y_hat, y)
        self.log('train_mse_loss', mse_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mae_loss', mae_loss, on_epoch=True, prog_bar=True, logger=True)
        return mse_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        y_hat = outputs.prediction_outputs  # Correctly accessing the prediction tensor
        val_mse_loss = F.mse_loss(y_hat, y)
        val_mae_loss = F.l1_loss(y_hat, y)
        self.log('val_mse_loss', val_mse_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mae_loss', val_mae_loss, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        y_hat = outputs.prediction_outputs  # Correctly accessing the prediction tensor
        test_mse_loss = F.mse_loss(y_hat, y)
        test_mae_loss = F.l1_loss(y_hat, y)
        self.log('test_mse_loss', test_mse_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_mae_loss', test_mae_loss, on_epoch=True, prog_bar=True, logger=True)

    # Calculate compute time for a single batch (excluding data loading time)
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.batch_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batch_time = time.time() - self.batch_time
        self.log('compute_time', self.batch_time, on_step=True, logger=True)
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start_time
        self.log('total_time', epoch_time, on_step=True, logger=True)
        data_loading_time = epoch_time - self.batch_time
        self.log('data_loading_time', data_loading_time, on_step=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
