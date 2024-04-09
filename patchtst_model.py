import torch
import time
import pytorch_lightning as pl
import torch.nn.functional as F

from transformers import PatchTSTConfig, PatchTSTForPrediction
from patchtst_flash import PatchTSTFlashAttention2

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
        else:
            config = PatchTSTConfig(causal=True, #TODO: check whether this should be true
                                    do_mask_input=False,
                                    context_length=model_config["context_length"],
                                    patch_length=model_config["patch_length"],
                                    num_input_channels=model_config["num_input_channels"],
                                    patch_stride=model_config["patch_length"],
                                    prediction_length=model_config["forecast_horizon"],
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
            self.model = PatchTSTFlashAttention2(config=config)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        y_hat = outputs.prediction_outputs  # Correctly accessing the prediction tensor
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        y_hat = outputs.prediction_outputs  # Correctly accessing the prediction tensor
        val_loss = F.mse_loss(y_hat, y)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        y_hat = outputs.prediction_outputs  # Correctly accessing the prediction tensor
        test_loss = F.mse_loss(y_hat, y)
        self.log('test_loss', test_loss, on_epoch=True, prog_bar=True, logger=True)

    def on_train_batch_start(self, batch, batch_idx):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_time = time.time() - self.batch_start_time
        self.log('batch_time', batch_time, on_step=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
