import torch
import time
import pytorch_lightning as pl
import torch.nn.functional as F

from transformers import PatchTSTConfig, PatchTSTForPrediction
from patchtst_flash import PatchTSTFlashConfig, PatchTSTFlashAttention2
from pruning_utils import prune_head, dynamic_prune

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

            if model_config["prune_heads"]:
                for head in model_config["prune_heads"]:
                    self.prune_attention_layers(head_to_prune=head)
            
            if model_config["dynamic_prune"]:
                self.prune_attention_layers(dynamic=True)

    def prune_attention_layers(self, head_to_prune=None, dynamic=False):
        for encoder_layer in self.model.model.encoder.layers:
            if dynamic:
                encoder_layer.self_attn = dynamic_prune(encoder_layer.self_attn)
            else:
                encoder_layer.self_attn = prune_head(encoder_layer.self_attn, head_to_prune)

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
