from transformers import (
    PatchTSTConfig,
    PatchTSTForPrediction
)

def get_vanilla_patchtst(params):
    config = PatchTSTConfig(do_mask_input=False,
                            context_length=params["context_length"],
                            patch_length=params["patch_length"],
                            num_input_channels=params["num_input_channels"],
                            patch_stride=params["patch_length"],
                            prediction_length=params["prediction_length"],
                            d_model=params["d_model"],
                            num_attention_heads=params["num_attention_heads"],
                            num_hidden_layers=params["num_hidden_layers"],
                            ffn_dim=512,
                            dropout=0.2,
                            head_dropout=0.2,
                            pooling_type=None,
                            channel_attention=False,
                            scaling="std",
                            loss="mse",
                            pre_norm=True,
                            norm_type="batchnorm")
    model = PatchTSTForPrediction(config=config)
    return model
