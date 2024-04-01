
from transformers import PatchTSTConfig, PatchTSTForRegression
import torch

config = PatchTSTConfig(
    num_input_channels=6,  # Number of features in the input time series
    context_length=32,     # Length of the input sequence
    prediction_length=24,  # Length of the output sequence for forecasting
    num_hidden_layers=3,   # Number of Transformer layers
    d_model=128,           # Dimensionality of the Transformer layers
    num_attention_heads=4, # Number of attention heads in the Transformer layers
    ffn_dim=512,           # Dimension of the feed-forward network
    patch_length=1,        # Length of each patch
    patch_stride=1,        # Stride of the patchification process
    # Other parameters as needed
)

model = PatchTSTForRegression(config=config)
