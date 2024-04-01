import yaml
import argparse
import wandb

from dataloader import get_ETT_dataloaders
from patchtst_vanilla import get_vanilla_patchtst

parser = argparse.ArgumentParser(description="Training PatchTST vanilla and FlashAttention2 models for benchmarking and profiling")
parser.add_argument("-c", "--config", type=str, default="./config.yaml")
args = parser.parse_args()

config = yaml.safe_load(open(args.config))
model_config = config["model"]
train_config = config["training"]
data_config = config["data"]
log_config = config["logging"]

run = wandb.init(project=log_config["wandb_project"], 
           entity=log_config["wandb_entity"],
           name=log_config["wandb_run_name"])
assert run is wandb.run

dataset = train_config["dataset"]

if dataset == "ETTh1" or dataset == "ETTm1":
    train_dataloader, valid_dataloader, test_dataloader = get_ETT_dataloaders(data_config[dataset], 
                                                                          model_config["context_length"], 
                                                                          model_config["forecast_horizon"],
                                                                          train_config["batch_size"],
                                                                          train_config["num_workers"])
else:
    pass

if model_config["model_type"] == "vanilla":
    model = get_vanilla_patchtst(model_config)
else:
    pass
