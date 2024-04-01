import yaml
import argparse
import wandb

from dataloader import get_ETT_dataloaders

parser = argparse.ArgumentParser(description="Training PatchTST vanilla and FlashAttention2 models for benchmarking and profiling")
parser.add_argument("-c", "--config", type=str, default="./config.yaml")
args = parser.parse_args()

config = yaml.safe_load(open(args.config))

run = wandb.init(project=config["logging"]["wandb_project"], 
           entity=config["logging"]["wandb_entity"],
           name=config["logging"]["wandb_run_name"])
assert run is wandb.run

train_dataloader, valid_dataloader, test_dataloader = get_ETT_dataloaders(config["data"], 
                                                                          config["model"]["context_length"], 
                                                                          config["model"]["forecast_horizon"],
                                                                          config["training"]["batch_size"],
                                                                          config["training"]["num_workers"])



