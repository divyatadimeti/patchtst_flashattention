import yaml
import argparse
import wandb

parser = argparse.ArgumentParser(description="Training PatchTST vanilla and FlashAttention2 models for benchmarking and profiling")
parser.add_argument("-c", "--config", type=str, default="./config.yaml")
args = parser.parse_args()

config = yaml.safe_load(open(args.config))

wandb.init(project=config["logging"]["wandb_project"], entity=config["logging"]["wandb_entity"])



