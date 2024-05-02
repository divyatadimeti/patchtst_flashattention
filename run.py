import yaml
import argparse

from experiment_utils import patch_sizes_experiment, batch_sizes_experiment, dataset_experiment, num_workers_experiment, driver


def main(args):
    # Load the configs for model, training, and logging parameters
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    
    default_config = config["default_config"]
    model_config = default_config["model"]
    train_config = default_config["training"]
    data_config = default_config["data"]
    log_config = default_config["logging"]

    # Ensure only one experiment flag is set to True
    experiment_flags = [args.patch_size_exp, args.batch_size_exp, args.dataset_exp, args.num_workers_exp]
    assert sum(experiment_flags) == 1 or sum(experiment_flags) == 0, "Only one experiment flag should be set to True at a time."
    
    if sum(experiment_flags) == 1:
        attn_types = ["vanilla", "flash"]

        for attn in attn_types:
            model_config["attn_type"] = attn
            if args.patch_size_exp:
                patch_sizes_experiment(data_config, model_config, train_config, log_config)
            elif args.batch_size_exp:
                batch_sizes_experiment(data_config, model_config, train_config, log_config)
            elif args.dataset_exp:
                dataset_experiment(data_config, model_config, train_config, log_config)
            elif args.num_workers_exp:            
                num_workers_experiment(data_config, model_config, train_config, log_config)
    else:
        driver(data_config, model_config, train_config, log_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training PatchTST vanilla and FlashAttention2 models for benchmarking and profiling")
    parser.add_argument("-c", "--config", type=str, default="./config.yaml")
    parser.add_argument("--patch_size_exp", action="store_true")
    parser.add_argument("--batch_size_exp", action="store_true")
    parser.add_argument("--dataset_exp", action="store_true")
    parser.add_argument("--num_workers_exp", action="store_true")
    args = parser.parse_args()

    main(args)
