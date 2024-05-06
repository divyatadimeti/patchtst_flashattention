import yaml
import argparse

from experiment_utils import patch_sizes_experiment, batch_sizes_experiment, num_workers_experiment, driver

def main(args):
    """
    Main function to run different PatchTST model experiments based on the provided arguments for experiments.
    
    Args:
        args (Namespace): Parsed command line arguments.
        
    This function loads a configuration file, updates the PatchTST model configuration based on the
    experiment flags, and runs the specified experiment. Only one experiment can be
    conducted at a time as ensured by the assertion.
    """
    # Load configuration from the YAML file specified by the command line argument
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    
    # Extract default configurations for model, training, data, and logging
    default_config = config["default_config"]
    model_config = default_config["model"]
    train_config = default_config["training"]
    data_config = default_config["data"]
    log_config = default_config["logging"]

    # Check that only one experiment is set to run
    experiment_flags = [args.patch_size_exp, args.batch_size_exp, args.num_workers_exp]
    assert sum(experiment_flags) == 1 or sum(experiment_flags) == 0, "Only one experiment flag should be set to True at a time."
    
    # Determine which experiment to run based on the flags
    if sum(experiment_flags) == 1:
        attn_types = ["vanilla", "flash"]

        # Loop through each attention type and run the appropriate experiment
        for attn in attn_types:
            model_config["attn_type"] = attn
            if args.patch_size_exp:
                patch_sizes_experiment(data_config, model_config, train_config, log_config)
            elif args.batch_size_exp:
                batch_sizes_experiment(data_config, model_config, train_config, log_config)
            elif args.num_workers_exp:            
                num_workers_experiment(data_config, model_config, train_config, log_config)
    else:
        # Run the main driver function if no experiment flag is set
        driver(data_config, model_config, train_config, log_config)

if __name__ == '__main__':
    # Parser setup for command line arguments
    parser = argparse.ArgumentParser(description="Training PatchTST vanilla and FlashAttention2 models for benchmarking and profiling")
    parser.add_argument("-c", "--config", type=str, default="./config.yaml", help="The configuration file path to use for training the model")
    parser.add_argument("--patch_size_exp", action="store_true", help="Use this flag to run experiments with modifying patch size while keeping remaining hyperparameters as default")
    parser.add_argument("--batch_size_exp", action="store_true", help="Use this flag to run experiments with modifying batch size while keeping remaining hyperparameters as default")
    parser.add_argument("--num_workers_exp", action="store_true", help="Use this flag to run experiments with modifying number of workers while keeping remaining hyperparameters as default")
    args = parser.parse_args()

    main(args)
