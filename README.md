# HPML Project: PatchTST and FlashAttention2 Benchmarking

### Intallation
**Clone the project**
``` 
git clone https://github.com/divyatadimeti/patchtst_flashattention.git
```
**Install requirements**
Make sure to set up the local environment with the correct python version (3.9 >= v >= 3.11) and install the requirements. We use Python 3.10 in our experiments. Additionally, FlashAttention requires CUDA 11.6 and can only currently be run on a specific list of compatible GPUs.

From the official documentation, FlashAttention2 currently supports Ampere, Ada, or Hopper GPUs (e.g., A100, RTX 3090, RTX 4090, H100). Support for Turing GPUs (T4, RTX 2080) is coming soon, please use FlashAttention 1.x for Turing GPUs for now.
```
conda create -n pflash python==3.10
pip install -r requirements.txt
```

**Clone the tsfm repository**
We use IBM's tsfm repository to access the PatchTST models. Clone the repository using the following command and follow the set up instructions in the README.md of the GitHub:
```
git clone https://github.com/IBM/tsfm.git
cd tsfm
pip install ".[notebooks]"
```

**Install FlashAttention2**
We use the official code from the FlashAttention repository to run PatchTST with FlashAttention2. It can be installed using the below command:
```
pip install -U flash-attn --no-build-isolation
```

### Download the Datasets
**Download the ETDataset and the Traffic dataset**
We use the official ETDataset and Traffic datasets that are commonly used to train time-series models. The ETT datasets can be downloaded with the following commands:
```
wget -O data/ETTh1.csv https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTh1.csv
wget -O data/ETTm1.csv https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTm1.csv
```
The Traffic dataset can be installed by creating an account with Caltrans PeMS and downloading the data from the website: https://dot.ca.gov/programs/traffic-operations/mpr/pems-source.

### Running the Code
**Config file**
The configuration in `config.yaml` has been provided as a default to run PatchTST with Vanilla and FlashAttention. To change which model is being used, ensure that `attn_type` under the `model` parameters is set to be either `vanilla` or `flash`. Modify any other hyperparameters, experimental parameters, data paths, and logging parameters from the configuration file before running the code.

**Experiments**
The model can be trained using the following command. Use the `--help` flag to display the experiment set up configurations. Custom configuration files can be specified with a flag.
```
python run.py
```
By default, the above command while run the main driver with the configurations in `config.yaml`. To run experiments using mini hyperparameter sweeps as outlined in our paper with patch size, batch size, datasets and number of workers, utilize the appropriate flags: `--patch_size_exp`, `--batch_size_exp`, `--dataset_exp`, `num_workers_exp`.