# HPML Project: PatchTST and FlashAttention2 Benchmarking

### Intallation
**Clone the project**
``` 
git clone https://github.com/divyatadimeti/patchtst_flashattention.git
```
**Install requirements**
Make sure to set up the local environment with the correct python version (3.9 >= v >= 3.11) and install the requirements. We use Python 3.10 in our experiments.
```
conda create -n pflash python==3.10
pip install -r requirements.txt
```

**Clone the tsfm repository**
We use IBM's tsfm repository to access the PatchTST models. Clone the repository using the following command and follow the set up instructions in the README.md of the GitHub:
```
git clone https://github.com/IBM/tsfm.git
```
