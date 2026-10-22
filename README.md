# STPD

This repository contains the implementation of Semantic Tag-enhanced Preference Disentanglement for Cross-Domain Recommendation (STPD), along with baseline implementations.

## Prerequisites

Environment: Python 3.9.18, PyTorch 2.1.2, and CUDA 12.2.

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Datasets

### Downloading

Download the metadata and ratings-only files for the following categories from [Amazon Review Data (2018)](https://jmcauley.ucsd.edu/data/amazon_v2/index.html), and save them in the corresponding directories:

- Movie (Movies and TV): `datasets/raw/Amazon/Movie`
- Music (CDs and Vinyl): `datasets/raw/Amazon/Music`
- Cell (Cell Phones and Accessories): `datasets/raw/Amazon/Cell`
- Elec (Electronics): `datasets/raw/Amazon/Elec`

## Implementation Details

Model configurations and hyperparameters are specified in `config.yaml`.

## Running

From the directory containing `main.py`, run the following command to train, validate, and test the model. Use `--domains` to specify the domain pair:

```bash
python main.py --domains Movie-Music --approach STPD
```

The supported domain pairs are `Movie-Music` and `Cell-Elec`.

`STPD` is the default method and is implemented in `models/STPD.py`.
