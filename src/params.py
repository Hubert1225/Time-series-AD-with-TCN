"""This module provides parsed model parameters

Examples:
    >>> from params import tcn_ae_params
    >>> tcn_ae_params.n_epochs

"""


import yaml
from pydantic import BaseModel, PositiveInt, DirectoryPath, PositiveFloat

PARAMS_PATH = "params.yaml"


class TcnAeParams(BaseModel):
    """Template for object storing parameters for
    TCN Autoencoder in experiments
    """
    n_epochs: PositiveInt
    lr: PositiveFloat
    batch_size: PositiveInt
    train_window_len: PositiveInt
    checkpoints_dir: DirectoryPath


with open(PARAMS_PATH) as f:
    params_all = yaml.safe_load(f)


tcn_ae_params = TcnAeParams(**params_all["params"]["tcn_ae"])
