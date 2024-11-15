"""This module provides parsed model parameters

Examples:
    >>> from params import tcn_ae_params
    >>> tcn_ae_params.n_epochs

"""

from typing import Any

import yaml
from pydantic import BaseModel, PositiveInt, DirectoryPath, PositiveFloat

PARAMS_PATH = "params.yaml"


class TcnAeParams(BaseModel):
    """Template for object storing parameters for
    TCN Autoencoder in experiments
    """

    random_seed: int
    n_epochs: PositiveInt
    lr: PositiveFloat
    enc_channels: list[PositiveInt]
    hidden_dim: PositiveInt
    dec_channels: list[PositiveInt]
    dilation_base: PositiveInt
    kernel_size: PositiveInt
    batch_size: PositiveInt
    train_window_len: PositiveInt
    checkpoints_dir: DirectoryPath


class BaselineParams(BaseModel):
    """Template for object storing parameters for
    baseline models in experiments
    """

    random_seed: int
    lof_n_neighbors: PositiveInt
    lof_other_params: dict[str, Any] | None = None


with open(PARAMS_PATH) as f:
    params_all = yaml.safe_load(f)


tcn_ae_params = TcnAeParams(**params_all["params"]["tcn_ae"])
baseline_params = BaselineParams(**params_all["params"]["baseline"])
