import yaml
from pydantic import BaseModel, PositiveInt, DirectoryPath, PositiveFloat

PARAMS_PATH = "params.yaml"


class TcnAeParams(BaseModel):
    n_epochs: PositiveInt
    lr: PositiveFloat
    batch_size: PositiveInt
    train_window_len: PositiveInt
    checkpoints_dir: DirectoryPath


with open(PARAMS_PATH) as f:
    params_all = yaml.safe_load(f)


tcn_ae_params = TcnAeParams(**params_all["params"]["tcn_ae"])
