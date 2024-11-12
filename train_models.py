"""Run this script with PYTHONPATH set to the src/ directory
or set the src/ directory as a source directory before running

Hyperparameters can be set in the params.yaml file

"""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow

from utils import normalize_values
from data_loading import load_all_srw_series, TimeSeriesWithAnoms, SlidingWindowDataset
from networks import TCNAutoencoder
from training_utils import eval_model, training_step
from params import tcn_ae_params


def prepare_data() -> list[TimeSeriesWithAnoms]:
    """Loads all time series from files
    and performs normalization

    """
    all_series = load_all_srw_series()
    for series in all_series:
        series.values = normalize_values(series.values)
    return all_series


def train_model_for_series(series: TimeSeriesWithAnoms) -> None:
    """Trains TCN Autoencoder model for a given time series
    and saves it to the file.

    In the training process, the model is evaluated after each epoch,
    and the model with the lowest loss is saved.
    The model is saved as a torch state dict to the file:
    <chackpoints_dir>/tcnae_<series_name>.pth

    """

    # dataset and dataloader
    dataset = SlidingWindowDataset(
        series.values, window_len=tcn_ae_params.train_window_len
    )
    dataloader = DataLoader(dataset, batch_size=tcn_ae_params.batch_size, shuffle=True)

    # path to save the model
    model_path = os.path.join(tcn_ae_params.checkpoints_dir, series.name + ".pth")

    # model object
    model = TCNAutoencoder(
        in_channels=1,
        enc_channels=tcn_ae_params.enc_channels,
        hidden_dim=tcn_ae_params.hidden_dim,
        dec_channels=tcn_ae_params.dec_channels,
        input_size=tcn_ae_params.train_window_len,
        dilation_base=tcn_ae_params.dilation_base,
        kernel_size=tcn_ae_params.kernel_size,
    )

    # learning prerequisites
    loss_fun = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=tcn_ae_params.lr)

    # learning process
    with mlflow.start_run(run_name=series.name) as run:
        best_loss = float("inf")
        epochs = list(range(tcn_ae_params.n_epochs))
        mlflow.log_params(tcn_ae_params.dict())
        for n_epoch in tqdm(epochs, desc=series.name):
            model.decoder.set_output_size(tcn_ae_params.train_window_len)
            for train_windows_batch in dataloader:
                training_step(model, train_windows_batch, loss_fun, optimizer)
            model.decoder.set_output_size(series.values.shape[-1])
            cur_loss = eval_model(model, series.values, loss_fun)
            mlflow.log_metric("train_loss", cur_loss, step=n_epoch)
            if cur_loss < best_loss:
                best_loss = cur_loss
                torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("TCN_AE_anomaly_detection")
    all_series = prepare_data()
    for series in all_series:
        train_model_for_series(series)
