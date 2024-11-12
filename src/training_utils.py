import torch
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss

from networks import TCNAutoencoder


def eval_model(
    model: TCNAutoencoder,
    data: torch.Tensor,
    loss_fun: _Loss,
) -> float:
    """Computes loss value of the model in the evaluating mode

    Args:
        model: ``TCNAutoencoder`` class object
        data: tensor of shape (batch_size, n_channels, input_size)
        loss_fun: PyTorch loss to compute the loss value

    Returns:
        loss value

    """
    model.eval()
    data_recon = model.forward(data)
    loss = loss_fun(data_recon, data)
    return loss.item()


def training_step(
    model: TCNAutoencoder,
    data: torch.Tensor,
    loss_fun: _Loss,
    optimizer: Optimizer,
) -> None:
    """Performs one update of parameters of an autoencoder
    using reconstruction error on given data

    Args:
        model: ``TCNAutoencoder`` class object
        data: tensor of shape (batch_size, n_channels, input_size)
        loss_fun: PyTorch loss to compute the loss value
        optimizer: optimizer object

    """

    # prerequisites
    model.train()
    optimizer.zero_grad()

    # reconstruct data and compute reconstruction loss
    data_recon = model.forward(data)
    loss = loss_fun(data_recon, data)

    # update model parameters
    loss.backward()
    optimizer.step()
