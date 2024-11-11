"""This module contains definitions of neural networks classes
used in the experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tcn import TCN


class TemporalEncoder(nn.Module):
    """Encoder class for time series subsequence anomaly
    detection

    Args:
        n_inputs: number of channels of input
        n_channels: number of channels for consecutive hidden layers
        n_outputs: number of channels of output
        kernel_size: kernel size of convolutions in TCN
        dilation_base: number to determine dilation for each layer in TCN;
            i-th layer will have dilation ``dilation_base**i``

    """

    def __init__(
        self,
        n_inputs: int,
        n_channels: list[int],
        n_outputs: int,
        kernel_size: int,
        dilation_base: int,
    ):
        super().__init__()
        self.tcn = TCN(
            num_inputs=n_inputs,
            num_channels=n_channels,
            kernel_size=kernel_size,
            dilations=[dilation_base**i for i in range(len(n_channels))],
            no_padding=True
        )
        self.conv1 = nn.Conv1d(
            kernel_size=1, in_channels=n_channels[-1], out_channels=n_outputs
        )
        self.pool = nn.MaxPool1d(32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tcn(x)
        x = self.conv1(x)
        return self.pool(x)


class TemporalDecoder(nn.Module):
    """Decoder class for time series subsequence anomaly
    detection

    Args:
        n_inputs: number of channels of input
        n_channels: number of channels for consecutive hidden layers
        n_outputs: number of channels of output
        kernel_size: kernel size of convolutions in TCN
        dilation_base: number to determine dilation for each layer in TCN;
            (N-i)-th layer will have dilation ``dilation_base**i``,
            N - number of hidden layers
        output_size: desired size of the output (last dimension)

    Notes:
        `self.output_size` can be changed dynamically with
        the `set_output_size` method

    """

    def __init__(
        self,
        n_inputs: int,
        n_channels: list[int],
        n_outputs: int,
        kernel_size: int,
        dilation_base: int,
        output_size: int,
    ):
        super().__init__()
        self.tcn = TCN(
            num_inputs=n_inputs,
            num_channels=n_channels,
            kernel_size=kernel_size,
            dilations=[dilation_base**i for i in range(len(n_channels)-1, -1, -1)],
        )
        self.output_size = output_size
        self.conv1 = nn.Conv1d(
            kernel_size=1, in_channels=n_channels[-1], out_channels=n_outputs
        )

    def set_output_size(self, output_size: int) -> None:
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, (self.output_size,))
        x = self.tcn(x)
        return self.conv1(x)


class TCNAutoencoder(nn.Module):
    """Autoencoder class for time series subsequence anomaly
    detection

    Args:
        in_channels: number of channels of input
        enc_channels: number of channels for consecutive hidden layers
            in the encoder
        hidden_dim: number of channels of the encoder's output
            (latent representation)
        dec_channels: number of channels for consecutive hidden layers
            in the decoder
        input_size: size of the input (last dimension)
        dilation_base: number to determine dilation for each layer in TCN;
            i-th layer in the encoder (and (N-i)th layer in the decoder)
            will have dilation ``dilation_base**i``,
            N - number of hidden layers in the decoder
        kernel_size: kernel size of convolutions in TCN

    """

    def __init__(
        self,
        in_channels: int,
        enc_channels: list[int],
        hidden_dim: int,
        dec_channels: list[int],
        input_size: int,
        dilation_base: int = 2,
        kernel_size: int = 6,
    ):
        super().__init__()
        self.encoder = TemporalEncoder(
            n_inputs=in_channels,
            n_channels=enc_channels,
            n_outputs=hidden_dim,
            kernel_size=kernel_size,
            dilation_base=dilation_base,
        )
        self.decoder = TemporalDecoder(
            n_inputs=hidden_dim,
            n_channels=dec_channels,
            n_outputs=in_channels,
            output_size=input_size,
            kernel_size=kernel_size,
            dilation_base=dilation_base
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.decoder(x)
