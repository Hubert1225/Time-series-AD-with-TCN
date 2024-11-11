import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tcn import TCN


class TemporalEncoder(nn.Module):

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
