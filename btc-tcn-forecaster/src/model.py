import torch
import torch.nn as nn
from pytorch_tcn import TCN


class TCNModel(nn.Module):
    def __init__(self, num_inputs: int, num_channels: int, num_layers: int, kernel_size: int, dropout: float):
        super().__init__()
        self.tcn = TCN(
            num_inputs=num_inputs,
            num_channels=[num_channels] * num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.head = nn.Linear(num_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.tcn(x.transpose(1, 2))
        return self.head(y[:, :, -1]).squeeze(-1)
