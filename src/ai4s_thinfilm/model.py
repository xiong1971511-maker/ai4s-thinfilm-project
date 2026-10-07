"""MLP surrogate model for four thickness inputs and 41 reflectance outputs."""

from typing import Literal

import torch
from torch import nn


class MLPRegressor(nn.Module):
    """4-128-128-64-41 ReLU MLP with selectable output activation."""

    layer_sizes = (4, 128, 128, 64, 41)

    def __init__(self, output_activation: Literal["linear", "sigmoid"] = "sigmoid") -> None:
        super().__init__()
        if output_activation not in ("linear", "sigmoid"):
            raise ValueError("output_activation must be 'linear' or 'sigmoid'")
        self.output_activation = output_activation
        layers: list[nn.Module] = []
        for input_size, output_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            layers.append(nn.Linear(input_size, output_size))
            if output_size != self.layer_sizes[-1]:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, thicknesses_scaled: torch.Tensor) -> torch.Tensor:
        """Predict reflectance spectra from thicknesses scaled to [0, 1]."""

        logits = self.network(thicknesses_scaled)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(logits)
        return logits
