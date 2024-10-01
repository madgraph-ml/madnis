from typing import Callable

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        features_in: int,
        features_out: int,
        layers: int,
        units: int,
        activation: Callable[[], nn.Module] = nn.ReLU,
        layer_constructor: Callable[[int, int], nn.Module] = nn.Linear,
    ):
        super().__init__()
        input_dim = features_in
        layer_list = []
        for i in range(layers - 1):
            layer_list.append(layer_constructor(input_dim, units))
            layer_list.append(activation())
            input_dim = units
        layer_list.append(layer_constructor(input_dim, features_out))
        nn.init.zeros_(layer_list[-1].weight)
        nn.init.zeros_(layer_list[-1].bias)
        self.layers = nn.Sequential(*layer_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class StackedMLP(nn.Module):
    def __init__(
        self,
        features_in: int,
        features_out: int,
        channels: int,
        layers: int,
        units: int,
        activation: Callable[[], nn.Module] = nn.ReLU,
        layer_constructor: Callable[[int, int], nn.Module] = nn.Linear,
    ):
        pass

    def forward(
        self,
        x: torch.Tensor,
        channel: list[int] | int | None = None,
    ) -> torch.Tensor:
        return self.layers(x)
