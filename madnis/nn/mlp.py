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
        super().__init__()
        self.channels = channels
        self.activation = activation()
        self.features_out = features_out

        input_dim = features_in
        layer_dims = []
        for i in range(layers - 1):
            layer_dims.append((input_dim, units))
            input_dim = units
        layer_dims.append((input_dim, features_out))

        self.weights = nn.ParameterList([
            torch.empty((n_channels, n_out, n_in)) for n_in, n_out in layer_dims
        ])
        self.biases = nn.ParameterList([
            torch.empty((n_channels, n_out)) for n_in, n_out in layer_dims
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for ws, bs in zip(self.weights[:-1], self.biases[:-1]):
            for w, b in zip(ws, bs):
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(b, -bound, bound)
        nn.init.zeros_(self.weights[-1])
        nn.init.zeros_(self.biases[-1])

    def forward_single(self, x: torch.Tensor, channel: int) -> torch.Tensor:
        if x.shape[0] == 0:
            return x.new_zeros((x.shape[0], self.features_out))
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = self.activation(F.linear(x, w[channel], b[channel]))
        return F.linear(x, self.weights[-1][channel], self.biases[-1][channel])

    def forward_uniform(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.reshape(self.channels, batch_size // self.channels, x.shape[1])
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = self.activation(torch.baddbmm(b[:,None,:], x, w.transpose(1,2)))
        return torch.baddbmm(b[:,None,:], x, w.transpose(1,2))

    def forward(
        self,
        x: torch.Tensor,
        channel: list[int] | int | None = None,
    ) -> torch.Tensor:
        if isinstance(channel, list):
            return torch.cat([
                self.forward_single(xi, i) for i, xi in enumerate(x.split(channel, dim=0))
            ], dim=0)
        elif isinstance(channel, int):
            return self.forward_single(x, channel)
        else:
            return self.forward_uniform(x)
