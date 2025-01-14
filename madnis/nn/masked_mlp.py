import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMLP(nn.Module):
    def __init__(
        self,
        input_dims: list[int],
        output_dims: list[int],
        layers: int = 3,
        nodes_per_feature: int = 8,
        activation=nn.LeakyReLU,
        channels: int | None = None,
    ):
        super().__init__()

        in_degrees = []
        for i, in_dim in enumerate(input_dims):
            in_degrees.extend([i] * in_dim)
        hidden_degrees = torch.repeat_interleave(
            torch.arange(len(input_dims)), nodes_per_feature
        )
        out_degrees = []
        for i, out_dim in enumerate(output_dims):
            out_degrees.extend([i] * out_dim)
        hidden_layers = layers - 1
        layer_degrees = [
            torch.tensor(in_degrees),
            *([hidden_degrees] * hidden_layers),
            torch.tensor(out_degrees),
        ]

        self.in_slices = [[slice(0)] * layers]
        self.out_slices = [[slice(0)] * layers]
        hidden_dims = [nodes_per_feature] * hidden_layers
        for in_dim, out_dim in zip(input_dims, output_dims):
            self.in_slices.append(
                [
                    slice(0, prev_slice_in.stop + deg_in)
                    for deg_in, prev_slice_in in zip(
                        [in_dim, *hidden_dims], self.in_slices[-1]
                    )
                ]
            )
            self.out_slices.append(
                [
                    slice(prev_slice_out.stop, prev_slice_out.stop + deg_out)
                    for deg_out, prev_slice_out in zip(
                        [*hidden_dims, out_dim], self.out_slices[-1]
                    )
                ]
            )
        self.in_slices.pop(0)
        self.out_slices.pop(0)

        self.channels = 1 if channels is None else channels
        self.masks = nn.ParameterList()
        self.weights = nn.ModuleList()
        self.biases = nn.ModuleList()

        for i in range(self.channels):
            chan_weights = nn.ParameterList()
            chan_biases = nn.ParameterList()
            for deg_in, deg_out in zip(layer_degrees[:-1], layer_degrees[1:]):
                if i == 0:
                    self.masks.append(
                        nn.Parameter(
                            (deg_out[:, None] >= deg_in[None, :]).float(),
                            requires_grad=False,
                        )
                    )
                chan_weights.append(
                    nn.Parameter(torch.empty((len(deg_out), len(deg_in))))
                )
                chan_biases.append(nn.Parameter(torch.empty((len(deg_out),))))
            self.weights.append(chan_weights)
            self.biases.append(chan_biases)

        self.activation = activation()
        self.reset_parameters()

    def reset_parameters(self):
        for chan_weights, chan_biases in zip(self.weights, self.biases):
            for weight, bias in zip(chan_weights[:-1], chan_biases[:-1]):
                torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(bias, -bound, bound)
            nn.init.zeros_(chan_weights[-1])
            nn.init.zeros_(chan_biases[-1])

    def forward(
        self,
        x: torch.Tensor,
        channel: list[int] | int | None = None,
    ) -> torch.Tensor:
        if self.channels == 1:
            return self._forward_single(x, 0)
        elif isinstance(channel, list):
            return torch.cat(
                [
                    self._forward_single(xi, i)
                    for i, xi in enumerate(x.split(channel, dim=0))
                ],
                dim=0,
            )
        elif isinstance(channel, int):
            return self._forward_single(x, channel)
        else:
            # TODO: implement optimized version for the uniform special case
            return torch.cat(
                [
                    self._forward_single(xi, i)
                    for i, xi in enumerate(x.split(self.channels, dim=0))
                ],
                dim=0,
            )

    def forward_cached(
        self,
        x: torch.Tensor,
        feature: int,
        caches: list[list[torch.Tensor] | None] | None = None,
        channel: list[int] | int | None = None,
    ) -> tuple[torch.Tensor, list[list[torch.Tensor] | None]]:
        caches = itertools.repeat(None) if caches is None else caches
        if self.channels == 1:
            return self._forward_cached_single(x, feature, next(iter(caches)), 0)
        elif isinstance(channel, list):
            xs, caches = zip(
                *[
                    self._forward_cached_single(xi, feature, cache, i)
                    for i, (xi, cache) in enumerate(
                        zip(x.split(channel, dim=0), caches)
                    )
                ]
            )
            return torch.cat(xs, dim=0), caches
        elif isinstance(channel, int):
            return self._forward_cached_single(x, feature, next(iter(caches)), channel)
        else:
            # TODO: implement optimized version for the uniform special case
            xs, caches = zip(
                *[
                    self._forward_cached_single(xi, feature, cache, i)
                    for i, (xi, cache) in enumerate(
                        zip(x.split(self.channels, dim=0), caches)
                    )
                ]
            )
            return torch.cat(xs, dim=0), caches

    def _forward_single(self, x: torch.Tensor, channel: int) -> torch.Tensor:
        weights, biases = self.weights[channel], self.biases[channel]
        for weight, bias, mask in zip(weights[:-1], biases[:-1], self.masks[:-1]):
            x = self.activation(F.linear(x, mask * weight, bias))
        return F.linear(x, self.masks[-1] * weights[-1], biases[-1])

    def _forward_cached_single(
        self,
        x: torch.Tensor,
        feature: int,
        cache: list[torch.Tensor] | None,
        channel: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        weights, biases = self.weights[channel], self.biases[channel]
        if cache is None:
            cache = [None] * len(weights)
        new_cache = []
        in_slices = self.in_slices[feature]
        out_slices = self.out_slices[feature]
        first = True
        for weight, bias, in_slice, out_slice, x_cached in zip(
            weights, biases, in_slices, out_slices, cache
        ):
            if first:
                first = False
            else:
                x = self.activation(x)
            if x_cached is not None:
                x = torch.cat((x_cached, x), dim=1)
            new_cache.append(x)
            x = F.linear(x, weight[out_slice, in_slice], bias[out_slice])
        return x, new_cache
