from typing import Callable, Literal
import math

import torch
import torch.nn as nn
import numpy as np

from .splines import unconstrained_rational_quadratic_spline
from .mlp import MLP, StackedMLP


L2PI = -0.5 * math.log(2 * math.pi)

Mapping = Callable[[torch.Tensor, bool], tuple[torch.Tensor, torch.Tensor]]


class Flow(nn.Module):
    def __init__(
        self,
        dims_in: int,
        dims_c: int = 0,
        uniform_latent: bool = True,
        condition_masks: torch.Tensor | None = None,
        permutations: Literal["log", "random", "exchange"] = "log",
        subnet_constructor: Callable[[int, int], nn.Module] | None = None,
        blocks: int | None = None,
        layers: int = 3,
        units: int = 32,
        bins: int = 10,
        activation: Callable[[], nn.Module] = nn.ReLU,
        layer_constructor: Callable[[int, int], nn.Module] = nn.Linear,
        spline_bounds: float = 10.0,
        channels: int | None = None,
        mapping: Mapping | list[Mapping] | None = None,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_bin_derivative: float = 1e-3,
    ):
        super().__init__()

        self.dims_in = dims_in
        self.dims_c = dims_c
        self.uniform_latent = uniform_latent
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_bin_derivative = min_bin_derivative

        if uniform_latent:
            spline_low = -spline_bounds
            spline_high = spline_bounds
        else:
            spline_low = 0
            spline_high = 1

        if subnet_constructor is None:
            if channels is None:
                subnet_constructor = lambda features_in, features_out: MLP(
                    features_in,
                    features_out,
                    layers,
                    units,
                    activation,
                    layer_constructor,
                )
            else:
                subnet_constructor = lambda features_in, features_out: StackedMLP(
                    features_in,
                    features_out,
                    channels,
                    layers,
                    units,
                    activation,
                    layer_constructor,
                )

        if condition_masks is None:
            if permutations == "log":
                n_perms = int(np.ceil(np.log2(dims_in)))
                blocks = int(2 * n_perms)
                condition_masks = (
                    torch.tensor(
                        [
                            [int(i) for i in np.binary_repr(i, n_perms)]
                            for i in range(dims_in)
                        ]
                    )
                    .flip(dims=(1,))
                    .bool()
                    .t()
                    .repeat_interleave(2, dim=0)
                )
                condition_masks[1::2, :] ^= True
            elif permutations == "exchange":
                condition_masks = torch.cat(
                    (
                        torch.ones(dims_in // 2, dtype=torch.bool),
                        torch.zeros(dims_in - dims_in // 2, dtype=torch.bool),
                    )
                )[None, :].repeat((blocks, 1))
                condition_masks[1::2, :] ^= True
            elif permutations == "random":
                condition_masks = torch.cat(
                    (
                        torch.ones(dims_in // 2, dtype=torch.bool),
                        torch.zeros(dims_in - dims_in // 2, dtype=torch.bool),
                    )
                )[None, :].repeat((blocks, 1))
                for i in range(blocks):
                    condition_masks[i] = condition_masks[i][torch.randperm(dims_in)]
            else:
                raise ValueError(f"Unknown permutation type {permutation}")
        self.condition_masks = condition_masks

        self.subnets = nn.ModuleList()
        for mask in self.condition_masks:
            dims_cond = torch.count_nonzero(mask)
            self.subnets.append(
                subnet_constructor(
                    dims_cond + dims_c, (dims_in - dims_cond) * (3 * bins + 1)
                )
            )

    def transform(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        inverse: bool = False,
        channel: torch.Tensor | list[int] | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        channel_perm = None
        if channel_equal:
            channel_sizes = []
            channel_id = -2
        elif channel_sizes is not None:
            channel_id = -1
        elif isinstance(channel, int):
            channel_id = channel
        elif channel is not None:
            channel_id = -1
            if not channel_sorted:
                channel_perm = torch.argsort(channel)
                x = x[channel_perm]
                c = c[channel_perm]
                channel = channel[channel_perm]
            _, channel_sizes = torch.unique(channel, return_counts=True)

        jac = 0.0

        if channel_sizes is None:
            channel_args = ()
        else:
            channel_args = (channel_sizes, channel_id)

        batch_size = x.shape[0]
        if inverse:
            blocks = zip(reversed(self.condition_masks), reversed(self.subnets))
        else:
            blocks = zip(self.condition_masks, self.subnets)
        for mask, subnet in blocks:
            inv_mask = ~mask
            x_trafo = x[:, inv_mask]
            if c is not None:
                x_cond = torch.cat((x_cond, c), dim=1)
            subnet_out = subnet(x_cond, *channel_args)
            x_out, block_jac = unconstrained_rational_quadratic_spline(
                x_trafo,
                subnet_out.reshape((batch_size, x_trafo.shape[1], -1)),
                inverse,
                self.min_bin_width,
                self.min_bin_height,
                self.min_bin_derivative,
            )
            x[:, inv_mask] = x_out
            if inverse:
                jac -= block_jac.sum(dim=1)
            else:
                jac += block_jac.sum(dim=1)

        if channel_perm is not None:
            channel_perm_inv = torch.argsort(channel_perm)
            x = x[channel_perm_inv]
            jac = jac[channel_perm_inv]

        return x, jac

    def latent_log_prob(self, z: torch.Tensor):
        if self.uniform_latent:
            return 0.0
        else:
            return z.shape[1] * L2PI - z.square().sum(dim=1) / 2

    def log_prob(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        channel: torch.Tensor | list[int] | int | None = None,
        return_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        z, jac = self.transform(x, c, False, channel)
        log_prob = self.latent_log_prob(z) + jac
        if return_latent:
            return log_prob, z
        else:
            return log_prob

    def prob(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        channel: torch.Tensor | list[int] | int | None = None,
        return_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        log_prob, z = self.log_prob(x, c, channel, True)
        if return_latent:
            return log_prob.exp(), z
        else:
            return log_prob.exp()

    def sample(
        self,
        n: int | None = None,
        c: torch.Tensor | None = None,
        channel: torch.Tensor | list[int] | int | None = None,
        return_log_prob: bool = False,
        return_prob: bool = False,
        return_latent: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if n is None:
            n = len(c)
        options = {} if c is None else {"device": c.device, "dtype": c.dtype}
        if device is not None:
            options["device"] = device
        if dtype is not None:
            options["dtype"] = dtype

        if self.uniform_latent:
            z = torch.rand((n,), **options)
        else:
            z = torch.randn((n,), **options)
        x, jac = self.transform(
            z, c, True, channel, channel_sizes, channel_sorted, channel_equal
        )
        if return_log_prob or return_prob:
            log_prob_latent = self.latent_log_prob(z)
            log_prob = log_prob_latent + jac

        extra_returns = []
        if return_log_prob:
            extra_returns.append(log_prob)
        if return_prob:
            extra_returns.append(prob)
        if return_latent:
            extra_returns.append(z)
        if len(extra_returns) > 0:
            return x, *extra_returns
        else:
            return x
