from collections.abc import Iterable
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow import Distribution
from .masked_mlp import MaskedMLP


class DiscreteMADE(nn.Module, Distribution):
    def __init__(
        self,
        dims_in: list[int],
        dims_c: int = 0,
        channels: int | None = None,
        prior_prob_function: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
        channel_remap_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
        **mlp_kwargs,  # TODO: replace this with default arguments for MLP
    ):
        """ """
        super().__init__()

        self.autoreg_net = MaskedMLP(
            input_dims=[dims_c, *dims_in[:-1]],
            output_dims=dims_in,
            channels=channels,
            **mlp_kwargs,
        )
        self.dims_in = dims_in
        self.max_dim = max(dims_in)
        self.channels = 1 if channels is None else channels
        self.prior_prob_function = prior_prob_function
        self.channel_remap_function = channel_remap_function

        discrete_indices = []
        one_hot_mask = []
        for i, dim in enumerate(dims_in):
            discrete_indices.extend([i] * dim)
            one_hot_mask.extend([True] * dim + [False] * (self.max_dim - dim))
        self.register_buffer("discrete_indices", torch.tensor(discrete_indices))
        self.register_buffer("one_hot_mask", torch.tensor(one_hot_mask))
        self.register_buffer("dims_in_tensor", torch.tensor(dims_in))
        self.register_buffer("dummy", torch.tensor([0.0]))

    def _sort_channels(
        self,
        channel: torch.Tensor | list[int] | int | None,
        tensors: Iterable[torch.Tensor | None, ...],
    ) -> tuple[
        list[int] | int | None, torch.Tensor | None, tuple[torch.Tensor | None, ...]
    ]:
        if isinstance(channel, torch.Tensor):
            if self.channel_remap_function is not None:
                channel = self.channel_remap_function(channel)
            channel_perm = torch.argsort(channel)
            channel_sizes = channel.bincount(minlength=self.channels).tolist()
            perm_tensors = tuple(
                tensor[channel_perm] if tensor is not None else None
                for tensor in tensors
            )
            return channel_sizes, channel_perm, perm_tensors

        if self.channel_remap_function is not None:
            if isinstance(channel, int):
                channel = self.channel_remap_function(channel)
            elif isinstance(channel, list):
                raise ValueError(
                    "channel_remap_function not supported if called "
                    "with list of channel sizes"
                )
        return channel, None, tensors

    def _unsort_channels(
        self,
        channel_perm: torch.Tensor | None,
        tensors: Iterable[torch.Tensor | None, ...],
    ) -> tuple[torch.Tensor | None, ...]:
        if channel_perm is None:
            return tensors
        channel_perm_inv = torch.argsort(channel_perm)
        return tuple(
            tensor[channel_perm_inv] if tensor is not None else None
            for tensor in tensors
        )

    def log_prob(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        channel: torch.Tensor | list[int] | int | None = None,
        return_one_hot: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """ """
        if self.prior_prob_function is not None:
            prior_probs = torch.cat(
                [
                    self.prior_prob_function(x[:, :i], i)
                    for i, _ in enumerate(self.dims_in)
                ],
                dim=1,
            )
        else:
            prior_probs = None

        x_one_hot = (
            F.one_hot(x, self.max_dim)
            .to(self.dummy.dtype)
            .flatten(start_dim=1)[:, self.one_hot_mask]
        )

        channel, channel_perm, (x_one_hot, c, prior_probs) = self._sort_channels(
            channel, (x_one_hot, c, prior_probs)
        )

        net_input = x_one_hot if c is None else torch.cat((c, x_one_hot), dim=1)
        net_prob = self.autoreg_net(net_input[:, : -self.dims_in[-1]], channel).exp()
        unnorm_prob = (
            net_prob if self.prior_prob_function is None else net_prob * prior_probs
        )

        prob_norms = torch.zeros_like(x, dtype=x_one_hot.dtype).scatter_add_(
            dim=1,
            index=self.discrete_indices[None, :].expand(x.shape[0], -1),
            src=unnorm_prob,
        )
        prob_sums = torch.zeros_like(prob_norms).scatter_add_(
            dim=1,
            index=self.discrete_indices[None, :].expand(x.shape[0], -1),
            src=unnorm_prob * x_one_hot,
        )
        prob = torch.prod(prob_sums / prob_norms, dim=1)

        if return_one_hot:
            return self._unsort_channels(channel_perm, (prob.log(), net_input))
        else:
            return self._unsort_channels(channel_perm, (prob.log(),))[0]

    def sample(
        self,
        n: int | None = None,
        c: torch.Tensor | None = None,
        channel: torch.Tensor | list[int] | int | None = None,
        return_log_prob: bool = False,
        return_prob: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        return_one_hot: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """ """
        if n is None:
            n = len(c)
        if c is None:
            options = {"device": device, "dtype": dtype}
            x_in = torch.zeros((n, 0), **options)
        else:
            options = {"device": c.device, "dtype": c.dtype}
            x_in = c

        prob = torch.ones((n,), **options)
        net_cache = None

        channel, channel_perm, (x_in,) = self._sort_channels(channel, (x_in,))

        x_out = torch.zeros(
            (n, len(self.dims_in)),
            device=options["device"],
            dtype=torch.int64,
        )
        for i, dim in enumerate(self.dims_in):
            y, net_cache = self.autoreg_net.forward_cached(x_in, i, net_cache, channel)
            net_probs = y.exp()
            if self.prior_prob_function is None:
                unnorm_probs = net_probs
            else:
                prior_probs = self.prior_prob_function(x_out[:, :i], i)
                unnorm_probs = net_probs * prior_probs
            cdf = unnorm_probs.cumsum(dim=1)
            norm = cdf[:, -1]
            cdf = cdf / norm[:, None]
            r = torch.rand((y.shape[0], 1), **options)
            samples = torch.searchsorted(cdf, r)[:, 0]
            x_out[:, i] = samples
            prob = prob * torch.gather(unnorm_probs, 1, samples[:, None])[:, 0] / norm
            x_in = F.one_hot(samples, dim).to(y.dtype)

        return_list = [x_out]
        if return_log_prob:
            return_list.append(prob.log())
        if return_prob:
            return_list.append(prob)
        if return_one_hot:
            x_in_prev = torch.cat([cache[0] for cache in net_cache], dim=0)
            return_list.append(torch.cat((x_in_prev, x_in), dim=1))
        return_list = self._unsort_channels(channel_perm, return_list)
        if len(return_list) > 1:
            return return_list
        else:
            return return_list[0]
