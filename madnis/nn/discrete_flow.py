from typing import Any, Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow import Distribution, Flow
from .masked_mlp import MaskedMLP

PriorProbFunction = (
    Callable[[torch.Tensor, int], torch.Tensor]
    | Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor | None]]
)


class DiscreteFlow(nn.Module, Distribution):
    def __init__(
        self,
        dims_in: list[int],
        dims_c: int = 0,
        channels: int | None = None,
        prior_prob_function: PriorProbFunction | None = None,
        prior_prob_mode: Literal["indices", "states"] = "indices",
        mode: Literal["indices", "cdf"] = "indices",
        channel_remap_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
        **mlp_kwargs,  # TODO: replace this with default arguments for MLP
    ):
        """ """
        super().__init__()

        self.masked_net = MaskedMLP(
            input_dims=[dims_c, *dims_in[:-1]], output_dims=dims_in, **mlp_kwargs
        )
        self.dims_in = dims_in
        self.max_dim = max(dims_in)
        self.prior_prob_function = prior_prob_function
        self.channel_remap_function = channel_remap_function
        if prior_prob_mode == "indices":
            self.prior_uses_indices = True
        elif prior_prob_mode == "states":
            self.prior_uses_indices = False
        else:
            raise ValueError(f"Unknown prior probability mode '{prior_prob_mode}'")
        if mode == "indices":
            self.cdf_mode = False
        elif mode == "cdf":
            self.cdf_mode = True
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        discrete_indices = []
        one_hot_mask = []
        for i, dim in enumerate(dims_in):
            discrete_indices.extend([i] * dim)
            one_hot_mask.extend([True] * dim + [False] * (self.max_dim - dim))
        self.register_buffer("discrete_indices", torch.tensor(discrete_indices))
        self.register_buffer("one_hot_mask", torch.tensor(one_hot_mask))
        self.register_buffer("dims_in_tensor", torch.tensor(dims_in))
        self.register_buffer("dummy", torch.tensor([0.0]))

    def _init_state(
        self,
        n: int,
        channel: torch.Tensor | list[int] | int | None,
        device: torch.device,
    ):
        # TODO: do sth with remapped channel here
        if channel is None:
            return torch.zeros(n, dtype=torch.int64, device=device)
        if isinstance(channel, int):
            return torch.full(n, channel, device=device)
        if isinstance(channel, torch.Tensor):
            return channel

        if channel_remap_function is not None:
            raise ValueError(
                "channel_remap_function not supported if called with list of channel sizes"
            )
        state = torch.zeros((sum(channel),), dtype=torch.int64, device=device)
        start_index = 0
        for chan_id, chan_size in enumerate(channel):
            end_index = start_index + chan_size
            state[start_index:end_index] = chan_id
        return state

    def log_prob(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        channel: torch.Tensor | list[int] | int | None = None,
        return_net_input: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """ """
        if self.prior_prob_function is None:
            if self.cdf_mode:
                x_indices = (x * self.dims_in_tensor).long()
            else:
                x_indices = x
        else:
            if self.cdf_mode:
                x_indices = torch.ones(
                    (x.shape[0], len(self.dims_in)), device=x.device, dtype=torch.int64
                )
            else:
                x_indices = x
            prior_probs = []
            if self.prior_uses_indices:
                for i, _ in enumerate(self.dims_in):
                    probs = self.prior_prob_function(x_indices[:, :i], i)
                    prior_probs.append(probs)
                    if self.cdf_mode:
                        cdf = probs.cumsum(dim=1)
                        x_indices[:, i : i + 1] = torch.searchsorted(
                            cdf / cdf[:, -1:], x[:, i : i + 1]
                        )
            else:
                state = self._init_state(x.shape[0], channel, device=x.device)
                for i, _ in enumerate(self.dims_in):
                    probs, new_states = self.prior_prob_function(state, i)
                    prior_probs.append(probs)
                    if self.cdf_mode:
                        cdf = probs.cumsum(dim=1)
                        x_indices[:, i : i + 1] = torch.searchsorted(
                            cdf / cdf[:, -1:], x[:, i : i + 1].clone()
                        )
                    if i != len(self.dims_in) - 1:
                        state = torch.gather(
                            new_states, dim=1, index=x_indices[:, i : i + 1]
                        )[:, 0]
            prior_probs = torch.cat(prior_probs, dim=1)

        x_one_hot = (
            F.one_hot(x_indices, self.max_dim)
            .to(self.dummy.dtype)
            .flatten(start_dim=1)[:, self.one_hot_mask]
        )

        net_input = x_one_hot if c is None else torch.cat((c, x_one_hot), dim=1)
        net_prob = self.masked_net(net_input[:, : -self.dims_in[-1]]).exp()
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
        if self.cdf_mode:
            if self.prior_prob_function is None:
                prob = torch.prod(prob_sums / prob_norms * self.dims_in_tensor, dim=1)
            else:
                prior_prob_norms = torch.zeros_like(
                    x, dtype=x_one_hot.dtype
                ).scatter_add_(
                    dim=1,
                    index=self.discrete_indices[None, :].expand(x.shape[0], -1),
                    src=prior_probs,
                )
                prior_prob_sums = torch.zeros_like(prob_norms).scatter_add_(
                    dim=1,
                    index=self.discrete_indices[None, :].expand(x.shape[0], -1),
                    src=prior_probs * x_one_hot,
                )
                prob = torch.prod(
                    (prob_sums / prob_norms) / (prior_prob_sums / prior_prob_norms),
                    dim=1,
                )
        else:
            prob = torch.prod(prob_sums / prob_norms, dim=1)

        if return_net_input:
            return prob.log(), net_input
        else:
            return prob.log()

    def sample(
        self,
        n: int | None = None,
        c: torch.Tensor | None = None,
        channel: torch.Tensor | list[int] | int | None = None,
        return_log_prob: bool = False,
        return_prob: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        return_net_input: bool = False,
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
        if not self.prior_uses_indices:
            state = self._init_state(n, channel, device=options["device"])

        x_out = torch.ones(
            (n, len(self.dims_in)),
            device=options["device"],
            dtype=prob.dtype if self.cdf_mode else torch.int64,
        )
        for i, dim in enumerate(self.dims_in):
            y, net_cache = self.masked_net.forward_cached(x_in, i, net_cache)
            net_probs = y.exp()
            if self.prior_prob_function is None:
                unnorm_probs = net_probs
            else:
                if self.prior_uses_indices:
                    prior_probs = self.prior_prob_function(x_out[:, :i], i)
                else:
                    prior_probs, new_states = self.prior_prob_function(state, i)
                unnorm_probs = net_probs * prior_probs
            cdf = unnorm_probs.cumsum(dim=1)
            norm = cdf[:, -1]
            cdf = cdf / norm[:, None]
            r = torch.rand((y.shape[0], 1), **options)
            samples = torch.searchsorted(cdf, r)[:, 0]
            if self.cdf_mode:
                if self.prior_prob_function is None:
                    prob = (
                        prob
                        * torch.gather(unnorm_probs, 1, samples[:, None])[:, 0]
                        / norm
                        * dim
                    )
                    x_out[:, i] = (samples + 0.5) / dim
                else:
                    cdf_prior = prior_probs.cumsum(dim=1)
                    norm_prior = cdf_prior[:, -1]
                    cdf_prior = cdf_prior / norm_prior[:, None]
                    prob_y = (
                        torch.gather(unnorm_probs, 1, samples[:, None])[:, 0] / norm
                    )
                    prob_x = (
                        torch.gather(prior_probs, 1, samples[:, None])[:, 0]
                        / norm_prior
                    )
                    cdf_x = torch.gather(cdf_prior, 1, samples[:, None])[:, 0]
                    prob = prob * prob_y / prob_x
                    x_out[:, i] = cdf_x - prob_x / 2
            else:
                x_out[:, i] = samples
                prob = (
                    prob * torch.gather(unnorm_probs, 1, samples[:, None])[:, 0] / norm
                )
            if not self.prior_uses_indices and i != len(self.dims_in) - 1:
                state = torch.gather(new_states, dim=1, index=samples[:, None])[:, 0]
            x_in = F.one_hot(samples, dim).to(y.dtype)

        extra_returns = []
        if return_log_prob:
            extra_returns.append(prob.log())
        if return_prob:
            extra_returns.append(prob)
        if return_net_input:
            extra_returns.append(torch.cat((net_cache[0], x_in), dim=1))
        if len(extra_returns) > 0:
            return x_out, *extra_returns
        else:
            return x_out

    def init_with_grid(self, grid: torch.Tensor):
        probs = []
        for i, dim in enumerate(self.dims_in):
            # grid_i and x are clone because searchsorted would complain otherwise
            grid_i = grid[..., i, :].clone()
            x = (
                torch.linspace(0, 1, dim + 1)[(None,) * (len(grid_i.shape) - 1)]
                .expand(*grid_i.shape[:-1], -1)
                .clone()
            )
            cdf_vals = torch.searchsorted(grid_i, x)
            probs.append(cdf_vals.diff(dim=-1))
        log_probs = torch.cat(probs, dim=-1).log()

        weights = self.masked_net.weights[-1]
        biases = self.masked_net.biases[-1]
        # TODO: multichannel case
        nn.init.zeros_(weights)
        with torch.no_grad():
            biases.copy_(log_probs[0, :])


class MixedFlow(nn.Module, Distribution):
    def __init__(
        self,
        dims_in_continuous: int,
        dims_in_discrete: list[int],
        dims_c: int = 0,
        discrete_dims_position: Literal["first", "last"] = "first",
        channels: int | None = None,
        continuous_kwargs: dict[str, Any] = {},
        discrete_kwargs: dict[str, Any] = {},
    ):
        """ """
        super().__init__()
        if discrete_dims_position == "first":
            self.discrete_dims_first = True
        elif discrete_dims_position == "last":
            self.discrete_dims_first = False
        else:
            raise ValueError("discrete_dims_position must be 'first' or 'last'")
        self.dims_in_continuous = dims_in_discrete
        self.dims_in_discrete = len(dims_in_discrete)
        if self.discrete_dims_first:
            dims_c_discrete = dims_c
            dims_c_continuous = sum(dims_in_discrete) + dims_c
        else:
            dims_c_discrete = dims_c + dims_in_continuous
            dims_c_continuous = dims_c
        self.discrete_flow = DiscreteFlow(
            dims_in=dims_in_discrete,
            dims_c=dims_c_discrete,
            channels=channels,
            **discrete_kwargs,
        )
        self.continuous_flow = Flow(
            dims_in=dims_in_continuous,
            dims_c=dims_c_continuous,
            channels=channels,
            **continuous_kwargs,
        )

    def log_prob(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        channel: torch.Tensor | list[int] | int | None = None,
    ) -> torch.Tensor:
        """
        Computes the log-probabilities of the input data.

        Args:
            x: input data, shape (n, dims_in)
            c: condition, shape (n, dims_c) or None for an unconditional flow
            channel: encodes the channel of the samples. It must have one of the following types:

                - ``Tensor``: integer tensor of shape (n, ), containing the channel index for every
                  input sample;
                - ``list``: list of integers, specifying the number of samples in each channel;
                - ``int``: integer specifying a single channel containing all the samples;
                - ``None``: used in the single-channel case or to indicate that all channels contain
                  the same number of samples in the multi-channel case.
        Returns:
            log-probabilities with shape (n, )
        """
        if self.discrete_dims_first:
            x_discrete = x[:, : self.dims_in_discrete]
            log_prob_discrete, condition = self.discrete_flow.log_prob(
                x_discrete, c=c, channel=channel, return_net_input=True
            )
            log_prob_continuous = self.continuous_flow.log_prob(
                x[:, self.dims_in_discrete :], c=condition, channel=channel
            )
        else:
            x_continuous = x[:, : self.dims_in_continuous]
            log_prob_continuous = self.continuous_flow.log_prob(
                x_continuous, c=c, channel=channel
            )
            condition = (
                x_continuous if c is None else torch.cat((c, x_continuous), dim=1)
            )
            log_prob_discrete = self.discrete_flow.log_prob(
                x[:, self.dim_in_continuous], c=condition, channel=channel
            )
        return log_prob_discrete + log_prob_continuous

    def sample(
        self,
        n: int | None = None,
        c: torch.Tensor | None = None,
        channel: torch.Tensor | list[int] | int | None = None,
        return_log_prob: bool = False,
        return_prob: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """ """
        if self.discrete_dims_first:
            x_discrete, prob_discrete, condition = self.discrete_flow.sample(
                n=n,
                c=c,
                channel=channel,
                return_prob=True,
                device=device,
                dtype=dtype,
                return_net_input=True,
            )
            x_continuous, log_prob_continuous = self.continuous_flow.sample(
                c=condition,
                channel=channel,
                return_log_prob=True,
                device=device,
                dtype=dtype,
            )
            x = torch.cat((x_discrete, x_continuous), dim=1)
        else:
            x_continuous, log_prob_continuous = self.continuous_flow.sample(
                n=n,
                c=c,
                channel=channel,
                return_log_prob=True,
                device=device,
                dtype=dtype,
            )
            condition = (
                x_continuous if c is None else torch.cat((c, x_continuous), dim=1)
            )
            x_discrete, prob_discrete = self.discrete_flow.sample(
                c=condition,
                channel=channel,
                return_prob=True,
                device=device,
                dtype=dtype,
            )
            x = torch.cat((x_continuous, x_discrete), dim=1)

        extra_returns = []
        if return_log_prob:
            extra_returns.append(prob_discrete.log() + log_prob_continuous)
        if return_prob:
            extra_returns.append(prob_discrete * log_prob_continuous.exp())
        if len(extra_returns) > 0:
            return x, *extra_returns
        else:
            return x

    def init_with_grid(self, grid: torch.Tensor):
        """
        Initializes the flow using a VEGAS grid, i.e. from bins with varying width and equal
        probability. It splits the grid dimensions into discrete and continuous features and
        then calls the ``init_with_grid`` methods of the discrete and continuous flow classes.
        """
        if self.discrete_dims_first:
            grid_discrete = grid[:, : self.dims_in_discrete]
            grid_continuous = grid[:, self.dims_in_discrete :]
        else:
            grid_discrete = grid[:, self.dims_in_continuous :]
            grid_continuous = grid[:, : self.dims_in_continuous]
        self.discrete_flow.init_with_grid(grid_discrete)
        self.continuous_flow.init_with_grid(grid_continuous)
