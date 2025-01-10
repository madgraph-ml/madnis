from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow import Distribution, Flow
from .masked_mlp import MaskedMLP

PriorProbFunction = (
    Callable[[torch.Tensor, int], torch.Tensor]
    | Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]]
)


class DiscreteFlow(nn.Module, Distribution):
    def __init__(
        self,
        dims_in: list[int],
        dims_c: int = 0,
        mlp_kwargs: dict = {},
        prior_prob_function: PriorProbFunction | None = None,
        prior_prob_mode: Literal["indices", "states"] = "indices",
        output_mode: Literal["int", "float"] = "int",
    ):
        """ """
        super().__init__()

        self.masked_net = MaskedMLP(
            input_dims=[dims_c, *discrete_dims[:-1]],
            output_dims=dims_in_discrete,
            **discrete_kwargs,
        )
        self.discrete_dims = discrete_dims
        self.max_dim = max(discrete_dims)
        self.prior_prob_function = prior_prob_function
        if prior_prob_mode == "indices":
            self.prior_uses_indices = True
        elif prior_prob_mode == "states":
            self.prior_uses_indices = False
        else:
            raise ValueError(f"Unknown prior probability mode '{prior_prob_mode}'")
        if output_mode == "int":
            self.register_buffer("output_scale", torch.tensor(discrete_dims))
        elif prior_prob_mode == "float":
            self.output_scale = None
        else:
            raise ValueError(f"Unknown output mode '{prior_prob_mode}'")

        discrete_indices = []
        one_hot_mask = []
        for i, dim in enumerate(discrete_dims):
            discrete_indices.extend([i] * dim)
            one_hot_mask.extend([True] * dim + [False] * (self.max_dim - dim))
        self.register_buffer("discrete_indices", torch.tensor(discrete_indices))
        self.register_buffer("one_hot_mask", torch.tensor(one_hot_mask))

    def one_hot(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_scale is not None:
            x = (x * self.output_scale).long()
        return (
            F.one_hot(x, self.max_dim)
            .to(x.dtype)
            .flatten(start_dim=1)[:, self.one_hot_mask]
        )

    def log_prob(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        channel: torch.Tensor | list[int] | int | None = None,
    ) -> torch.Tensor:
        """ """
        x_one_hot = self.one_hot(x)
        if c is None:
            input_disc = one_hot
        else:
            input_disc = torch.cat((c, one_hot), dim=1)

        prior_probs = []
        if self.prior_uses_indices:
            for i, _ in enumerate(self.discrete_dims):
                prior_probs.append(self.prior_prob_func(x[:, :i], i))
        else:
            # TODO: pass channel number here
            state = torch.zeros((len(x),), dtype=torch.int64, device=x.device)
            for i, _ in enumerate(self.discrete_dims):
                probs, new_states = self.prior_prob_func(state, i)
                prior_probs.append(probs)
                state = torch.gather(new_states, dim=1, index=x[:, i : i + 1])[:, 0]

        prior_probs = torch.cat(prior_probs, dim=1)

        unnorm_prob_disc = (
            self.masked_net(input_disc[:, : -self.discrete_dims[-1]]).exp()
            * discrete_probs
        )
        prob_norms = torch.zeros_like(indices, dtype=x.dtype).scatter_add_(
            1, self.discrete_indices[None, :].expand(x.shape[0], -1), unnorm_prob_disc
        )
        prob_sums = torch.zeros_like(prob_norms).scatter_add_(
            1,
            self.discrete_indices[None, :].expand(x.shape[0], -1),
            unnorm_prob_disc * x_discrete,
        )
        return torch.prod(prob_sums / prob_norms, dim=1).log()

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
        if n is None:
            n = len(c)
        if c is None:
            options = {"device": device, "dtype": dtype}
            x_in = torch.zeros((n, 0), **options)
        else:
            options = {"device": c.device, "dtype": c.dtype}
            x_in = c

        log_prob = torch.ones((n,), **options)
        net_cache = None
        if self.prior_uses_indices:
            # TODO: pass channel number here
            state = torch.zeros((n,), device=options["device"], dtype=torch.int64)

        x_out = torch.ones(
            (n, len(self.discrete_dims)), device=options["device"], dtype=torch.int64
        )
        for i, dim in enumerate(self.discrete_dims):
            y, net_cache = self.masked_net.forward_cached(x, i, net_cache)
            if self.prior_uses_indices:
                prior_probs = self.prior_prob_func(x[:, :i], i)
            else:
                prior_probs, state = self.prior_prob_func(state, i)
            unnorm_probs = y.exp() * pred_probs
            cdf = unnorm_probs.cumsum(dim=1)
            norm = cdf[:, -1]
            cdf = cdf / norm[:, None]
            r = torch.rand((y.shape[0], 1), **options)
            samples = torch.searchsorted(cdf, r)[:, 0]
            x_out[:, i] = samples
            prob = torch.gather(unnorm_probs, 1, samples[:, None])[:, 0] / norm * prob
            x = F.one_hot(samples, dim).to(y.dtype)

        if self.output_scale is not None:
            x_out = (x_out + 0.5) / self.output_scale
        extra_returns = []
        if return_log_prob:
            extra_returns.append(log_prob)
        if return_prob:
            extra_returns.append(log_prob.exp())
        if len(extra_returns) > 0:
            return x_out, *extra_returns
        else:
            return x_out

    def init_with_grid(self, grid: torch.Tensor):
        # TODO: implement VEGAS init
        pass


class MixedFlow(nn.Module, Distribution):
    def __init__(
        self,
        dims_in_continuous: int,
        dims_in_discrete: list[int],
        dims_c: int = 0,
        discrete_dims_first: bool = True,
        continuous_kwargs: dict = {},
        discrete_kwargs: dict = {},
    ):
        """ """
        super().__init__()
        self.discrete_dims_first = discrete_dims_first
        self.dims_in_continuous = dims_in_discrete
        self.dims_in_discrete = len(dims_in_discrete)
        if discrete_dims_first:
            dims_c_discrete = dims_c
            dims_c_continuous = sum(dims_in_discrete) + dims_c
        else:
            dims_c_discrete = dims_c + dims_in_continuous
            dims_c_continuous = dims_c
        self.discrete_flow = DiscreteFlow(
            dims_in=dims_in_discrete, dims_c=dims_c_discrete, **discrete_kwargs
        )
        self.continuous_flow = Flow(
            dims_in=dims_in_continuous, dims_c=dims_c_continuous, **continuous_kwargs
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
            log_prob_discrete = self.discrete_flow.log_prob(
                x_discrete, c=input_disc, channel=channel
            )
            x_discrete_one_hot = self.discrete_flow.one_hot(x_discrete)
            condition = (
                x_discrete_one_hot
                if c is None
                else torch.cat((c, x_discrete_one_hot), dim=1)
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
            x_discrete, log_prob_discrete = self.flow_discrete.sample(
                n=n,
                c=c,
                channel=channel,
                return_log_prob=True,
                device=device,
                dtype=dtype,
            )
            x_discrete_one_hot = self.discrete_flow.one_hot(x_discrete)
            condition = (
                x_discrete_one_hot
                if c is None
                else torch.cat((c, x_discrete_one_hot), dim=1)
            )
            x_continuous, log_prob_continuous = self.continuous_flow.log_prob(
                c=condition,
                channel=channel,
                return_log_prob=True,
                device=device,
                dtype=dtype,
            )
            x = torch.cat((x_discrete, x_continuous), dim=1)
        else:
            x_continuous, log_prob_continuous = self.continuous_flow.log_prob(
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
            x_discrete, log_prob_discrete = self.flow_discrete.sample(
                c=condition,
                channel=channel,
                return_log_prob=True,
                device=device,
                dtype=dtype,
            )
            x = torch.cat((x_continuous, x_discrete), dim=1)

        log_prob = log_prob_discrete + log_prob_continuous
        extra_returns = []
        if return_log_prob:
            extra_returns.append(log_prob)
        if return_prob:
            extra_returns.append(log_prob.exp())
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
