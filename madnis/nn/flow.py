import math
from abc import ABC, abstractmethod
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn as nn

from .mlp import MLP, StackedMLP
from .splines import unconstrained_rational_quadratic_spline

L2PI = -0.5 * math.log(2 * math.pi)

Mapping = Callable[[torch.Tensor, bool], tuple[torch.Tensor, torch.Tensor]]


class Distribution(ABC):
    """
    Abstract base class for a (potentially learnable) distribution that can be used for sampling and
    density estimation, like a normalizing flow.
    """

    @abstractmethod
    def sample(
        self,
        n: int,
        channel: torch.Tensor | list[int] | int | None = None,
        return_log_prob: bool = False,
        return_prob: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Draws samples following the distribution

        Args:
            n: number of samples
            channel: encodes the channel of the samples. It must have one of the following types:

                - ``Tensor``: integer tensor of shape (n, ), containing the channel index for every
                  input sample;
                - ``list``: list of integers, specifying the number of samples in each channel;
                - ``int``: integer specifying a single channel containing all the samples;
                - ``None``: used in the single-channel case or to indicate that all channels contain
                  the same number of samples in the multi-channel case.
            return_log_prob: if True, also return the log-probabilities
            return_prob: if True, also return the probabilities
            device: device of the returned tensor. Only required if no condition is given.
            dtype: dtype of the returned tensor. Only required if no condition is given.
        Returns:
            samples with shape (n, dims_in). Depending on the arguments ``return_log_prob``,
            ``return_prob``, this function should also return the log-probabilities with shape (n, ),
            the probabilities with shape (n, ).
        """
        pass

    @abstractmethod
    def log_prob(
        self,
        x: torch.Tensor,
        channel: torch.Tensor | list[int] | int | None = None,
    ) -> torch.Tensor:
        """
        Computes the log-probabilities of the input data.

        Args:
            x: input data, shape (n, dims_in)
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
        pass

    @abstractmethod
    def prob(
        self,
        x: torch.Tensor,
        channel: torch.Tensor | list[int] | int | None = None,
        return_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the probabilities of the input data.

        Args:
            x: input data, shape (n, dims_in)
            channel: encodes the channel of the samples. It must have one of the following types:

                - ``Tensor``: integer tensor of shape (n, ), containing the channel index for every
                  input sample;
                - ``list``: list of integers, specifying the number of samples in each channel;
                - ``int``: integer specifying a single channel containing all the samples;
                - ``None``: used in the single-channel case or to indicate that all channels contain
                  the same number of samples in the multi-channel case.
        Returns:
            probabilities with shape (n, )
        """
        pass


class Flow(nn.Module, Distribution):
    """
    Coupling-block based normalizing flow (1605.08803) using rational quadratic spline
    transformations (1906.04032). Both conditional and non-conditional flows are supported. The
    class also allows to build multi-channel flows, i.e. an efficient implementation of multiple
    independent flows with the same hyperparameters.
    """

    def __init__(
        self,
        dims_in: int,
        dims_c: int = 0,
        uniform_latent: bool = True,
        permutations: Literal["log", "random", "exchange"] = "log",
        condition_masks: torch.Tensor | None = None,
        blocks: int | None = None,
        subnet_constructor: Callable[[int, int], nn.Module] | None = None,
        layers: int = 3,
        units: int = 32,
        activation: Callable[[], nn.Module] = nn.ReLU,
        layer_constructor: Callable[[int, int], nn.Module] = nn.Linear,
        channels: int | None = None,
        mapping: Mapping | list[Mapping] | None = None,
        bins: int = 10,
        spline_bounds: float = 10.0,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_bin_derivative: float = 1e-3,
    ):
        """
        Args:
            dims_in: input dimension
            dims_c: condition dimension
            uniform_latent: If True, encode mapping from [0,1]^d to [0,1]^d and use a uniform
                latent space distribution. If False, encode mapping from R^d to R^d and use
                Gaussian latent space distribution.
            permutations: Defines the strategy to permute the input dimensions between coupling
                blocks. "log": logarithmic decomposition, so that every dimension is conditioned on
                every other dimension at least once. "random": randomly permute dimensions.
                "exchange": condition the first half of the input on the second half, then the
                other way around, repeatedly.
            condition_masks: Overwrites the permutation strategy with a custom conditioning mask
                with shape (blocks, dims_in). Components where the mask is True are used as
                condition, and components where it is False are transformed.
            blocks: number of coupling blocks. Only needed if permutations is "random" or
                "exchange"
            subnet_constructor: function used to construct the flow sub-networks, with the
                number of input features and output features of the subnet as arguments. If None,
                the MLP (single channel) or StackedMLP (multi-channel) classes are used.
            layers: number of subnet layers. Only relevant if subnet_constructor=None.
            units: number of subnet hidden nodes. Only relevant if subnet_constructor=None.
            activation: function that builds a nn.Module used as activation function. Only
                relevant if subnet_constructor=None.
            layer_constructor: function used to construct the subnet layers, given the number of
                input and output features. Only relevant if subnet_constructor=None.
            channels: If None, build single-channel flow. If integer, build multi-channel flow
                with this number of channels.
            mapping: Specifies a single mapping function or a list of mapping functions (one per
                channel) that are applied to the input before it enters the flow (forward
                direction) or after drawing samples using the flow (inverse direction). The
                arguments of the function are the input data and a boolean whether the
                transformation is inverted. It must return the transformed value and the logarithm
                of the Jacobian determinant of the transformation.
            bins: number of RQ spline bins
            spline_bounds: If uniform_latent=False, the splines are defined on the interval
                [-spline_bounds, spline_bounds].
            min_bin_width: minimal width of the spline bins
            min_bin_height: minimal height of the spline bins
            min_bin_derivative: minimal derivative at the spline bin edges
        """
        super().__init__()

        self.dims_in = dims_in
        self.dims_c = dims_c
        self.channels = channels
        self.bins = bins
        self.uniform_latent = uniform_latent
        self.mapping = mapping
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_bin_derivative = min_bin_derivative

        if uniform_latent:
            self.spline_low = 0
            self.spline_high = 1
        else:
            self.spline_low = -spline_bounds
            self.spline_high = spline_bounds

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
                raise ValueError(f"Unknown permutation type {permutations}")
        self.condition_masks = condition_masks

        self.subnets = nn.ModuleList()
        for mask in self.condition_masks:
            dims_cond = torch.count_nonzero(mask)
            self.subnets.append(
                subnet_constructor(
                    dims_cond + dims_c, (dims_in - dims_cond) * (3 * bins + 1)
                )
            )

    def _apply_mappings(
        self, x: torch.Tensor, inverse: bool, channel: list[int] | int | None
    ) -> tuple[torch.Tensor, torch.Tensor | float]:
        """
        Applies the single mapping or channel-wise mappings to the input data

        Args:
            x: input data, shape (n, dims_in)
            inverse: if True, use inverted mapping (i.e. the sampling direction)
            channel: list of number of samples per channel, channel index, or None for no channels
                or equally sized channels
        Returns:
            tuple containing the transformed values with shape (n, dims_in), and log Jacobian
            determinants with shape (n, ) of the mapping
        """
        if self.mapping is None:
            return x, 0.0

        if isinstance(self.mapping, list):
            if isinstance(channel, int):
                mapping = self.mapping[channel]
                return mapping(x, inverse)
            else:
                map_x = []
                map_jac = []
                for xc, mapping in zip(x.split(channel, dim=0), self.mapping):
                    xm, jm = mapping(xc, inverse)
                    map_x.append(xm)
                    map_jac.append(jm)

    def transform(
        self,
        x: torch.Tensor,
        channel: torch.Tensor | list[int] | int | None = None,
        inverse: bool = False,
        c: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Transforms the input data into the latent space or back.

        Args:
            x: input data, shape (n, dims_in)
            channel: encodes the channel of the samples. It must have one of the following types:

                - ``Tensor``: integer tensor of shape (n, ), containing the channel index for every
                  input sample;
                - ``list``: list of integers, specifying the number of samples in each channel;
                - ``int``: integer specifying a single channel containing all the samples;
                - ``None``: used in the single-channel case or to indicate that all channels contain
                  the same number of samples in the multi-channel case.
            inverse: if True, use inverted transformation (i.e. the sampling direction)
            c: condition, shape (n, dims_c) or None for an unconditional flow
        Returns:
            tuple containing the transformed values with shape (n, dims_in), and log Jacobian
            determinants with shape (n, ) of the transformation
        """
        if isinstance(channel, torch.Tensor):
            channel_perm = torch.argsort(channel)
            x = x[channel_perm]
            c = None if c is None else c[channel_perm]
            channel = channel.bincount(minlength=self.channels).tolist()
        else:
            x = x.clone()
            channel_perm = None

        if inverse:
            jac = 0.0
        else:
            x, jac = self._apply_mappings(x, True, channel)

        if self.channels is None:
            channel_args = ()
        else:
            channel_args = (channel,)

        batch_size = x.shape[0]
        if inverse:
            blocks = zip(reversed(self.condition_masks), reversed(self.subnets))
        else:
            blocks = zip(self.condition_masks, self.subnets)
        for mask, subnet in blocks:
            inv_mask = ~mask
            x_trafo = x[:, inv_mask]
            x_cond = x[:, mask]
            if c is not None:
                x_cond = torch.cat((x_cond, c), dim=1)
            subnet_out = subnet(x_cond, *channel_args).reshape(
                (batch_size, x_trafo.shape[1], -1)
            )
            x_out, block_jac = unconstrained_rational_quadratic_spline(
                x_trafo,
                subnet_out[:, :, : self.bins],
                subnet_out[:, :, self.bins : 2 * self.bins],
                subnet_out[:, :, 2 * self.bins :],
                inverse,
                self.spline_low,
                self.spline_high,
                self.spline_low,
                self.spline_high,
                self.min_bin_width,
                self.min_bin_height,
                self.min_bin_derivative,
            )
            x[:, inv_mask] = x_out
            jac += block_jac.sum(dim=1)

        if inverse:
            x, map_jac = self._apply_mappings(x, False, channel)
            jac += map_jac

        if channel_perm is not None:
            channel_perm_inv = torch.argsort(channel_perm)
            x = x[channel_perm_inv]
            jac = jac[channel_perm_inv]

        return x, jac

    def _latent_log_prob(self, z: torch.Tensor):
        """
        Computes the log-probability of a vector in latent space.

        Args:
            z: latent space vector, shape (n, dims_in)
        Returns:
            log-probabilities, shape (n, )
        """
        if self.uniform_latent:
            return 0.0
        else:
            return z.shape[1] * L2PI - z.square().sum(dim=1) / 2

    def log_prob(
        self,
        x: torch.Tensor,
        channel: torch.Tensor | list[int] | int | None = None,
        c: torch.Tensor | None = None,
        return_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the log-probabilities of the input data.

        Args:
            x: input data, shape (n, dims_in)
            channel: encodes the channel of the samples. It must have one of the following types:

                - ``Tensor``: integer tensor of shape (n, ), containing the channel index for every
                  input sample;
                - ``list``: list of integers, specifying the number of samples in each channel;
                - ``int``: integer specifying a single channel containing all the samples;
                - ``None``: used in the single-channel case or to indicate that all channels contain
                  the same number of samples in the multi-channel case.
            c: condition, shape (n, dims_c) or None for an unconditional flow
            return_latent: if True, also return the latent space vector
        Returns:
            log-probabilities with shape (n, ). If ``return_latent`` is True, it also returns the
            latent space vector with shape (n, dims_in).
        """
        z, jac = self.transform(x, channel, False, c)
        log_prob = self._latent_log_prob(z) + jac
        if return_latent:
            return log_prob, z
        else:
            return log_prob

    def prob(
        self,
        x: torch.Tensor,
        channel: torch.Tensor | list[int] | int | None = None,
        c: torch.Tensor | None = None,
        return_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the probabilities of the input data.

        Args:
            x: input data, shape (n, dims_in)
            channel: encodes the channel of the samples. It must have one of the following types:

                - ``Tensor``: integer tensor of shape (n, ), containing the channel index for every
                  input sample;
                - ``list``: list of integers, specifying the number of samples in each channel;
                - ``int``: integer specifying a single channel containing all the samples;
                - ``None``: used in the single-channel case or to indicate that all channels contain
                  the same number of samples in the multi-channel case.
            c: condition, shape (n, dims_c) or None for an unconditional flow
            return_latent: if True, also return the latent space vector
        Returns:
            probabilities with shape (n, ). If ``return_latent`` is True, it also returns the
            latent space vector with shape (n, dims_in).
        """
        log_prob, z = self.log_prob(x, channel, c, True)
        if return_latent:
            return log_prob.exp(), z
        else:
            return log_prob.exp()

    def sample(
        self,
        n: int | None = None,
        channel: torch.Tensor | list[int] | int | None = None,
        return_log_prob: bool = False,
        return_prob: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        c: torch.Tensor | None = None,
        return_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Draws samples from the probability distribution encoded by the flow.

        Args:
            n: number of samples. Only required if no condition is given.
            channel: encodes the channel of the samples. It must have one of the following types:

                - ``Tensor``: integer tensor of shape (n, ), containing the channel index for every
                  input sample;
                - ``list``: list of integers, specifying the number of samples in each channel;
                - ``int``: integer specifying a single channel containing all the samples;
                - ``None``: used in the single-channel case or to indicate that all channels contain
                  the same number of samples in the multi-channel case.
            return_log_prob: if True, also return the log-probabilities
            return_prob: if True, also return the probabilities
            device: device of the returned tensor. Only required if no condition is given.
            dtype: dtype of the returned tensor. Only required if no condition is given.
            c: condition, shape (n, dims_c) or None for an unconditional flow
            return_latent: if True, also return the latent space vector
        Returns:
            samples with shape (n, dims_in). Depending on the arguments ``return_log_prob``,
            ``return_prob`` and ``return_latent``, this function will also return the log-probabilities
            with shape (n, ), the probabilities with shape (n, ) and the latent space vector with
            shape (n, dims_in).
        """
        if n is None:
            n = len(c)
        if c is None:
            options = {}
            if device is not None:
                options["device"] = device
            if dtype is not None:
                options["dtype"] = dtype
        else:
            options = {"device": c.device, "dtype": c.dtype}

        if self.uniform_latent:
            z = torch.rand((n, self.dims_in), **options)
        else:
            z = torch.randn((n, self.dims_in), **options)
        x, jac = self.transform(z, channel, True, c)
        if return_log_prob or return_prob:
            log_prob_latent = self._latent_log_prob(z)
            log_prob = log_prob_latent - jac

        extra_returns = []
        if return_log_prob:
            extra_returns.append(log_prob)
        if return_prob:
            extra_returns.append(log_prob.exp())
        if return_latent:
            extra_returns.append(z)
        if len(extra_returns) > 0:
            return x, *extra_returns
        else:
            return x

    def init_with_grid(self, grid: torch.Tensor):
        """
        Initializes the flow using a VEGAS grid, i.e. from bins with varying width and equal
        probability. The number of bins of this grid should be larger than the number of RQ spline
        bins. This function then performs the bin reduction algorithm as described in [2311.01548].

        Args:
            grid: edges of the VEGAS grid bins with shape (dims_in, vegas_bins+1) for single-channel
                flows or (channels, dims_in, vegas_bins+1) for multi-channel flows
        """
        # Initialize width, heights and derivatives from VEGAS grid
        w = grid.diff(dim=-1)
        h = grid.new_full(w.shape, 1 / w.shape[-1])
        d = (
            torch.cat((w[..., :1], (w[..., :-1] + w[..., 1:]) / 2, w[..., -1:]), dim=-1)
            * w.shape[-1]
        )

        # Run bin reduction algorithm
        while w.shape[-1] > self.bins:
            mask = (w[..., :-1] + w[..., 1:] < 2 / self.bins) & (
                h[..., :-1] + h[..., 1:] < 2 / self.bins
            )
            mask = mask | ~torch.any(mask, dim=-1, keepdim=True)
            diff = torch.abs(
                w[..., 1:] / h[..., 1:] - w[..., :-1] / h[..., :-1],
            ) + 1e9 * (~mask)
            mindiff = torch.argmin(diff, dim=-1, keepdim=True)
            delete_mask = torch.ones(
                w.shape, dtype=torch.bool, device=w.device
            ).scatter_(-1, mindiff, 0)
            next_shape = (*w.shape[:-1], w.shape[-1] - 1)
            w_col = w.gather(-1, mindiff)
            h_col = h.gather(-1, mindiff)
            w = w[delete_mask].reshape(next_shape)
            w.scatter_add_(-1, mindiff, w_col)
            h = h[delete_mask].reshape(next_shape)
            h.scatter_add_(-1, mindiff, h_col)
            d = torch.cat(
                (d[..., :1], d[..., 1:][delete_mask].reshape(next_shape)), dim=-1
            )

        # Invert softmax and softplus functions applied to subnet outputs
        w_unnorm = torch.log(
            (w - self.min_bin_width) / (1 - self.bins * self.min_bin_width)
        )
        w_unnorm -= torch.mean(w_unnorm, dim=-1, keepdim=True)
        h_unnorm = torch.log(
            (h - self.min_bin_height) / (1 - self.bins * self.min_bin_height)
        )
        h_unnorm -= torch.mean(h_unnorm, dim=-1, keepdim=True)
        d_unnorm = torch.log(
            torch.clip(
                torch.exp(
                    (self.min_bin_derivative + math.log(2)) / d
                    - self.min_bin_derivative
                )
                - 1,
                min=self.min_bin_derivative * 1e-5,
            )
        )
        rqs_grid = torch.cat((w_unnorm, h_unnorm, d_unnorm), dim=-1)

        # Initialize weights and biases of the last layer of the subnets of the first
        # two coupling blocks
        if len(grid.shape) == 3:
            weights_0 = self.subnets[0].weights[-1]
            weights_1 = self.subnets[1].weights[-1]
            biases_0 = self.subnets[0].biases[-1]
            biases_1 = self.subnets[1].biases[-1]
        else:
            weights_0 = self.subnets[0].layers[-1].weight
            weights_1 = self.subnets[1].layers[-1].weight
            biases_0 = self.subnets[0].layers[-1].bias
            biases_1 = self.subnets[1].layers[-1].bias
        nn.init.zeros_(weights_0)
        nn.init.zeros_(weights_1)
        with torch.no_grad():
            biases_0.copy_(
                rqs_grid[..., ~self.condition_masks[0], :].flatten(start_dim=-2)
            )
            biases_1.copy_(
                rqs_grid[..., ~self.condition_masks[1], :].flatten(start_dim=-2)
            )
