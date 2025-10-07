from typing import Any, Literal

import torch
import torch.nn as nn

from .discrete_made import DiscreteMADE
from .discrete_transformer import DiscreteTransformer
from .flow import Distribution, Flow


class MixedFlow(nn.Module, Distribution):
    def __init__(
        self,
        dims_in_continuous: int,
        dims_in_discrete: list[int],
        dims_c: int = 0,
        discrete_dims_position: Literal["first", "last"] = "first",
        discrete_model: Literal["made", "transformer"] = "made",
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

        if discrete_model == "made":
            self.discrete_flow = DiscreteMADE(
                dims_in=dims_in_discrete,
                dims_c=dims_c_discrete,
                channels=channels,
                **discrete_kwargs,
            )
        elif discrete_model == "transformer":
            self.discrete_flow = DiscreteTransformer(
                dims_in=dims_in_discrete,
                dims_c=dims_c_discrete,
                **discrete_kwargs,
            )
            if channels is not None:
                raise NotImplementedError(
                    "DiscreteTransformer only supported for single-channel integration"
                )
        else:
            raise ValueError("discrete_model must be 'made' or 'transformer'")

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
            x_discrete = x[:, : self.dims_in_discrete].long()
            log_prob_discrete, condition = self.discrete_flow.log_prob(
                x_discrete, c=c, channel=channel, return_one_hot=True
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
                x[:, self.dim_in_continuous].long(), c=condition, channel=channel
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
                return_one_hot=True,
            )
            x_continuous, log_prob_continuous = self.continuous_flow.sample(
                c=condition,
                channel=channel,
                return_log_prob=True,
                device=device,
                dtype=dtype,
            )
            x = torch.cat((x_discrete.to(x_continuous.dtype), x_continuous), dim=1)
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
            x = torch.cat((x_continuous.to(x_continuous.dtype), x_discrete), dim=1)

        extra_returns = []
        if return_log_prob:
            extra_returns.append(prob_discrete.log() + log_prob_continuous)
        if return_prob:
            extra_returns.append(prob_discrete * log_prob_continuous.exp())
        if len(extra_returns) > 0:
            return x, *extra_returns
        else:
            return x
