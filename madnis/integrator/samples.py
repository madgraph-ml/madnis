from dataclasses import astuple, dataclass
from typing import Callable

import torch
import torch.nn as nn


@dataclass
class SampleBatch:
    """
    Class to store a batch of samples

    Args:
        x:
            Samples generated either uniformly or by the flow with shape (nsamples, ndim)
        y:
            Samples after transformation through analytic mappings or from the call to
            the integrand (if integrand_has_channels is True) with shape (nsamples, nfeatures)
        q_sample:
            Test probability of all mappings (analytic + flow) with shape (nsamples,)
        func_vals:
            True probability with shape (nsamples,)
        channels:
            Tensor encoding which channel to use with shape (nsamples,)
        alphas_prior:
            Prior for the channel weights with shape (nsamples, nchannels)
        z:
            Random number/latent space point used to generate the sample, shape (nsamples, ndim)
    """

    x: torch.Tensor
    y: torch.Tensor
    q_sample: torch.Tensor
    func_vals: torch.Tensor
    channels: torch.Tensor
    alphas_prior: torch.Tensor | None = None
    z: torch.Tensor | None = None
    alpha_channel_indices: torch.Tensor | None = None

    def __iter__(self):
        return iter(astuple(self))

    def map(self, func: Callable[[torch.Tensor], torch.Tensor]):
        return SampleBatch(*(None if field is None else func(field) for field in self))

    @staticmethod
    def cat(batches: list["SampleBatch"]):
        return SampleBatch(
            *(None if item[0] is None else torch.cat(item, dim=0) for item in zip(*batches))
        )


class SampleBuffer(nn.Module):
    pass
