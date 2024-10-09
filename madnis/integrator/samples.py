from collections.abc import Iterable
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
    def cat(batches: Iterable["SampleBatch"]):
        return SampleBatch(
            *(
                None if item[0] is None else torch.cat(item, dim=0)
                for item in zip(*batches)
            )
        )


class Buffer(nn.Module):
    def __init__(self, capacity: int, shapes: list[tuple[int, ...]]):
        super().__init__()
        self.keys = []
        for i, shape in enumerate(shapes):
            key = f"buffer{i}"
            self.register_buffer(
                key, None if shape is None else torch.zeros(capacity, *shape)
            )
            self.keys.append(key)
        self.capacity = capacity
        self.size = 0
        self.store_index = 0

    def _batch_slices(self, batch_size: int):
        start = self.store_index
        while start < self.size:
            stop = min(start + batch_size, self.size)
            yield slice(start, stop)
            start = stop
        start = 0
        while start < self.store_index:
            stop = min(start + batch_size, self.store_index)
            yield slice(start, stop)
            start = stop

    def _buffer_fields(self):
        for key in self.keys:
            yield getattr(self, key)

    def __iter__(self):
        for key in self.keys:
            yield getattr(self, key)[: self.size]

    def store(self, *tensors: torch.Tensor | None):
        store_slice1 = None
        for buffer, data in zip(self._buffer_fields(), tensors):
            if data is None:
                continue
            if store_slice1 is None:
                size = min(data.shape[0], self.capacity)
                end_index = self.store_index + size
                if end_index < self.capacity:
                    store_slice1 = slice(self.store_index, end_index)
                    store_slice2 = slice(0, 0)
                    load_slice1 = slice(0, size)
                    load_slice2 = slice(0, 0)
                else:
                    store_slice1 = slice(self.store_index, self.capacity)
                    store_slice2 = slice(0, end_index - self.capacity)
                    load_slice1 = slice(0, self.capacity - self.store_index)
                    load_slice2 = slice(self.capacity - self.store_index, size)
                self.store_index = end_index % self.capacity
                self.size = min(self.size + size, self.capacity)
            buffer[store_slice1] = data[load_slice1]
            buffer[store_slice2] = data[load_slice2]

    def filter(
        self,
        predicate: Callable[[tuple[torch.Tensor | None, ...]], torch.Tensor],
        batch_size: int = 100000,
    ):
        masks = []
        masked_size = 0
        for batch_slice in self._batch_slices(batch_size):
            mask = predicate(
                tuple(
                    None if t is None else t[batch_slice] for t in self._buffer_fields()
                )
            )
            masked_size += torch.count_nonzero(mask)
            masks.append(mask)
        for buffer in self._buffer_fields():
            buffer[:masked_size] = torch.cat(
                [
                    buffer[batch_slice][mask]
                    for batch_slice, mask in zip(self._batch_slices(batch_size), masks)
                ],
                dim=0,
            )
        self.size = masked_size
        self.store_index = masked_size % self.capacity

    def sample(self, count: int):
        weights = next(b for b in self._buffer_fields() if b is not None).new_ones(
            self.size
        )
        indices = torch.multinomial(weights, min(count, self.size), replacement=False)
        return [buffer[indices] for buffer in self._buffer_fields()]
