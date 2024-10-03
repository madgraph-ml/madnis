from typing import Any, Callable

import torch
import torch.nn as nn

from ..nn import Flow
from .integrand import Integrand
from .samples import SampleBatch, SampleBuffer


class Integrator(nn.Module):
    def __init__(
        self,
        integrand: Callable[[torch.Tensor], torch.Tensor] | Integrand,
        flow: Flow | None = None,
        flow_kwargs: dict[str, Any] = {},
        train_channel_weights: bool = False,
        cwnet: nn.Module | None = None,
        cwnet_kwargs: dict[str, Any] = {},
        loss: Callable = None,  # TODO
        optimizer: torch.optim.Optimizer | None = None,
        learning_rate: float = 1e-3,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        uniform_channel_ratio: float = 1.0,
        variance_history_length: int = 20,
        buffer_capacity: int = 0,
        buffered_batches: int = 0,
        max_stored_channel_weights: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Args:
            integrand: the function to be integrated. In the case of a simple single-channel
                integration, the integrand function can directly be passed to the integrator.
                In more complicated cases, like multi-channel integrals, use the `Integrand` class.
            flow: normalizing flow used for the integration. If None, a flow is constructed using
                the `Flow` class.
            flow_kwargs: If flow is None, these keyword arguments are passed to the `Flow`
                constructor.
            train_channel_weights: If True, construct a channel weight network and train it. Only
                necessary if cwnet is None.
            cwnet: network used for the trainable channel weights. If None and
                train_channel_weights is True, the cwnet is built using the `MLP` class.
            cwnet_kwargs: If cwnet is None and train_channel_weights is True, these keyword
                arguments are passed to the `MLP` constructor.
            loss: Loss function used for training.
            optimizer: optimizer for the training. If None, the Adam optimizer is used.
            learning_rate: learning rate used for the Adam optimizer
            scheduler: learning rate scheduler for the training. If None, a constant learning rate
                is used.
            uniform_channel_ratio: part of samples in each batch that will be distributed equally
                between all channels, value has to be between 0 and 1.
            variance_history_length: number of batches for which the channel-wise means and
                variances are stored. This is used for stratified sampling during integration, and
                during the training if uniform_channel_ratio is different from one.
            buffer_capacity: number of samples that are stored for buffered training
            buffered_batches: number of optimization steps on buffered samples after every online
                training step
            max_stored_channel_weights: number of prior channel weights that are buffered for each
                sample. If None, all prior channel weights are saved, otherwise only those for the
                channels with the largest contributions.
            device: torch device used for training and integration. If None, use default device.
            dtype: torch dtype used for training and integration. If None, use default dtype.
        """
        super().__init__()

        if device is not None:
            self.to(device)
        if dtype is not None:
            self.to(dtype)
