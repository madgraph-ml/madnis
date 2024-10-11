import itertools
import signal
import warnings
from collections.abc import Iterable
from dataclasses import astuple, dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn import MLP, Flow
from .buffer import Buffer
from .integrand import Integrand


@dataclass
class TrainingStatus:
    """
    Contains the MadNIS training status to pass it to a callback function.

    Args:
        step: optimization step
        online_loss: loss from the optimization step on new samples
        buffered_loss: average loss from the optimization steps on buffered samples
        learning_rate: current learning rate if learning rate scheduler is present
        dropped_channels: number of channels dropped after this optimization step
    """

    step: int
    online_loss: float
    buffered_loss: float | None
    learning_rate: float | None
    dropped_channels: int


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

    def split(self, batch_size: int) -> SampleBatch:
        for batch in zip(
            itertools.repeat(None) if field is None else field.split(batch_size)
            for field in self
        ):
            yield SampleBatch(*batch)

    @staticmethod
    def cat(batches: Iterable["SampleBatch"]):
        return SampleBatch(
            *(
                None if item[0] is None else torch.cat(item, dim=0)
                for item in zip(*batches)
            )
        )


class Integrator(nn.Module):
    """
    Implements MadNIS training and integration logic. MadNIS integrators are torch modules, so
    their state can easily be saved and loaded using the torch.save and torch.load methods.
    """

    def __init__(
        self,
        integrand: Callable[[torch.Tensor], torch.Tensor] | Integrand,
        dims: int = 0,
        flow: Flow | None = None,
        flow_kwargs: dict[str, Any] = {},
        train_channel_weights: bool = False,
        cwnet: nn.Module | None = None,
        cwnet_kwargs: dict[str, Any] = {},
        loss: Callable = None,  # TODO
        optimizer: torch.optim.Optimizer | None = None,
        batch_size: int = 1024,
        learning_rate: float = 1e-3,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        uniform_channel_ratio: float = 1.0,
        variance_history_length: int = 20,
        drop_zero_integrands: bool = False,
        batch_size_threshold: float = 0.5,
        buffer_capacity: int = 0,
        minimum_buffer_size: int = 50,
        buffered_steps: int = 0,
        max_stored_channel_weights: int | None = None,
        channel_dropping_threshold: float = 0.0,
        channel_dropping_interval: int = 100,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Args:
            integrand: the function to be integrated. In the case of a simple single-channel
                integration, the integrand function can directly be passed to the integrator.
                In more complicated cases, like multi-channel integrals, use the `Integrand` class.
            dims: dimension of the integration space. Only required if a simple function is given
                as integrand.
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
            batch_size: Training batch size
            learning_rate: learning rate used for the Adam optimizer
            scheduler: learning rate scheduler for the training. If None, a constant learning rate
                is used.
            uniform_channel_ratio: part of samples in each batch that will be distributed equally
                between all channels, value has to be between 0 and 1.
            variance_history_length: number of batches for which the channel-wise means and
                variances are stored. This is used for stratified sampling during integration, and
                during the training if uniform_channel_ratio is different from one.
            drop_zero_integrands: If True, points with integrand zero are dropped and not used for
                the optimization.
            batch_size_threshold: If drop_zero_integrands is True, new samples are drawn until the
                number of samples is at least batch_size_threshold * batch_size.
            buffer_capacity: number of samples that are stored for buffered training
            minimum_buffer_size: minimal size of the buffer to run buffered training
            buffered_steps: number of optimization steps on buffered samples after every online
                training step
            max_stored_channel_weights: number of prior channel weights that are buffered for each
                sample. If None, all prior channel weights are saved, otherwise only those for the
                channels with the largest contributions.
            channel_dropping_threshold: all channels which a cumulated contribution to the
                integrand that is smaller than this threshold are dropped
            channel_dropping_interval: number of training steps after which channel dropping
                is performed
            device: torch device used for training and integration. If None, use default device.
            dtype: torch dtype used for training and integration. If None, use default dtype.
        """
        super().__init__()

        if not isinstance(integrand, Integrand):
            integrand = Integrand(integrand, dims)
        if flow is None:
            flow = Flow(
                dims_in=integrand.y_dim, channels=integrand.channels, **flow_kwargs
            )
        if cwnet is None and train_channel_weights:
            cwnet = MLP(integrand.x_dim, integrand.channels, **cwnet_kwargs)
        parameters = [*flow.parameters()]
        if cwnet is not None:
            parameters.extend(cwnet.parameters())
        if optimizer is None:
            optimizer = torch.optim.Adam(parameters, learning_rate)

        self.integrand = integrand
        self.multichannel = integrand.channels is not None
        self.flow = flow
        self.cwnet = cwnet
        self.optimizer = optimizer
        self.loss = loss
        self.scheduler = scheduler
        self.uniform_channel_ratio = uniform_channel_ratio
        self.drop_zero_integrands = drop_zero_integrands
        self.batch_size_threshold = batch_size_threshold
        if buffer_capacity > 0:
            self.buffer = Buffer(buffer_capacity, ..., persistent=False)
        self.minimum_buffer_size = minimum_buffer_size
        self.buffered_steps = buffered_steps
        self.max_stored_channel_weights = (
            None
            if max_stored_channel_weights is None
            or max_stored_channel_weights >= integrand.channels
            else max_stored_channel_weights
        )
        self.channel_dropping_threshold = channel_dropping_threshold
        self.channel_dropping_interval = channel_dropping_interval
        if self.multichannel:
            hist_shape = (self.integrand.channels,)
            self.variance_history = Buffer(
                variance_history_length, [hist_shape, hist_shape, hist_shape]
            )
        else:
            self.variance_history = None
        self.step = 0

        if device is not None:
            self.to(device)
        if dtype is not None:
            self.to(dtype)

    def _get_alphas(self, samples: SampleBatch) -> torch.Tensor:
        """
        Runs the channel weight network and returns the normalized channel weights, taking prior
        channel weights and dropped channels into account.

        Args:
            samples: batch of samples
        Returns:
            channel weights, shape (n, channels)
        """
        if samples.alphas_prior is None:
            log_alpha_prior = samples.x.new_zeros(
                (samples.x.shape[0], self.integrand.channels)
            )
        else:
            log_alpha_prior = self._restore_prior(samples).log()
        log_alpha = log_alpha_prior + self.cwnet(samples.y)
        alpha = torch.zeros_like(log_alpha)
        alpha[:, self.active_channel_mask] = F.softmax(
            log_alpha[:, self.active_channel_mask], dim=1, keepdim=True
        )
        return alpha

    def _optimization_step(
        self,
        samples: SampleBatch,
    ) -> tuple[float, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Perform one optimization step of the networks for the given samples

        Args:
            samples: batch of samples
        Returns:
            A tuple containing

              - value of the loss
              - channel-wise means of the integral, shape (channels, )
              - channel-wise variances of the integral, shape (channels, )
              - channel-wise number of samples, shape (channels, )

            The latter three values are None in the single-channel case
        """
        self.optimizer.zero_grad()
        q_test = self.flow.prob(samples.x, channel=samples.channels)

        if self.multichannel:
            alphas = torch.gather(
                self._get_alphas(samples), index=samples.channels[:, None], dim=1
            )[:, 0]
            f_all = (
                alphas * samples.func_vals.abs()
            )  # abs needed for non-positive integrands
            f_div_q = f_all.detach() / samples.q_sample.detach()
            counts = torch.bin_count(
                samples.channels, minlength=self.integrand.channels
            )
            means = torch.bin_count(
                samples.channels, weights=f_div_q, minlength=self.integrand.channels
            ) / torch.maximum(counts, 1)
            variances = (
                torch.bin_count(
                    samples.channels,
                    weight=(f_div_q - means[samples.channels]).square(),
                    minlength=self.integrand.channels,
                )
                / counts
            )
            f_true = f_all / means.sum()
        else:
            f_all = samples.func_vals.abs()
            f_div_q = f_all.detach() / samples.q_sample.detach()
            f_true = f_all / f_div_q.mean()
            means = None
            counts = None
            variances = None

        loss = self.loss_func(
            samples.channels, f_true, q_test, q_sample=samples.q_sample
        )
        if loss.isnan().item():
            warnings.warn("nan batch: skipping optimization")
        else:
            loss.backward()
            self.optimizer.step()

        return loss.item(), means, variances, counts

    def _restore_prior(self, samples: SampleBatch) -> torch.Tensor:
        """
        Restores the full prior channel weights if only the largest channel weights and their
        indices were saved.

        Args:
            samples: batch of samples
        Returns:
            Tensor of prior channel weights with shape (n, channels)
        """
        if samples.alpha_channel_indices is None:
            return samples.alphas_prior

        n_rest = self.integrand.channels - self.max_stored_channel_weights
        alphas_prior_reduced = samples.alphas_prior
        epsilon = torch.finfo(alphas_prior_reduced.dtype).eps

        # strategy 1: distribute difference to 1 evenly among non-stored channels
        # alphas_prior = torch.clamp(
        #    (1 - alphas_prior_reduced.sum(dim=1, keepdims=True)) / n_rest,
        #    min=epsilon,
        # ).repeat(1, self.integrand.channels)
        # alphas_prior.scatter_(1, samples.alpha_channel_indices, alphas_prior_reduced)
        # return alphas_prior

        # strategy 2: set non-stored channel alphas to epsilon, normalize again
        alphas_prior = alphas_prior_reduced.new_full(
            (alphas_prior_reduced.shape[0], self.integrand.channels), epsilon
        )
        alphas_prior.scatter_(1, samples.alpha_channel_indices, alphas_prior_reduced)
        return alphas_prior / alphas_prior.sum(dim=1, keepdims=True)

    def _get_variance_weights(self, expect_full_history=True) -> torch.Tensor:
        """
        Uses the list of saved variances to compute the contribution of each channel for
        stratified sampling.

        Args:
            expect_full_history: If True, the variance history has to be full, otherwise uniform
                weights are returned.
        Returns:
            Weights for sampling the channels with shape (channels,)
        """
        min_len = self.variance_history.capacity if expect_full_history else 1
        if self.variance_history.size < min_len:
            return torch.ones(self.n_channels)
        var_hist, _, count_hist = self.variance_history
        count_hist = torch.where(var_hist.isnan(), np.nan, count_hist)
        hist_weights = count_hist / count_hist.nansum(dim=0)
        return torch.nansum(hist_weights * var_hist, dim=0).sqrt()

    def _disable_unused_channels(self) -> int:
        """
        Determines channels with a total relative contribution below
        ``channel_dropping_threshold``, disables them and removes them from the buffer.

        Returns:
            Number of channels that were disabled
        """
        if self.channel_dropping_threshold == 0.0:
            return 0
        if (self.steps + 1) % self.channel_dropping_interval != 0:
            return 0

        _, mean_hist, count_hist = self.variance_history
        mean_hist = torch.nan_to_num(mean_hist)
        hist_weights = count_hist / count_hist.sum(dim=0)
        channel_integrals = torch.nansum(hist_weights * mean_hist, dim=0)
        channel_rel_integrals = channel_integrals / channel_integrals.sum()
        cri_sort, cri_argsort = torch.sort(channel_rel_integrals)
        n_irrelevant = torch.count_nonzero(
            cri_sort.cumsum(dim=0) < self.channel_dropping_threshold
        )
        n_disabled = torch.count_nonzero(
            self.active_channels_mask[cri_argsort[:n_irrelevant]]
        )
        self.active_channels_mask[cri_argsort[:n_irrelevant]] = False
        self.buffer.filter(
            lambda batch: self.active_channels_mask[SampleBatch(*batch).channels]
        )
        return n_disabled

    def _store_samples(self, samples: SampleBatch):
        """
        Stores the generated samples and probabilites for reuse during buffered training. If
        ``max_stored_channel_weights`` is set, the largest channel weights are determined and only
        those and their weights are stored.

        Args:
            samples: Object containing a batch of samples
        """
        if self.buffer is None:
            return

        if (
            self.max_stored_channel_weights is not None
            and self.integrand.channel_weight_prior
        ):
            # Hack to ensure that the alpha for the channel that the sample was generated with
            # is always stored
            alphas_prior_mod = torch.scatter(
                samples.alphas_prior,
                dim=1,
                index=samples.channels[:, None],
                src=torch.tensor([[2.0]]).expand(*samples.alphas_prior.shape),
            )
            largest_alphas, alpha_indices = torch.sort(
                alphas_prior_mod, descending=True, dim=1
            )
            largest_alphas[:, 0] = torch.gather(
                samples.alphas_prior, dim=1, index=samples.channels[:, None]
            )[:, 0]
            samples.alphas_prior = largest_alphas[
                :, : self.max_stored_channel_weights
            ].clone()
            samples.alpha_channel_indices = alpha_indices[
                :, : self.max_stored_channel_weights
            ].clone()

        self.buffer.store(*samples)

    def _get_channels(
        self,
        n: int,
        channel_weights: torch.Tensor,
        uniform_channel_ratio: float,
    ) -> torch.Tensor:
        """
        Create a tensor of channel indices in two steps:
        1. Split up n * uniform_channel_ratio equally among all the channels
        2. Sample the rest of the events from the distribution given by channel_weights
           after correcting for the uniformly distributed samples
        This allows stratified sampling by variance weighting while ensuring stable training
        because there are events in every channel.
        Args:
            n: Number of samples as scalar integer tensor
            channel_weights: Weights of the channels (not normalized) with shape (n,)
            uniform_channel_ratio: Number between 0.0 and 1.0 to determine the ratio of samples
                that will be distributed uniformly first
        Returns:
            Tensor of channel numbers with shape (n,)
        """
        assert channel_weights.shape == (self.integrand.channels,)
        n_active_channels = torch.count_nonzero(self.active_channels_mask)
        uniform_per_channel = int(
            np.ceil(n * uniform_channel_ratio / n_active_channels)
        )
        n_per_channel = channel_weights.new_full(
            (self.n_channels,), uniform_per_channel
        )
        n_per_channel[~self.active_channels_mask] = 0

        n_weighted = max(n - n_per_channel.sum(), 0)
        if n_weighted > 0:
            normed_weights = (
                channel_weights / channel_weights[self.active_channels_mask].sum()
            )
            normed_weights[~self.active_channels_mask] = 0.0
            probs = torch.clamp(
                normed_weights - uniform_channel_ratio / n_active_channels, min=0
            )
            n_per_channel += torch.ceil(probs * n_weighted / probs.sum()).int()

        remove_chan = 0
        while n_per_channel.sum() > n:
            if n_per_channel[remove_chan] > 0:
                n_per_channel[remove_chan] -= 1
            remove_chan = (remove_chan + 1) % self.n_channels
        assert n_per_channel.sum() == n

        return torch.cat(
            [channel_weights.new_full((npc,), i) for i, npc in enumerate(n_per_channel)]
        )

    def _get_samples(
        self, n: int, channels: torch.Tensor | None = None, train: bool = False
    ) -> SampleBatch:
        """
        Draws samples from the flow for the given channels and evaluates the integrand

        Args:
            n: number of samples
            channels: Tensor encoding which channel to use with shape (n,) or None for a
                single-channel integrand
            train: If True, the function is used in training mode, i.e. samples where the integrand
                is zero will be removed if drop_zero_integrands is True
        Returns:
            Object containing a batch of samples
        """
        batches_out = []
        x_mask_cache = []
        current_batch_size = 0

        while True:
            with torch.no_grad():
                x, prob = self.flow.sample(n, return_prob=True)

            weight, y, alphas_prior = self.integrand(x, channels)
            batch = SampleBatch(x, y, prob, weight, channels, alphas_prior)

            mask = True
            if train and self.drop_zero_integrands:
                mask = (weight != 0.0) & mask
            mask = ~(weight.isnan() | x.isnan().any(dim=1)) & mask
            if mask is not True:
                batch = batch.map(lambda t: t[mask])
            current_batch_size += batch.x.shape[0]
            batches_out.append(batch)
            if current_batch_size > self.batch_size_threshold * n:
                break

        return SampleBatch.cat(batches_out)

    def train_step(self) -> TrainingStatus:
        """
        Performs a single training step

        Returns:
            Training status
        """

        channels = self._get_channels(
            self.batch_size, self._get_variance_weights(), self.uniform_channel_ratio
        )
        samples, _ = self._get_samples(channels)
        online_loss, means, variances, counts = self._optimization_step(samples)
        self._store_samples(samples)
        self.variance_history.store(variances, counts, means)

        if self.buffered_steps != 0 and self.buffer.size > self.minimum_buffer_size:
            all_samples = SampleBatch(
                *self.buffer.sample(self.buffered_steps * self.batch_size)
            )
            buffered_loss = 0.0
            count = 0
            for samples in all_samples.split(self.batch_size):
                loss, _, _, _ = self._optimization_step(samples)
                buffered_loss += loss
                count += 1
            buffered_loss /= count
        else:
            buffered_loss = None

        dropped_channels = self._disable_unused_channels()
        status = TrainingStatus(
            step=self.step,
            online_loss=online_loss,
            buffered_loss=buffered_loss,
            learning_rate=(
                None if self.scheduler is None else self.scheduler.get_last_lr()[0]
            ),
            dropped_channels=dropped_channels,
        )

        if self.scheduler is not None:
            self.scheduler.step()
        self.step += 1
        return status

    def train(
        self,
        steps: int,
        callback: Callable[[TrainingStatus], None] | None = None,
        capture_keyboard_interrupt: bool = False,
    ):
        """
        Performs multiple training steps

        Args:
            steps: number of training steps
            callback: function that is called after each training step with the training status
                as argument
            capture_keyboard_interrupt: If True, a keyboard interrupt does not raise an exception.
                Instead, the current training step is finished and the training is aborted
                afterwards.
        """
        interrupted = False
        if capture_keyboard_interrupt:

            def handler(sig, frame):
                nonlocal interrupted
                interrupted = True

            old_handler = signal.signal(signal.SIGINT, handler)

        try:
            for _ in range(steps):
                status = self.train_step()
                if callback is not None:
                    callback(status)
                if interrupted:
                    break
        finally:
            if capture_keyboard_interrupt:
                signal.signal(signal.SIGINT, old_handler)
