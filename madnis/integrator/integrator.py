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
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..nn import MLP, Flow
from .buffer import Buffer
from .integrand import Integrand
from .losses import kl_divergence


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
    Contains a batch of samples

    Args:
        x: samples generated by the flow, shape (n, dim)
        y: remapped samples returned by the integrand, shape (n, remapped_dim)
        q_sample: probabilities of the samples, shape (n, )
        func_vals: integrand value, shape (n, )
        channels: channels indices for multi-channel integration, shape (n, ), otherwise None
        alphas_prior: prior channel weights, shape (n, channels), or None for single-channel
            integration
        alpha_channel_indices: channel indices if not all prior channel weights are stored,
            otherwise None
    """

    x: torch.Tensor
    y: torch.Tensor | None
    q_sample: torch.Tensor
    func_vals: torch.Tensor
    channels: torch.Tensor | None
    alphas_prior: torch.Tensor | None = None
    alpha_channel_indices: torch.Tensor | None = None

    def __iter__(self) -> Iterable[torch.Tensor | None]:
        """
        Returns iterator over the fields of the class
        """
        return iter(astuple(self))

    def map(self, func: Callable[[torch.Tensor], torch.Tensor]) -> "SampleBatch":
        """
        Applies function to all fields in the batch that are not None and returns a new SampleBatch

        Args:
            func: function that is applied to all fields in the batch. Expects a tensor as argument
                and returns a new tensor
        Returns:
            Transformed SampleBatch
        """
        return SampleBatch(*(None if field is None else func(field) for field in self))

    def split(self, batch_size: int) -> Iterable["SampleBatch"]:
        """
        Splits up the fields into batches and yields SampleBatch objects for every batch.

        Args:
            batch_size: maximal size of the batches
        Returns:
            Iterator over the batches
        """
        for batch in zip(
            *(
                itertools.repeat(None) if field is None else field.split(batch_size)
                for field in self
            )
        ):
            yield SampleBatch(*batch)

    @staticmethod
    def cat(batches: Iterable["SampleBatch"]) -> "SampleBatch":
        """
        Concatenates multiple batches

        Args:
            batches: Iterable over SampleBatch objects
        Return:
            New SamplaBatch object containing the concatenated batches
        """
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
        loss: Callable = kl_divergence,
        optimizer: (
            Optimizer | Callable[[Iterable[nn.Parameter]], Optimizer] | None
        ) = None,
        batch_size: int = 1024,
        learning_rate: float = 1e-3,
        scheduler: LRScheduler | Callable[[Optimizer], LRScheduler] | None = None,
        uniform_channel_ratio: float = 1.0,
        integration_history_length: int = 20,
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
            optimizer: optimizer for the training. Can be an optimizer object or function that is
                called with the model parameters as argument and returns the optimizer. If None, the
                Adam optimizer is used.
            batch_size: Training batch size
            learning_rate: learning rate used for the Adam optimizer
            scheduler: learning rate scheduler for the training. Can be a learning rate scheduler
                object or a function that gets the optimizer as argument and returns the scheduler.
                If None, a constant learning rate is used.
            uniform_channel_ratio: part of samples in each batch that will be distributed equally
                between all channels, value has to be between 0 and 1.
            integration_history_length: number of batches for which the channel-wise means and
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
                dims_in=integrand.input_dim,
                channels=integrand.channel_count,
                **flow_kwargs,
            )
        if cwnet is None and train_channel_weights:
            cwnet = MLP(integrand.remapped_dim, integrand.channel_count, **cwnet_kwargs)
        if cwnet is None:
            parameters = flow.parameters()
        else:
            parameters = itertools.chain(flow.parameters(), cwnet.parameters())
        if optimizer is None:
            self.optimizer = torch.optim.Adam(parameters, learning_rate)
        elif isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer(parameters)
        if scheduler is None or isinstance(scheduler, LRScheduler):
            self.scheduler = scheduler
        else:
            self.scheduler = scheduler(optimizer)

        self.integrand = integrand
        self.multichannel = integrand.channel_count is not None
        self.flow = flow
        self.cwnet = cwnet
        self.batch_size = batch_size
        self.loss = loss
        self.uniform_channel_ratio = uniform_channel_ratio
        self.drop_zero_integrands = drop_zero_integrands
        self.batch_size_threshold = batch_size_threshold

        self.minimum_buffer_size = minimum_buffer_size
        self.buffered_steps = buffered_steps
        self.max_stored_channel_weights = (
            None
            if max_stored_channel_weights is None
            or max_stored_channel_weights >= integrand.channel_count
            else max_stored_channel_weights
        )
        if buffer_capacity > 0:
            channel_count = self.max_stored_channel_weights or integrand.channel_count
            buffer_fields = [
                (integrand.input_dim,),
                None if integrand.remapped_dim is None else (integrand.remapped_dim,),
                (),
                (),
                None if integrand.channel_count is None else (),
                None if not integrand.has_channel_weight_prior else (channel_count,),
                None if self.max_stored_channel_weights is None else (channel_count,),
            ]
            buffer_dtypes = [None, None, None, None, torch.int64, None, torch.int64]
            self.buffer = Buffer(
                buffer_capacity, buffer_fields, persistent=False, dtypes=buffer_dtypes
            )
        else:
            self.buffer = None
        self.channel_dropping_threshold = channel_dropping_threshold
        self.channel_dropping_interval = channel_dropping_interval
        hist_shape = (self.integrand.channel_count or 1,)
        self.integration_history = Buffer(
            integration_history_length, [hist_shape, hist_shape, hist_shape]
        )
        self.step = 0
        if self.multichannel:
            self.register_buffer(
                "active_channels_mask",
                torch.ones((self.integrand.channel_count,), dtype=torch.bool),
            )
        # Dummy to determine device and dtype
        self.register_buffer("dummy", torch.zeros((1,)))

        self.register_buffer(
            "channel_id_map",
            (
                None
                if self.integrand.channel_grouping is None
                else torch.tensor(
                    [
                        channel.target_index
                        for channel in self.integrand.channel_grouping.channels
                    ]
                )
            ),
        )

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
                (samples.x.shape[0], self.integrand.channel_count)
            )
        else:
            log_alpha_prior = self._restore_prior(samples).log()
        log_alpha = log_alpha_prior.clone()
        mask = samples.func_vals != 0
        log_alpha[mask] += self.cwnet(samples.y[mask])
        alpha = torch.zeros_like(log_alpha)
        alpha[:, self.active_channels_mask] = F.softmax(
            log_alpha[:, self.active_channels_mask], dim=1
        )
        return alpha

    def _compute_integral(
        self, samples: SampleBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes normalized integrand and channel-wise means, variances and counts

        Args:
            samples: batch of samples
        Returns:
            A tuple containing

              - normalized integrand, shape (n, )
              - channel-wise means of the integral, shape (channels, )
              - channel-wise variances of the integral, shape (channels, )
              - channel-wise number of samples, shape (channels, )
        """
        if self.multichannel:
            alphas = torch.gather(
                self._get_alphas(samples), index=samples.channels[:, None], dim=1
            )[:, 0]
            # abs needed for non-positive integrands
            f_all = alphas * samples.func_vals.abs()
            f_div_q = f_all.detach() / samples.q_sample.detach()
            counts = torch.bincount(
                samples.channels, minlength=self.integrand.channel_count
            )
            means = torch.bincount(
                samples.channels,
                weights=f_div_q,
                minlength=self.integrand.channel_count,
            ) / counts.clip(min=1)
            variances = (
                torch.bincount(
                    samples.channels,
                    weights=(f_div_q - means[samples.channels]).square(),
                    minlength=self.integrand.channel_count,
                )
                / counts
            )
            f_true = f_all / means.sum()
        else:
            f_all = samples.func_vals.abs()
            f_div_q = f_all.detach() / samples.q_sample.detach()
            f_true = f_all / f_div_q.mean()
            means = f_div_q.mean(dim=0, keepdim=True)
            counts = torch.full_like(means, f_div_q.shape[0])
            variances = f_div_q.var(dim=0, keepdim=True)
        return f_true, means, variances, counts

    def _optimization_step(
        self,
        samples: SampleBatch,
    ) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        """
        self.optimizer.zero_grad()
        q_test = torch.ones_like(samples.func_vals)
        mask = samples.func_vals != 0.0  # TODO: think about cuts more carefully
        q_test[mask] = self.flow.prob(samples.x[mask], channel=samples.channels[mask])
        f_true, means, variances, counts = self._compute_integral(samples)
        channels = (
            samples.channels
            if self.channel_id_map is None
            else self.channel_id_map[samples.channels]
        )
        loss = self.loss(f_true, q_test, q_sample=samples.q_sample, channels=channels)
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

        n_rest = self.integrand.channel_count - self.max_stored_channel_weights
        alphas_prior_reduced = samples.alphas_prior
        epsilon = torch.finfo(alphas_prior_reduced.dtype).eps

        # strategy 1: distribute difference to 1 evenly among non-stored channels
        # alphas_prior = torch.clamp(
        #    (1 - alphas_prior_reduced.sum(dim=1, keepdims=True)) / n_rest,
        #    min=epsilon,
        # ).repeat(1, self.integrand.channel_count)
        # alphas_prior.scatter_(1, samples.alpha_channel_indices, alphas_prior_reduced)
        # return alphas_prior

        # strategy 2: set non-stored channel alphas to epsilon, normalize again
        alphas_prior = alphas_prior_reduced.new_full(
            (alphas_prior_reduced.shape[0], self.integrand.channel_count), epsilon
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
        min_len = self.integration_history.capacity if expect_full_history else 1
        if self.integration_history.size < min_len:
            return torch.ones(
                self.integrand.channel_count,
                device=self.dummy.device,
                dtype=self.dummy.dtype,
            )
        _, var_hist, count_hist = self.integration_history
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
        if (self.step + 1) % self.channel_dropping_interval != 0:
            return 0

        mean_hist, _, count_hist = self.integration_history
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
            and self.integrand.has_channel_weight_prior
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
        assert channel_weights.shape == (self.integrand.channel_count,)
        n_active_channels = torch.count_nonzero(self.active_channels_mask)
        uniform_per_channel = int(
            np.ceil(n * uniform_channel_ratio / n_active_channels)
        )
        n_per_channel = torch.full(
            (self.integrand.channel_count,),
            uniform_per_channel,
            device=self.dummy.device,
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
            remove_chan = (remove_chan + 1) % self.integrand.channel_count
        assert n_per_channel.sum() == n

        return torch.cat(
            [
                torch.full((npc,), i, device=self.dummy.device)
                for i, npc in enumerate(n_per_channel)
            ]
        )

    def _get_samples(
        self, n: int, uniform_channel_ratio: float = 0.0, train: bool = False
    ) -> SampleBatch:
        """
        Draws samples from the flow and evaluates the integrand

        Args:
            n: number of samples
            uniform_channel_ratio: Number between 0.0 and 1.0 to determine the ratio of samples
                that will be distributed uniformly first
            train: If True, the function is used in training mode, i.e. samples where the integrand
                is zero will be removed if drop_zero_integrands is True
        Returns:
            Object containing a batch of samples
        """
        channels = (
            self._get_channels(
                self.batch_size, self._get_variance_weights(), uniform_channel_ratio
            )
            if self.multichannel
            else None
        )

        batches_out = []
        x_mask_cache = []
        current_batch_size = 0
        while True:
            with torch.no_grad():
                x, prob = self.flow.sample(
                    n,
                    channel=channels,
                    return_prob=True,
                    device=self.dummy.device,
                    dtype=self.dummy.dtype,
                )

            weight, y, alphas_prior = self.integrand(x, channels)
            batch = SampleBatch(x, y, prob, weight, channels, alphas_prior)

            mask = True
            if train and self.drop_zero_integrands:
                mask = (weight != 0.0) & mask
            mask = ~(weight.isnan() | x.isnan().any(dim=1)) & mask
            current_batch_size += n if mask is True else mask.sum()
            # if mask is not True:
            #    batch = batch.map(lambda t: t[mask])
            # current_batch_size += batch.x.shape[0]
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

        samples = self._get_samples(
            self.batch_size, self.uniform_channel_ratio, train=True
        )
        online_loss, means, variances, counts = self._optimization_step(samples)
        self._store_samples(samples)
        self.integration_history.store(means[None], variances[None], counts[None])

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

    def integrate(self, n: int, batch_size: int = 1000000) -> tuple[float, float]:
        """
        Draws new samples and computes the integral.

        Args:
            n: number of samples
            batch_size: batch size used for sampling and calling the integrand
        Returns:
            tuple with the value of the integral and the MC integration error
        """
        samples = []
        for n_batch in range(0, n, batch_size):
            samples.append(self._get_samples(min(n - n_batch, batch_size)))
        with torch.no_grad():
            _, means, variances, counts = self._compute_integral(
                SampleBatch.cat(samples)
            )
        self.integration_history.store(means[None], variances[None], counts[None])
        integral = means.sum().item()
        error = torch.nansum(variances / counts).sqrt().item()
        return integral, error

    def integral(self) -> tuple[float, float]:
        """
        Returns the current estimate of the integral based on previous training iterations and calls
        to the ``integrate`` function.

        Returns:
            tuple with the value of the integral and the MC integration error
        """
        mean_hist, var_hist, count_hist = self.integration_history
        integrals = mean_hist.sum(dim=1)
        variances = torch.nansum(var_hist / count_hist, dim=1)
        weights = torch.nan_to_num(1 / variances)
        weight_sum = weights.sum()
        integral = torch.sum(weights / weight_sum * integrals).item()
        error = torch.sqrt(1 / weight_sum).item()
        return integral, error
