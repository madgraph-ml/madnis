from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn

from ..nn import MLP, Flow
from .integrand import Integrand
from .samples import SampleBatch, SampleBuffer


@dataclass
class TrainingStatus:
    """
    Contains the MadNIS training status to pass it to a callback function.
    """

    step: int
    loss: float
    dropped_channels: int


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
        learning_rate: float = 1e-3,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        uniform_channel_ratio: float = 1.0,
        variance_history_length: int = 20,
        drop_zero_integrands: bool = False,
        batch_size_threshold: float = 0.5,
        buffer_capacity: int = 0,
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
            integrand = Integrand(dims, integrand)
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
        self.flow = flow
        self.cwnet = cwnet
        self.optimizer = optimizer
        self.loss = loss
        self.scheduler = scheduler
        self.uniform_channel_ratio = uniform_channel_ratio
        self.variance_history_length = variance_history_length
        self.drop_zero_integrands = drop_zero_integrands
        self.batch_size_threshold = batch_size_threshold
        self.buffer = SampleBuffer(buffer_capacity)
        self.buffered_steps = buffered_steps
        self.max_stored_channel_weights = (
            None
            if max_stored_channel_weights is None
            or max_stored_channel_weights >= integrand.channels
            else max_stored_channel_weights
        )
        self.channel_dropping_threshold = channel_dropping_threshold
        self.channel_dropping_interval = channel_dropping_interval
        self.step = 0

        if device is not None:
            self.to(device)
        if dtype is not None:
            self.to(dtype)

    ###################################################################################################
    # OLD SHIT BEGINS HERE                                                                            #
    ###################################################################################################

    def _store_samples(
        self,
        samples: SampleBatch,
    ):
        """Stores the generated samples and probabilites to re-use for the buffered training.
        Args:
            samples (SampleBatch):
                Object containing a batch of samples
        """
        if self.sample_capacity == 0:
            return

        if self.max_stored_alphas is not None and samples.alphas_prior is not None:
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
            samples.alphas_prior = largest_alphas[:, : self.max_stored_alphas].clone()
            samples.alpha_channel_indices = alpha_indices[
                :, : self.max_stored_alphas
            ].clone()

        self.stored_samples.append(samples)
        del self.stored_samples[: -self.sample_capacity]

    def _get_integral_and_alphas(
        self,
        samples: SampleBatch,
    ):
        """Return the weighted integrand and the channel weights
        Args:
            samples (SampleBatch):
                Object containing a batch of samples
        Returns:
            integrand (torch.Tensor):
                True probability weighted by alpha / qsample (nsamples,)
            alphas (torch.Tensor):
                Channel weights with shape (nsamples, nchannels)
        """
        _, _, alphas = self._get_probs_alphas(samples)
        channel_alphas = torch.gather(alphas, index=samples.channels[:, None], dim=1)[
            :, 0
        ]
        return channel_alphas * samples.func_vals / samples.q_sample, alphas

    def _get_probs_alphas(self, samples: SampleBatch):
        """Run the flow to extract the probability and its logarithm for the given samples x.
        Compute the corresponding channel weights using y.
        Args:
            samples (SampleBatch):
                Object containing a batch of samples
        Returns:
            q_test (torch.Tensor):
                true probability weighted by alpha / qsample (nsamples,)
            logq (torch.Tensor):
                true probability weighted by alpha / qsample (nsamples,)
            alphas (torch.Tensor):
                Channel weights with shape (nsamples, nchannels)
        """
        one_hot_channels = F.one_hot(samples.channels, self.n_channels)
        logq = self.dist.log_prob(samples.x, condition=one_hot_channels)
        q_test = logq.exp()

        nsamples = samples.y.shape[0]
        if self.train_mcw:
            if self.use_weight_init:
                if samples.alphas_prior is None:
                    init_weights = torch.full(
                        (nsamples, self.n_channels),
                        1 / self.n_channels,
                        device=samples.y.device,
                    )
                else:
                    init_weights = self._restore_prior(samples)
                alphas = self.mcw_model(
                    [samples.y, init_weights, self.active_channels_mask]
                )
            else:
                alphas = self.mcw_model(samples.y)
        else:
            if samples.alphas_prior is not None:
                alphas = self._restore_prior(samples)
            else:
                alphas = torch.full(
                    (nsamples, self.n_channels),
                    1 / self.n_channels,
                    device=samples.y.device,
                )

        return q_test, logq, alphas

    def _restore_prior(self, samples: SampleBatch):
        if samples.alpha_channel_indices is None:
            return samples.alphas_prior

        n_rest = self.n_channels - self.max_stored_alphas
        alphas_prior_reduced = samples.alphas_prior

        # strategy 1: distribute difference to 1 evenly among non-stored channels
        # alphas_prior = torch.clamp(
        #    (1 - alphas_prior_reduced.sum(dim=1, keepdims=True)) / n_rest,
        #    min=_EPSILON,
        # ).repeat(1, self.n_channels)
        # alphas_prior.scatter_(1, samples.alpha_channel_indices, alphas_prior_reduced)
        # return alphas_prior

        # strategy 2: set non-stored channel alphas to _EPSILON, normalize again
        alphas_prior = torch.full(
            (alphas_prior_reduced.shape[0], self.n_channels),
            _EPSILON,
            device=alphas_prior_reduced.device,
            dtype=alphas_prior_reduced.dtype,
        )
        alphas_prior.scatter_(1, samples.alpha_channel_indices, alphas_prior_reduced)
        return alphas_prior / alphas_prior.sum(dim=1, keepdims=True)

    def _get_funcs_integral(
        self,
        alphas: torch.Tensor,
        q_sample: torch.Tensor,
        func_vals: torch.Tensor,
        channels: torch.Tensor,
    ):
        """Compute the channel wise means and variances of the integrand and return the
        channel-wise alpha-weighted integrands.
        Args:
            alphas (torch.Tensor):
                Channel weights with shape (nsamples, nchannels)
            q_sample (torch.Tensor):
                Test probability of all mappings (analytic + flow) with shape (nsamples,)
            func_vals (torch.Tensor):
                True probability/function with shape (nsamples,)
            channels (torch.Tensor):
                Tensor encoding which channel to use with shape (nsamples,)
        Returns:
            f_true (torch.Tensor):
                True channel integrands (incl. alpha) with shape (nsamples,)
            means (torch.Tensor):
                Channel-wise means of the integrand with shape (nchannels,)
            vars (torch.Tensor):
                Channel-wise variances of the integrand with shape (nchannels,)
            counts (torch.Tensor):
                Channel-wise number of samples with shape (nchannels,)
        """
        alphas = torch.gather(alphas, index=channels[:, None], dim=1)[:, 0]
        f_all = alphas * func_vals.abs()  # abs needed for non-positive integrands
        means = []
        vars = []
        counts = []
        for i in range(self.n_channels):
            mask = channels == i
            fi = f_all[mask]
            qi = q_sample[mask]
            fi_div_qi = fi / qi
            meani = (
                fi_div_qi.mean()
                if fi.shape[0] > 0
                else torch.tensor(0.0, device=fi.device, dtype=fi.dtype)
            )
            means.append(meani)
            vars.append(fi_div_qi.var(False))
            counts.append(fi.shape[0])
        # check norm again, comment if not needed
        norm = sum(means).detach()
        f_true = f_all / norm
        logf = torch.log(f_true + _EPSILON)  # Maybe kick-out later

        return (
            f_true,
            logf,
            torch.tensor(means),
            torch.tensor(vars),
            torch.tensor(counts),
        )

    def _optimization_step(
        self,
        samples: SampleBatch,
    ):
        """Perform one optimization step of the networks with the given samples
        Args:
            samples (SampleBatch):
                Object containing a batch of samples
        Returns:
            loss (float or tuple[float, float]):
                Result for the loss (scalar)
            means (torch.Tensor):
                Channel-wise means of the integrand with shape (nchannels,)
            vars (torch.Tensor):
                Channel-wise variances of the integrand with shape (nchannels,)
            counts (torch.Tensor):
                Channel-wise number of samples with shape (nchannels,)
        """
        self.optimizer.zero_grad()
        q_test, logq, alphas = self._get_probs_alphas(samples)
        f_true, logf, means, vars, counts = self._get_funcs_integral(
            alphas, samples.q_sample, samples.func_vals, samples.channels
        )
        loss = self.loss_func(
            samples.channels, f_true, q_test, q_sample=samples.q_sample
        )
        if loss.isnan().item():
            print("nan batch: skipping optimization")
        else:
            loss.backward()
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        ret_loss = loss.item()

        return ret_loss, means, vars, counts

    def _get_variance_weights(self, expect_full_history=True):
        """Use the list of saved variances to compute weights for sampling the
        channels to allow for stratified sampling.
        Returns:
            w (torch.Tensor):
                Weights for sampling the channels with shape (nchannels,)
        """
        min_len = self.variance_history_length if expect_full_history else 1
        if len(self.variance_history) < min_len:
            return torch.ones(self.n_channels)

        count_hist = torch.stack(self.count_history, dim=0).type(
            torch.get_default_dtype()
        )
        var_hist = torch.stack(self.variance_history, dim=0)
        count_hist[var_hist.isnan()] = np.nan
        hist_weights = count_hist / count_hist.nansum(dim=0)
        w = torch.nansum(hist_weights * var_hist, dim=0).sqrt()
        return w

    def disable_unused_channels(self, irrelevance_threshold: float):
        count_hist = torch.stack(self.count_history, dim=0).type(
            torch.get_default_dtype()
        )
        mean_hist = torch.stack(self.mean_history, dim=0)
        mean_hist[mean_hist.isnan()] = 0.0
        hist_weights = count_hist / count_hist.sum(dim=0)
        channel_integrals = torch.nansum(hist_weights * mean_hist, dim=0)
        channel_rel_integrals = channel_integrals / channel_integrals.sum()
        cri_sort, cri_argsort = torch.sort(channel_rel_integrals)
        n_irrelevant = torch.count_nonzero(
            cri_sort.cumsum(dim=0) < irrelevance_threshold
        )
        n_disabled = torch.count_nonzero(
            self.active_channels_mask[cri_argsort[:n_irrelevant]]
        )
        self.active_channels_mask[cri_argsort[:n_irrelevant]] = False
        for samples in self.stored_samples:
            mask = self.active_channels_mask[samples.channels]
            samples.x = samples.x[mask]
            samples.y = samples.y[mask]
            samples.q_sample = samples.q_sample[mask]
            samples.func_vals = samples.func_vals[mask]
            samples.channels = samples.channels[mask]
            samples.alphas_prior = (
                None if samples.alphas_prior is None else samples.alphas_prior[mask]
            )
            samples.z = None if samples.z is None else samples.z[mask]
            samples.alpha_channel_indices = (
                None
                if samples.alpha_channel_indices is None
                else samples.alpha_channel_indices[mask]
            )
        return n_disabled

    def delete_samples(self):
        """Delete all stored samples."""
        del self.stored_samples[:]

    def append_variance(self, vars, counts, means):
        self.variance_history.append(vars)
        del self.variance_history[: -self.variance_history_length]
        self.count_history.append(counts)
        del self.count_history[: -self.variance_history_length]
        self.mean_history.append(means)
        del self.mean_history[: -self.variance_history_length]

    def train_one_step(
        self,
        nsamples: int,
        integral: bool = False,
    ):
        """Perform one step of integration and improve the sampling.
        Args:
            nsamples (int):
                Number of samples to be taken in a training step
            integral (bool, optional):
                return the integral value. Defaults to False.
        Returns:
            loss:
                Value of the loss function for this step
            integral (optional):
                Estimate of the integral value
            uncertainty (optional):
                Integral statistical uncertainty
        """

        # Sample from flow and update
        channels = self._get_channels(
            nsamples,
            self._get_variance_weights(),
            self.uniform_channel_ratio,
        )
        samples, _ = self._get_samples(channels)
        loss, means, vars, counts = self._optimization_step(samples)
        self._store_samples(samples)

        self.append_variance(vars, counts, means)

        if integral:
            return (
                loss,
                means.sum(),
                torch.sqrt(torch.sum(vars / (counts - 1.0))),
            )

        return loss

    def train_on_stored_samples(self, batch_size: int):
        """Train the network on all saved samples.
        Args:
            batch_size (int):
                Size of the batches that the saved samples are split up into
        Returns:
            loss:
                Averaged value of the loss function
        """
        sample_count = sum(int(item.x.shape[0]) for item in self.stored_samples)
        perm = torch.randperm(sample_count)

        keys = [k for k, v in asdict(self.stored_samples[0]).items() if v is not None]
        dataset = torch.utils.data.TensorDataset(
            *(
                torch.cat(item, dim=0)[perm]
                for item in zip(*self.stored_samples)
                if item[0] is not None
            )
        )
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True
        )

        losses = []
        for fields in data_loader:
            loss, _, _, _ = self._optimization_step(
                SampleBatch(**dict(zip(keys, fields)))
            )
            losses.append(loss)
        return np.nanmean(losses, axis=0)

    ###################################################################################################
    # OLD SHIT ENDS HERE                                                                              #
    ###################################################################################################

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
            batch = SampleBatch(x, y, prob, weight, channels, alphas_prior, z)

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
        status = TrainingStatus()

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
