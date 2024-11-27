import signal
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import torch

from .integrator import Integrator, SampleBatch
from .metrics import (
    IntegrationMetrics,
    UnweightingMetrics,
    integration_metrics,
    unweighting_metrics,
)


@dataclass
class VegasTrainingStatus:
    """
    Contains the VEGAS training status to pass it to a callback function.

    Args:
        step: optimization step
        variance: integration variance in this iteration. Combined for all channels in the
            multi-channel case
    """

    step: int
    variance: float
    # TODO: maybe add more, like channel-wise statistics?


class VegasPreTraining:
    """
    Implements VEGAS pre-training. It wraps around an ``Integrator`` object and uses its integrand,
    sample buffer and integration history. In addition, it also defines the functions sample,
    integrate, integration_metrics and unweighting_metrics to allow for comparisions between VEGAS
    and MadNIS.
    """

    def __init__(
        self,
        integrator: Integrator,
        bins: int = 64,
        damping: float = 0.7,
    ):
        """ """
        import vegas  # vegas module is optional dependency, so import locally

        # TODO: also maybe replace this with our own VEGAS implementation in pytorch?

        self.integrator = integrator
        self.integrand = integrator.integrand
        self.damping = damping

        if self.integrator.multichannel:
            if self.integrand.channel_grouping is None:
                self.grid_channels = [
                    [channel] for channel in range(self.integrand.channel_count)
                ]
            else:
                self.grid_channels = [
                    group.channel_indices
                    for group in self.integrand.channel_grouping.groups
                ]
        else:
            self.grid_channels = [[0]]
        self.grids = [
            vegas.AdaptiveMap(grid=[[0, 1]] * self.integrand.input_dim, ninc=bins)
            for _ in self.grid_channels
        ]
        self.rng = np.random.default_rng()
        self.step = 0

    def train_step(self, samples_per_channel: int) -> VegasTrainingStatus:
        """
        Performs a single VEGAS training iteration

        Args:
            samples_per_channel: number of training samples per channel
        Returns:
            ``VegasTrainingStatus`` object containing metrics of the training progress
        """
        n_channels = self.integrand.channel_count if self.integrator.multichannel else 1
        input_dim = self.integrand.input_dim
        variances = torch.zeros(n_channels)
        counts = torch.zeros(n_channels)
        means = torch.zeros(n_channels)
        # TODO: maybe implement stratified training instead of uniform channels
        for grid, grid_channels in zip(self.grids, self.grid_channels):
            x = np.empty((samples_per_channel, input_dim), float)
            jac = np.empty(samples_per_channel, float)

            r = self.rng.random((samples_per_channel, input_dim))
            grid.map(r, x, jac)
            x_torch = torch.as_tensor(x, dtype=self.integrator.dummy.dtype)
            if self.integrator.multichannel:
                # TODO: no need for random numbers here
                channels = torch.from_numpy(
                    self.rng.choice(grid_channels, (samples_per_channel,))
                )
            else:
                channels = None
            func_vals, y, alphas = self.integrand(x_torch, channels)
            if self.integrator.multichannel:
                alpha = torch.gather(alphas, index=channels[:, None], dim=1)[:, 0]
                f = jac * func_vals.numpy() * alpha.numpy()
            else:
                f = jac * func_vals.numpy()
            if self.integrator.drop_zero_integrands:
                mask_np = f != 0.0
                f = f[mask_np]
                r = r[mask_np]
                jac = jac[mask_np]
                mask_torch = torch.as_tensor(mask_np)
                if self.integrator.multichannel:
                    zero_counts = torch.bincount(
                        channels[~mask_torch], minlength=self.integrand.channel_count
                    )
                    channels = channels[mask_torch]
                    alphas = alphas[mask_torch]
                else:
                    zero_counts = torch.full((1,), torch.count_nonzero(~mask_torch))
                x_torch = x_torch[mask_torch]
                y = y[mask_torch]
                func_vals = func_vals[mask_torch]
            else:
                zero_counts = None
            grid.add_training_data(r, f**2)
            grid.adapt(alpha=self.damping)

            for chan in grid_channels:
                f_chan = torch.from_numpy(f)
                if self.integrator.multichannel:
                    f_chan = f_chan[channels == chan]
                counts_chan = len(f_chan)
                if zero_counts is not None:
                    counts_chan += zero_counts[chan]
                counts[chan] = counts_chan
                means[chan] = f_chan.sum() / counts_chan
                variances[chan] = (f_chan - means[chan]).square().sum() / counts_chan
            self.integrator._store_samples(
                SampleBatch(
                    x_torch,
                    y,
                    torch.from_numpy(1 / jac),
                    func_vals,
                    channels,
                    alphas,
                    zero_counts=zero_counts,
                ).map(lambda t: t.to(self.integrator.dummy.device))
            )
        self.integrator.integration_history.store(
            means[None], variances[None], counts[None]
        )
        status = VegasTrainingStatus(
            step=self.step,
            variance=torch.sqrt(torch.nansum(variances / counts) * counts.sum()).item()
            / means.sum().item(),
        )
        self.step += 1
        return status

    def train(
        self,
        samples_per_channel: list[int],
        callback: Callable[[VegasTrainingStatus], None] | None = None,
        capture_keyboard_interrupt: bool = False,
    ):
        """
        Performs multiple training steps

        Args:
            samples_per_channel: list of the number of samples per channel, with one entry for every
                training iteration
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
            for sample_count in samples_per_channel:
                status = self.train_step(sample_count)
                if callback is not None:
                    callback(status)
                if interrupted:
                    break
        finally:
            if capture_keyboard_interrupt:
                signal.signal(signal.SIGINT, old_handler)

    def initialize_integrator(self):
        """
        Initializes the flows in the integrator object using the trained VEGAS grid
        """
        grids_torch = [
            torch.as_tensor(
                grid.extract_grid(),
                device=self.integrator.dummy.device,
                dtype=self.integrator.dummy.dtype,
            )
            for grid in self.grids
        ]
        grids = (
            torch.stack(grids_torch, dim=0)
            if self.integrator.multichannel
            else grids_torch[0]
        )
        self.integrator.flow.init_with_grid(grids)

    def sample(
        self,
        n: int,
        batch_size: int = 100000,
        channel_weight_mode: Literal["uniform", "mean", "variance"] = "variance",
    ) -> SampleBatch:
        """
        Draws samples and computes their integration weight

        Args:
            n: number of samples
            batch_size: batch size used for sampling and calling the integrand
            channel_weight_mode: specifies whether the channels are weighted by their mean,
                variance or uniformly. Note that weighting by mean can lead to problems for
                non-positive functions
        Returns:
            ``SampleBatch`` object, see its documentation for details
        """
        if channel_weight_mode == "uniform":
            uniform_channel_ratio = 1.0
            channel_weight_mode = "variance"
        else:
            uniform_channel_ratio = 0.0
        channel_weights = self.integrator._get_channel_contributions(
            False, channel_weight_mode
        )
        samples_per_channel = self.integrator._get_channels(
            n, channel_weights, uniform_channel_ratio, return_counts=True
        )

        input_dim = self.integrand.input_dim
        samples = []
        for grid, grid_channels in zip(self.grids, self.grid_channels):
            n_samples = samples_per_channel[grid_channels].sum()
            x = np.empty((n_samples, input_dim), float)
            jac = np.empty(n_samples, float)

            r = self.rng.random((n_samples, input_dim))
            grid.map(r, x, jac)
            x_torch = torch.as_tensor(x, dtype=self.integrator.dummy.dtype)
            if self.integrator.multichannel:
                channels = torch.cat(
                    [
                        torch.full((samples_per_channel[channel_index],), channel_index)
                        for channel_index in grid_channels
                    ]
                )
            else:
                channels = None
            func_vals, y, alphas = self.integrand(x_torch, channels)
            alpha = torch.gather(alphas, index=channels[:, None], dim=1)[:, 0]
            f = jac * func_vals.numpy() * alpha.numpy()

            sample_batch = SampleBatch(
                x_torch,
                y,
                torch.from_numpy(1 / jac),
                func_vals,
                channels,
                alphas,
                None,
                torch.from_numpy(f),
                alphas,
            )
            samples.append(
                sample_batch.map(lambda t: t.to(self.integrator.dummy.device))
            )
        return SampleBatch.cat(samples)

    def _compute_integral(
        self, samples: SampleBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.integrator.multichannel:
            counts = torch.bincount(
                samples.channels, minlength=self.integrand.channel_count
            )
            means = torch.bincount(
                samples.channels,
                weights=samples.weights,
                minlength=self.integrand.channel_count,
            ) / counts.clip(min=1)
            variances = (
                torch.bincount(
                    samples.channels,
                    weights=(samples.weights - means[samples.channels]).square(),
                    minlength=self.integrand.channel_count,
                )
                / counts
            )
        else:
            means = samples.weights.mean(dim=0, keepdim=True)
            counts = torch.full((1,), samples.weights.shape[0], device=means.device)
            variances = samples.weights.var(dim=0, keepdim=True)
        return means, variances, counts

    def integrate(self, n: int) -> tuple[float, float]:
        """
        Draws samples and computes the integral.

        Args:
            n: number of samples
            batch_size: batch size used for sampling and calling the integrand
        Returns:
            tuple with the value of the integral and the MC integration error
        """
        means, variances, counts = self._compute_integral(self.sample(n))
        self.integrator.integration_history.store(
            means[None], variances[None], counts[None]
        )
        integral = means.sum().item()
        error = torch.nansum(variances / counts).sqrt().item()
        return integral, error

    def integration_metrics(
        self, n: int, batch_size: int = 100000
    ) -> IntegrationMetrics:
        """
        Draws samples and computes metrics for the total and channel-wise integration quality.

        Args:
            n: number of samples
            batch_size: batch size used for sampling and calling the integrand
        Returns:
            ``IntegrationMetrics`` object, see its documentation for details
        """
        means, variances, counts = self._compute_integral(self.sample(n))
        self.integrator.integration_history.store(
            means[None], variances[None], counts[None]
        )
        return integration_metrics(means, variances, counts)

    def unweighting_metrics(
        self,
        n: int,
        batch_size: int = 100000,
        channel_weight_mode: Literal["uniform", "mean", "variance"] = "mean",
    ) -> UnweightingMetrics:
        """
        Draws samples and computes metrics for the total and channel-wise integration quality.
        This function is only suitable for functions that are non-negative everywhere.

        Args:
            n: number of samples
            batch_size: batch size used for sampling and calling the integrand
            channel_weight_mode: specifies whether the channels are weighted by their mean,
                variance or uniformly.
        Returns:
            ``UnweightingMetrics`` object, see its documentation for details
        """
        samples = self.sample(n, batch_size, channel_weight_mode)
        return unweighting_metrics(
            samples.weights, samples.channels, self.integrand.channel_count
        )
