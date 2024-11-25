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
    pass


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

    def train_step(self, samples_per_channel: int) -> VegasTrainingStatus:
        """
        Performs a single VEGAS training iteration

        Args:
            samples_per_channel: number of training samples per channel
        Returns:
            ``VegasTrainingStatus`` object containing metrics of the training progress
        """
        n_channels = len(self.grid_channels)
        input_dim = self.integrand.input_dim
        variances = torch.zeros(n_channels)
        counts = torch.zeros(n_channels)
        means = torch.zeros(n_channels)
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
            alpha = torch.gather(alphas, index=channels[:, None], dim=1)[:, 0]
            f = jac * func_vals.numpy() * alpha.numpy()
            if self.integrator.drop_zero_integrands:
                mask_np = f != 0.0
                f = f[mask_np]
                r = r[mask_np]
                jac = jac[mask_np]
                mask_torch = torch.as_tensor(mask_np)
                y = y[mask_torch]
                func_vals = func_vals[mask_torch]
                channels = channels[mask_torch]
                alphas = alphas[mask_torch]
            grid.add_training_data(r, f**2)
            grid.adapt(alpha=self.damping)

            for chan in grid_channels:
                variances[chan] = np.var(f)
                counts[chan] = len(f)
                means[chan] = np.mean(f)
                self.integrator.integration_history.store(
                    means[None], variances[None], counts[None]
                )
            self.integrator._store_samples(
                SampleBatch(
                    x_torch, y, torch.from_numpy(1 / jac), func_vals, channels, alphas
                ).map(lambda t: t.to(self.integrator.dummy.device))
            )

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
        # if channel_weight_mode == "uniform":
        #    n_per_channel =
        # else:
        samples_per_channel = n // len(self.grids)
        input_dim = self.integrand.input_dim
        samples = []
        for i, grid in enumerate(self.grids):
            x = np.empty((samples_per_channel, input_dim), float)
            jac = np.empty(samples_per_channel, float)

            r = self.rng.random((samples_per_channel, input_dim))
            grid.map(r, x, jac)
            x_torch = torch.as_tensor(x, dtype=self.integrator.dummy.dtype)
            if self.integrator.multichannel:
                # TODO: no need for random numbers here
                channels = torch.from_numpy(
                    self.rng.choice(self.grid_channels[i], (samples_per_channel,))
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

    def integrate(self, n: int) -> tuple[float, float]:
        """
        Draws samples and computes the integral.

        Args:
            n: number of samples
            batch_size: batch size used for sampling and calling the integrand
        Returns:
            tuple with the value of the integral and the MC integration error
        """
        pass

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
        pass

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
        pass
