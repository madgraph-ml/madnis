""" Implementation of multi-channel loss functions """

from functools import wraps
from typing import Callable

import torch


def wrapped_loss(func: Callable) -> Callable:
    """Implement multi-channel decorator.

    This decorator wraps the function to implement multi-channel
    functionality by splitting into the different contributions and
    summing up the different channel contributions

    Arguments:
        func: function to be wrapped

    Returns:
        Callable: decoratated divergence
    """

    @wraps(func)
    def wrapped_multi(
        self,
        channels: torch.Tensor,
        f_true: torch.Tensor,
        q_test: torch.Tensor,
        q_sample: torch.Tensor | None = None,
        f_true_detached_x: torch.Tensor | None = None,
    ):
        if q_sample is None:
            q_sample = q_test

        n_samples = torch.tensor(
            q_sample.shape[0], dtype=f_true.dtype, device=f_true.device
        )

        loss_tot = 0
        for mask in self.mask_iter(channels):
            fi, qti, qsi = f_true[mask], q_test[mask], q_sample[mask]
            fi_detached_x = f_true_detached_x[mask] if self.single_pass_opt else None
            ni = mask.sum().type(f_true.dtype)
            loss_tot += (
                n_samples / ni * func(self, fi, qti, qsi, fi_detached_x)
                if ni > 0
                else 0.0
            )

        return loss_tot

    return wrapped_multi


class MultiChannelLoss:
    """Multi-channel loss.

    This class contains the variance loss
    appropriate to be used in combination with a multi-channel
    loss and stratified sampling.

    **Remarks:**:
    - We only employ the variance loss (so far) as this is the only one
      which can handle unnormalized integrands (which is needed here).

    - It uses importance sampling explicitly, i.e. the estimator is divided
      by an additional factor of ``q_sample``.

    - [20/03] Added more losses! Be careful im multi-channel setting!

    """

    def __init__(
        self,
        n_channels: int = 1,
        single_pass_opt: bool = False,
        alpha: float = None,
        channel_grouping: Optional[ChannelGrouping] = None,
    ):
        """
        Args:
            n_channels (int, optional): the number of channels used for the integration.
                Defaults to 1.
            single_pass_opt (bool, optional): allows single-pass optimization.
                Defaults to False.
            alpha (float, optional): needed for alpha-divergence. Defaults to None.
        """
        self.alpha = alpha
        self.n_channels = n_channels
        self.single_pass_opt = single_pass_opt
        self.divergences = [
            x
            for x in dir(self)
            if (
                "__" not in x
                and "mask_iter" not in x
                and "n_channels" not in x
                and "single_pass_opt" not in x
                and "alpha" not in x
            )
        ]

        if channel_grouping is None:
            self.grouping_masks = None
        else:
            grouping_masks = torch.zeros(
                (self.n_channels, self.n_channels), dtype=torch.bool
            )
            for channel in enumerate(channel_grouping.channels):
                grouping_masks[channel.target_index, channel.channel_index] = True
            self.grouping_masks = grouping_masks[grouping_masks.any(dim=1)]

    @wrapped_loss
    def variance(
        self,
        f_true: torch.Tensor,
        q_test: torch.Tensor,
        q_sample: torch.Tensor,
        f_true_detached_x: Optional[torch.Tensor] = None,
    ):
        """Implement basic variance.

        This function returns the variance loss for two given sets
        of functions, ``f_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_sample``.

        **Remarks:**
            1. In the variance loss the ``f_true`` function does not be normalized to 1.
               In the multi-channel application, do not individually normalize.
            2. When using only a small number of integrand calls (N) as relevant for an
               efficient batching, the following theorem should not be used:

                   <g (f/g - E[f/g])^2> = <f^2/g> - E[f/g]^2

               because it uses that ``<g> = 1``, which is not necessarily true
               if this calculated using a monte carlo estimate. This becomes even more
               problematic if the sampling is happening via a different function q !=g.
               Then, in general, the numerical calculation

                   <g> = < g q/q> ~ 1/N sum_{x_i} g(x_i)/q(x_i),

               might be different from 1 for not large enough N.

        Arguments:
            f_true (torch.Tensor): true function. Does not have to be normalized.
            q_test (torch.Tensor): estimated function/probability
            q_sample (torch.Tensor): sampling probability

        Returns:
            torch.Tensor: computed variance loss

        """
        if self.single_pass_opt:
            ratio = 1.0
            mean = torch.mean(f_true_detached_x / q_test.detach())
        else:
            ratio = q_test / q_sample
            mean = torch.mean(f_true / q_sample)
        sq = (f_true / q_test - mean) ** 2
        return (
            torch.mean(sq * ratio)
            if len(f_true) > 0
            else torch.tensor(0.0, device=f_true.device, dtype=f_true.dtype)
        )

    @wrapped_loss
    def test_variance(
        self,
        f_true: torch.Tensor,
        q_test: torch.Tensor,
        q_sample: torch.Tensor,
        f_true_detached_x: Optional[torch.Tensor] = None,
    ):
        """Implement basic variance for testing

        Arguments:
            f_true (torch.Tensor): true function. Does not have to be normalized.
            q_test (torch.Tensor): estimated function/probability
            q_sample (torch.Tensor): sampling probability

        Returns:
            torch.Tensor: computed variance loss

        """
        del f_true_detached_x
        if self.single_pass_opt:
            ratio = 1.0
            mean = 1.0
        else:
            ratio = q_test / q_sample
            mean = 1.0
        sq = (f_true / q_test - mean) ** 2
        return (
            torch.mean(sq * ratio)
            if len(f_true) > 0
            else torch.tensor(0.0, device=f_true.device, dtype=f_true.dtype)
        )

    @wrapped_loss
    def kl_divergence(
        self,
        f_true: torch.Tensor,
        q_test: torch.Tensor,
        q_sample: torch.Tensor,
        f_true_detached_x: Optional[torch.Tensor] = None,
    ):
        """Implement Kullback-Leibler (KL) divergence.

        This function returns the Kullback-Leibler divergence for two given sets
        of probabilities, ``f_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_sample``.

        Arguments:
            f_true (torch.Tensor): true function. Should be normalized.
            q_test (torch.Tensor): estimated function/probability
            q_sample (torch.Tensor): sampling probability

        Returns:
            torch.Tensor: computed KL divergence

        """
        del f_true_detached_x
        if self.single_pass_opt:
            ratio = f_true / q_test
        else:
            ratio = f_true / q_sample
        logq = torch.log(q_test)
        logf = torch.log(f_true + _EPSILON)
        return torch.mean(ratio * (logf - logq))

    @wrapped_loss
    def gkl_divergence(
        self,
        f_true: torch.Tensor,
        q_test: torch.Tensor,
        q_sample: torch.Tensor,
        f_true_detached_x: Optional[torch.Tensor] = None,
    ):
        """Implement generalized Kullback-Leibler (GKL) divergence.

        This function returns the Kullback-Leibler divergence for two given sets
        of probabilities, ``f_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_sample``.

        Arguments:
            f_true (torch.Tensor): true function. Does not need be normalized.
            q_test (torch.Tensor): estimated function/probability
            q_sample (torch.Tensor): sampling probability

        Returns:
            torch.Tensor: computed generalized KL divergence

        """
        del f_true_detached_x
        if self.single_pass_opt:
            ratio = f_true / q_test
        else:
            ratio = f_true / q_sample
        logq = torch.log(q_test)
        logf = torch.log(f_true + _EPSILON)
        phi = (logf - logq) + q_test / f_true - 1.0
        return torch.mean(ratio * phi)

    @wrapped_loss
    def rkl_divergence(
        self,
        f_true: torch.Tensor,
        q_test: torch.Tensor,
        q_sample: torch.Tensor,
        f_true_detached_x: Optional[torch.Tensor] = None,
    ):
        """Implement reverse Kullback-Leibler (RKL) divergence.

        This function returns the reverse Kullback-Leibler divergence for two given sets
        of probabilities, ``f_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_sample``.

        Arguments:
            f_true (torch.Tensor): true function. Should be normalized.
            q_test (torch.Tensor): estimated function/probability
            q_sample (torch.Tensor): sampling probability

        Returns:
            torch.Tensor: computed RKL divergence

        """
        del f_true_detached_x
        if self.single_pass_opt:
            ratio = 1.0
        else:
            ratio = q_test / q_sample
        logq = torch.log(q_test)
        logf = torch.log(f_true + _EPSILON)
        return torch.mean(ratio * (logq - logf))

    @wrapped_loss
    def grkl_divergence(
        self,
        f_true: torch.Tensor,
        q_test: torch.Tensor,
        q_sample: torch.Tensor,
        f_true_detached_x: Optional[torch.Tensor] = None,
    ):
        """Implement generalized reverse Kullback-Leibler (GRKL) divergence.

        This function returns the reverse Kullback-Leibler divergence for two given sets
        of probabilities, ``f_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_sample``.

        Arguments:
            f_true (torch.Tensor): true function. Does not need be normalized.
            q_test (torch.Tensor): estimated function/probability
            q_sample (torch.Tensor): sampling probability

        Returns:
            torch.Tensor: computed generalized RKL divergence

        """
        del f_true_detached_x
        if self.single_pass_opt:
            ratio = 1.0
        else:
            ratio = q_test / q_sample
        logq = torch.log(q_test)
        logf = torch.log(f_true + _EPSILON)
        phi = (logq - logf) + f_true / q_test - 1.0
        return torch.mean(ratio * phi)

    @wrapped_loss
    def alpha_divergence(
        self,
        f_true: torch.Tensor,
        q_test: torch.Tensor,
        q_sample: torch.Tensor,
        f_true_detached_x: Optional[torch.Tensor] = None,
    ):
        """Implement alpha divergence. Defined for instance in:
            [1] 2208.01893

        This function returns the alpha divergence for two given sets
        of probabilities, ``f_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_sample``.

        Arguments:
            f_true (torch.Tensor): true function. Should be normalized.
            q_test (torch.Tensor): estimated function/probability
            q_sample (torch.Tensor): sampling probability

        Returns:
            torch.Tensor: computed alpha-divergence

        """
        del f_true_detached_x
        if self.alpha is None:
            raise (f"alpha is set to {self.alpha} instead of float")

        if self.single_pass_opt:
            ratio = 1.0
        else:
            ratio = q_test / q_sample
        prefactor = 1 / self.alpha / (self.alpha - 1.0)
        f_alpha = torch.pow(f_true, self.alpha)
        q_alpha = torch.pow(q_test, -self.alpha)
        return prefactor * torch.mean(ratio * f_alpha * q_alpha)

    # def variance(
    #     self,
    #     channels: torch.Tensor,
    #     f_true: torch.Tensor,
    #     q_test: torch.Tensor,
    #     q_sample: torch.Tensor = None,
    #     f_true_detached_x: Optional[torch.Tensor] = None,
    # ):
    #     if q_sample is None:
    #         q_sample = q_test

    #     n_samples = torch.tensor(
    #         q_sample.shape[0], dtype=f_true.dtype, device=f_true.device
    #     )

    #     # TODO: check this! q_sample should never have gradients in regular madnis training
    #     #       so no detach() should be necessary
    #     #if not self.single_pass_opt:
    #     #    q_sample = q_sample.detach()

    #     var_tot = 0
    #     for mask in self.mask_iter(channels):
    #         fi, qti, qsi = f_true[mask], q_test[mask], q_sample[mask]
    #         fi_detached_x = f_true_detached_x[mask] if self.single_pass_opt else None
    #         ni = mask.sum().type(f_true.dtype)
    #         var_tot += (
    #             n_samples / ni * self._variance(fi,qti,qsi,fi_detached_x)
    #             if ni > 0 else 0.
    #         )

    #     return var_tot

    def _variance(
        self,
        f_true: torch.Tensor,
        q_test: torch.Tensor,
        q_sample: torch.Tensor,
        f_true_detached_x: Optional[torch.Tensor] = None,
    ):
        # Might cause issues - see remark above
        # mean2 = torch.mean(f_true**2 / (q_test * q_sample))
        # mean = torch.mean(f_true / q_sample)
        # return mean2 - mean**2
        if self.single_pass_opt:
            ratio = 1.0
            mean = torch.mean(f_true_detached_x / q_test.detach())
        else:
            ratio = q_test / q_sample
            mean = torch.mean(f_true / q_sample)
        sq = (f_true / q_test - mean) ** 2
        return (
            torch.mean(sq * ratio)
            if len(f_true) > 0
            else torch.tensor(0.0, device=f_true.device, dtype=f_true.dtype)
        )
        return torch.mean(sq * ratio)

    def stratified_variance(
        self,
        channels: torch.Tensor,
        f_true: torch.Tensor,
        q_test: torch.Tensor,
        q_sample: torch.Tensor = None,
        f_true_detached_x: Optional[torch.Tensor] = None,
    ):
        if q_sample is None:
            q_sample = q_test

        stddev_sum = 0
        for mask in self.mask_iter(channels):
            fi, qti, qsi = f_true[mask], q_test[mask], q_sample[mask]
            fi_detached_x = f_true_detached_x[mask] if self.single_pass_opt else None
            stddev_sum += torch.sqrt(
                self._variance(fi, qti, qsi, fi_detached_x) + _EPSILON
            )

        return stddev_sum**2

    def mask_iter(self, channels: torch.Tensor):
        if self.grouping_masks is None:
            for i in range(self.n_channels):
                yield channels == i
        else:
            for mask in self.grouping_masks:
                yield mask[channels]

    def __call__(self, name):
        func = getattr(self, name, None)
        if func is not None:
            return func
        raise NotImplementedError(
            f"The requested loss function {name} is not implemented. "
            + f"Allowed options are {self.divergences}."
        )
