import math
from dataclasses import astuple, dataclass

import torch


@dataclass
class UnweightingMetrics:
    """
    Metrics for the unweighting performance

    Args:
        cut_eff: cut efficiency
        uweff_before_cuts: unweighting efficiency before cuts (computed as
            ``uweff_after_cuts * cut_eff``)
        uweff_before_cuts_partial: unweighting efficiency without over-weights before cuts (computed
            as ``uweff_after_cuts_partial * cut_eff``)
        uweff_after_cuts: unweighting efficiency after cuts
        uweff_after_cuts_partial: unweighting efficiency without over-weights after cuts
        over_weight_rate: fraction of over-weight samples
    """

    cut_eff: float
    uweff_before_cuts: float
    uweff_before_cuts_partial: float
    uweff_after_cuts: float
    uweff_after_cuts_partial: float
    over_weight_rate: float


def unweighting_metrics(
    weights: torch.Tensor,
    channels: torch.Tensor | None = None,
    channel_count: int | None = None,
    replica_count: int = 1000,
) -> UnweightingMetrics | tuple[UnweightingMetrics, list[UnweightingMetrics]]:
    """
    Calculate the unweighting efficiency as discussed in arXiv:2001.10028

    Args:
        weights: weights of the samples
        channels: channel indices of the samples
        channel_count: number of channels
        replica_count: number of replicas, called m in the reference
    Returns:
        An UnweightingMetrics object. In the multi-channel case, it also returns a list of
        UnweightingMetrics objects for all channels.
    """

    if channels is None:
        channel_count = 1
    uweffs = []
    integrals = []
    for channel in range(channel_count):
        w = weights if channel_count == 1 else weights[channels == channel]
        n_total = len(w)
        mask = w != 0.0
        w = w[mask]
        n_accepted = len(w)

        if n_accepted > 0:
            cut_efficiency = n_accepted / n_total
            integrals.append(w.mean().item() * cut_efficiency)
            sample = w[torch.randint(n_accepted, (replica_count, n_total))]
            s_max = sample.amax(dim=1)
            s_mean = sample.mean(dim=1)
            s_max_median = s_max.median()

            s_acc = torch.mean(s_mean / s_max_median).item()
            s_acc_partial = (sample / s_max_median).clip(max=1).mean().item()
            over_weight_rate = (sample > s_max_median).float().mean().item()
        else:
            integrals.append(0.0)
            s_acc = 0.0
            s_acc_partial = 0.0
            over_weight_rate = 0.0
            cut_efficiency = 0.0

        uweffs.append(
            UnweightingMetrics(
                cut_eff=cut_efficiency,
                uweff_before_cuts=s_acc * cut_efficiency,
                uweff_before_cuts_partial=s_acc_partial * cut_efficiency,
                uweff_after_cuts=s_acc,
                uweff_after_cuts_partial=s_acc_partial,
                over_weight_rate=over_weight_rate,
            )
        )

    integral_total = sum(integrals)
    if channel_count > 1:
        return (
            UnweightingMetrics(
                *(
                    sum(
                        integral / integral_total * metric
                        for integral, metric in zip(integrals, channel_metrics)
                    )
                    for channel_metrics in zip(*(astuple(uweff) for uweff in uweffs))
                )
            ),
            uweffs,
        )
    else:
        return uweffs[0]


@dataclass
class IntegrationMetrics:
    """
    Metrics for the integration performance

    Args:
        integral: total integration results
        count: number of integration samples
        error: Monte Carlo integration error
        rel_error: relative integration error
        rel_stddev: relative standard deviation (does not scale with number of samples)
        rel_stddev_opt: optimal relative standard deviation that would have been possible with
            stratified sampling
        channel_integrals: channel-wise integrals
        channel_counts: channel-wise number of samples
        channel_errors: channel-wise integration errors
        channel_rel_errors: channel-wise relative integration errors
        channel_rel_stddevs: channel-wise relative standard deviations
    """

    integral: float
    count: int
    error: float
    rel_error: float
    rel_stddev: float
    rel_stddev_opt: float
    channel_integrals: list[float]
    channel_counts: list[int]
    channel_errors: list[float]
    channel_rel_errors: list[float]
    channel_rel_stddevs: list[float]


def integration_metrics(
    channel_means: torch.Tensor,
    channel_variances: torch.Tensor,
    channel_counts: torch.Tensor,
) -> IntegrationMetrics:
    """
    Calculate metrics for the integration performance

    Args:
        channel_means: channel-wise integrals
        channel_variances: channel-wise variances
        channel_counts: channel-wise sample counts
    Returns:
        An IntegrationMetrics object
    """
    channel_square_errors = channel_variances / channel_counts
    integral = channel_means.sum().item()
    integral_abs = abs(integral)
    count = channel_counts.sum().item()
    error = channel_square_errors.nansum().sqrt().item()
    channel_stddevs = channel_variances.nan_to_num().sqrt()
    channel_errors = channel_square_errors.nan_to_num().sqrt()
    channel_means_abs = channel_means.abs()
    return IntegrationMetrics(
        integral=integral,
        count=count,
        error=error,
        rel_error=error / integral_abs,
        rel_stddev=error * math.sqrt(count) / integral_abs,
        rel_stddev_opt=(channel_stddevs.sum() * channel_stddevs).sum().sqrt().item()
        / integral,
        channel_counts=channel_counts.tolist(),
        channel_integrals=channel_means.tolist(),
        channel_errors=channel_errors.tolist(),
        channel_rel_errors=(channel_errors / channel_means_abs).tolist(),
        channel_rel_stddevs=(channel_stddevs / channel_means_abs).tolist(),
    )
