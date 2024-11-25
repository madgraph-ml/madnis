"""
This module contains functions and classes to train neural importance sampling networks and
evaluate the integration and sampling performance.
"""

from .buffer import Buffer
from .channel_grouping import ChannelData, ChannelGroup, ChannelGrouping
from .integrand import Integrand
from .integrator import Integrator, SampleBatch, TrainingStatus
from .losses import (
    kl_divergence,
    multi_channel_loss,
    rkl_divergence,
    stratified_variance,
)
from .metrics import (
    IntegrationMetrics,
    UnweightingMetrics,
    integration_metrics,
    unweighting_metrics,
)
from .vegas_pretraining import VegasPreTraining

__all__ = [
    "Integrator",
    "TrainingStatus",
    "SampleBatch",
    "Integrand",
    "Buffer",
    "multi_channel_loss",
    "stratified_variance",
    "kl_divergence",
    "rkl_divergence",
    "ChannelGroup",
    "ChannelData",
    "ChannelGrouping",
    "UnweightingMetrics",
    "unweighting_metrics",
    "IntegrationMetrics",
    "integration_metrics",
    "VegasPreTraining",
]
