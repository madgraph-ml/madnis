"""
This module containes functions and classes that implement the different types of neural network
architectures necessary for (multi-channel) neural importance sampling.
"""

from .discrete_made import DiscreteMADE
from .discrete_transformer import DiscreteTransformer
from .flow import Distribution, Flow
from .masked_mlp import MaskedMLP
from .mixed_flow import MixedFlow
from .mlp import MLP, StackedMLP
from .splines import rational_quadratic_spline, unconstrained_rational_quadratic_spline

__all__ = [
    "unconstrained_rational_quadratic_spline",
    "rational_quadratic_spline",
    "MLP",
    "StackedMLP",
    "Flow",
    "Distribution",
    "MaskedMLP",
    "DiscreteMADE",
    "DiscreteTransformer",
    "MixedFlow",
]
