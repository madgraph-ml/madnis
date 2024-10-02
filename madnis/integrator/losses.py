# Implement loss functions for the integrator model

from functools import wraps

import torch

_EPSILON = 1e-16
