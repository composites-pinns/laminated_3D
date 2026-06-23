"""Optimizer subpackage – custom optimizers for PINN training."""

from .nys_newton_cg import NysNewtonCG

__all__ = ["NysNewtonCG"]
