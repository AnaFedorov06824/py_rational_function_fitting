"""Rational function fitting over exact rational arithmetic.

Public API:
    fit(samples, numerator_degree, denominator_degree) -> RationalFunction
    RationalFunction: callable rational function with .numerator/.denominator
    FitError: raised when the linear system has no unique solution
    DegenerateFitError: raised when the denominator is identically zero
"""

from .core import (
    DegenerateFitError,
    FitError,
    RationalFunction,
    fit,
)

__all__ = [
    "DegenerateFitError",
    "FitError",
    "RationalFunction",
    "fit",
]
