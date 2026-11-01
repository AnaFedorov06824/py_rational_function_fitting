# py_rational_function_fitting

Fits a rational function p(x)/q(x) to sample data using exact linear algebra over `fractions.Fraction`, with no third-party dependencies.

## Usage

```python
from fractions import Fraction
from py_rational_function_fitting import fit

samples = [
    (Fraction(0), Fraction(1)),
    (Fraction(1), Fraction(3, 2)),
    (Fraction(2), Fraction(5, 3)),
]
rf = fit(samples, numerator_degree=1, denominator_degree=1)
print(rf(Fraction(3)))  # 7/4
print(rf.numerator)     # (Fraction(1), Fraction(2))
print(rf.denominator)   # (Fraction(1), Fraction(1))
```

`fit` returns a `RationalFunction`, which is callable and exposes `.numerator` and `.denominator` as tuples of `Fraction` in ascending degree order. `FitError` is raised when the sample set does not determine a unique fit, and `DegenerateFitError` (a subclass of `FitError`) when the fitted denominator collapses to zero.

## Why this exists

Fitting a rational function to data is usually done with floating point least squares. That is fast, but the coefficients it returns are not reproducible: a different BLAS, a different order of operations, or a slightly different machine can shift the last bits of every coefficient, and near-degenerate systems can shift them a lot. For small, exactly-known samples — calibration curves, textbook exercises, symbolic derivations — you often want the exact answer, or a clear failure when the samples do not pin one down.

This library builds the linear system p(x_i) = y_i q(x_i) directly from the samples, solves it with exact Gauss-Jordan elimination over `Fraction`, and returns the unique rational function. The trade-off is speed and scale: exact arithmetic on `Fraction` is much slower than floating point, and the solver is dense and O(n^3). It is meant for tens of samples, not thousands.

## Normalisation

The pair (p, q) is only defined up to a non-zero scalar multiple. This library fixes the scaling by requiring the denominator's constant term to be 1. That makes the returned object canonical, so two fits from the same data compare equal. If your true denominator has no constant term, the fit will still succeed as long as the system is non-singular, but the returned coefficients will be scaled so that the constant term is 1 — the function value is unchanged.

## The awkward edge

If you pass fewer samples than `numerator_degree + denominator_degree + 1`, or if the samples are redundant (for example, all at the same x), `fit` raises `FitError` rather than returning a best-effort answer. That is deliberate: an exact fit that is not unique is not an exact fit. If you need a least-squares answer for noisy data, this library is the wrong tool.
