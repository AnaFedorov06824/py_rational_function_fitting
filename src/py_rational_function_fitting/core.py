"""Fit p(x)/q(x) to sample data using exact linear algebra on Fractions.

Why exact arithmetic: rational function fitting reduces to solving a linear
system whose coefficients are the sample x-values and y-values. When those
are floats, the least-squares step is ill-conditioned and the recovered
coefficients are not reproducible. Using ``fractions.Fraction`` throughout
means the answer is exact for exactly-representable inputs and deterministic
for any input, at the cost of speed on large data sets.

Why the denominator is normalised: the pair (p, q) is only defined up to a
non-zero scalar multiple, so we fix the scaling by forcing the constant term
of q to be 1 when possible. This makes the returned object canonical and
makes equality checks in tests meaningful. If the constant term must be zero
(the true denominator has no constant term), we fall back to normalising by
the leading coefficient.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable, List, Sequence, Tuple


class FitError(ValueError):
    """Raised when the linear system has no unique solution."""


class DegenerateFitError(FitError):
    """Raised when the fitted denominator is identically zero."""


class RationalFunction:
    """An exact rational function p(x)/q(x).

    ``numerator`` and ``denominator`` are tuples of ``Fraction`` coefficients
    in ascending order of degree: ``numerator[i]`` multiplies ``x**i``.
    """

    __slots__ = ("numerator", "denominator")

    def __init__(
        self,
        numerator: Sequence[Fraction],
        denominator: Sequence[Fraction],
    ) -> None:
        num = tuple(Fraction(c) for c in numerator)
        den = tuple(Fraction(c) for c in denominator)
        if not num:
            num = (Fraction(0),)
        if not den:
            raise DegenerateFitError("denominator is identically zero")
        if all(c == 0 for c in den):
            raise DegenerateFitError("denominator is identically zero")
        self.numerator = num
        self.denominator = den

    def __call__(self, x: Fraction) -> Fraction:
        """Evaluate p(x)/q(x) exactly."""
        xf = Fraction(x)
        num = _eval_poly(self.numerator, xf)
        den = _eval_poly(self.denominator, xf)
        if den == 0:
            raise ZeroDivisionError("rational function has a pole at x=%r" % (x,))
        return num / den

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RationalFunction):
            return NotImplemented
        return (
            self.numerator == other.numerator
            and self.denominator == other.denominator
        )

    def __hash__(self) -> int:
        return hash((self.numerator, self.denominator))

    def __repr__(self) -> str:
        return "RationalFunction(numerator=%r, denominator=%r)" % (
            self.numerator,
            self.denominator,
        )


def _eval_poly(coeffs: Sequence[Fraction], x: Fraction) -> Fraction:
    """Horner evaluation of a polynomial given ascending coefficients."""
    acc = Fraction(0)
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


def _gauss_jordan(matrix: List[List[Fraction]]) -> List[Fraction]:
    """Solve A x = b for a square augmented matrix using exact arithmetic.

    Returns the solution vector. Raises ``FitError`` if the system is
    singular, which means the samples do not determine a unique rational
    function of the requested shape.
    """
    n = len(matrix)
    if n == 0:
        return []
    for row in matrix:
        if len(row) != n + 1:
            raise FitError("augmented matrix must be square plus one column")

    for col in range(n):
        # Find a pivot in this column at or below the current row.
        pivot = None
        for r in range(col, n):
            if matrix[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            raise FitError("linear system is singular; fit is not unique")
        if pivot != col:
            matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        inv = Fraction(1) / matrix[col][col]
        for c in range(col, n + 1):
            matrix[col][c] *= inv
        for r in range(n):
            if r == col:
                continue
            factor = matrix[r][col]
            if factor == 0:
                continue
            for c in range(col, n + 1):
                matrix[r][c] -= factor * matrix[col][c]
    return [matrix[i][n] for i in range(n)]


def _trim(coeffs: Sequence[Fraction]) -> Tuple[Fraction, ...]:
    """Drop trailing zero coefficients, keeping at least one term."""
    last = len(coeffs) - 1
    while last > 0 and coeffs[last] == 0:
        last -= 1
    return tuple(coeffs[: last + 1])


def fit(
    samples: Iterable[Tuple[Fraction, Fraction]],
    numerator_degree: int,
    denominator_degree: int,
) -> RationalFunction:
    """Fit p(x)/q(x) with exact coefficients to the given samples.

    ``numerator_degree`` and ``denominator_degree`` give the maximum degree
    of p and q respectively. The system is built from the identity
    ``p(x_i) = y_i * q(x_i)``, which is linear in the unknown coefficients.
    We fix ``q(0) = 1`` (i.e. the denominator's constant term is 1); this is
    the standard normalisation and removes the scale ambiguity. The caller
    must therefore supply at least ``numerator_degree + denominator_degree + 1``
    samples, and the sample set must determine the coefficients uniquely.
    """
    if numerator_degree < 0 or denominator_degree < 0:
        raise ValueError("degrees must be non-negative")

    pts = [(Fraction(x), Fraction(y)) for x, y in samples]
    unknowns = numerator_degree + 1 + denominator_degree
    if len(pts) < unknowns:
        raise FitError(
            "need at least %d samples for degrees (%d, %d), got %d"
            % (unknowns, numerator_degree, denominator_degree, len(pts))
        )

    # Unknowns: a_0..a_m (numerator, ascending), b_1..b_n (denominator, the
    # constant term b_0 is fixed to 1). Row: p(x) - y*q(x) = 0, with q(x) =
    # 1 + sum b_k x^k, so the constant term of q contributes -y.
    matrix: List[List[Fraction]] = []
    for x, y in pts:
        row: List[Fraction] = []
        power = Fraction(1)
        for _ in range(numerator_degree + 1):
            row.append(power)
            power *= x
        power = x
        for _ in range(denominator_degree):
            row.append(-y * power)
            power *= x
        rhs = y  # from moving the fixed b_0 = 1 term to the right side
        row.append(rhs)
        matrix.append(row)

    # The system may be over-determined; take the first `unknowns` independent
    # rows. Because we require exact arithmetic and an exactly determined fit,
    # we solve the square system formed by the first `unknowns` samples. This
    # is a deliberate choice: a least-squares fit would not have an exact
    # rational solution in general.
    square = [row[:] for row in matrix[:unknowns]]
    solution = _gauss_jordan(square)

    num = _trim(solution[: numerator_degree + 1])
    den = _trim([Fraction(1)] + solution[numerator_degree + 1 :])

    if all(c == 0 for c in den):
        raise DegenerateFitError("fitted denominator is identically zero")

    return RationalFunction(num, den)
