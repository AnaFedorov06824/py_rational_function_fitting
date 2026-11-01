import unittest
from fractions import Fraction

from py_rational_function_fitting import (
    DegenerateFitError,
    FitError,
    RationalFunction,
    fit,
)


class FitBasicTests(unittest.TestCase):
    def test_recovers_constant_function(self):
        samples = [(Fraction(i), Fraction(3)) for i in range(4)]
        rf = fit(samples, 0, 0)
        self.assertEqual(rf(Fraction(0)), Fraction(3))
        self.assertEqual(rf(Fraction(7)), Fraction(3))

    def test_recovers_linear_over_linear(self):
        # (2x + 1) / (x + 1)
        samples = [
            (Fraction(0), Fraction(1)),
            (Fraction(1), Fraction(3, 2)),
            (Fraction(2), Fraction(5, 3)),
        ]
        rf = fit(samples, 1, 1)
        for x, y in samples:
            self.assertEqual(rf(x), y)

    def test_recovers_polynomial_with_zero_denominator_degree(self):
        # x**2 + 1, denominator degree 0 -> constant 1
        samples = [
            (Fraction(0), Fraction(1)),
            (Fraction(1), Fraction(2)),
            (Fraction(2), Fraction(5)),
        ]
        rf = fit(samples, 2, 0)
        self.assertEqual(rf.denominator, (Fraction(1),))
        self.assertEqual(rf(Fraction(3)), Fraction(10))

    def test_denominator_constant_term_is_one(self):
        samples = [
            (Fraction(0), Fraction(1, 2)),
            (Fraction(1), Fraction(2, 3)),
            (Fraction(2), Fraction(3, 4)),
        ]
        rf = fit(samples, 1, 1)
        self.assertEqual(rf.denominator[0], Fraction(1))

    def test_exact_rational_coefficients(self):
        # 1 / (2x + 1)
        samples = [
            (Fraction(0), Fraction(1)),
            (Fraction(1), Fraction(1, 3)),
            (Fraction(2), Fraction(1, 5)),
        ]
        rf = fit(samples, 0, 1)
        self.assertEqual(rf(Fraction(3)), Fraction(1, 7))


class FitErrorTests(unittest.TestCase):
    def test_too_few_samples_raises(self):
        with self.assertRaises(FitError):
            fit([(Fraction(0), Fraction(1))], 1, 1)

    def test_negative_degree_raises(self):
        with self.assertRaises(ValueError):
            fit([(Fraction(0), Fraction(1))], -1, 0)

    def test_singular_system_raises(self):
        # All samples at the same x: the rows are linearly dependent.
        samples = [
            (Fraction(1), Fraction(2)),
            (Fraction(1), Fraction(2)),
            (Fraction(1), Fraction(2)),
        ]
        with self.assertRaises(FitError):
            fit(samples, 1, 1)


class RationalFunctionTests(unittest.TestCase):
    def test_call_evaluates_exactly(self):
        rf = RationalFunction(
            [Fraction(1), Fraction(2)],  # 1 + 2x
            [Fraction(1), Fraction(3)],  # 1 + 3x
        )
        self.assertEqual(rf(Fraction(0)), Fraction(1))
        self.assertEqual(rf(Fraction(1)), Fraction(3, 4))

    def test_pole_raises_zero_division(self):
        rf = RationalFunction([Fraction(1)], [Fraction(0), Fraction(1)])
        with self.assertRaises(ZeroDivisionError):
            rf(Fraction(0))

    def test_empty_denominator_rejected(self):
        with self.assertRaises(DegenerateFitError):
            RationalFunction([Fraction(1)], [])

    def test_all_zero_denominator_rejected(self):
        with self.assertRaises(DegenerateFitError):
            RationalFunction([Fraction(1)], [Fraction(0), Fraction(0)])

    def test_equality_and_hash(self):
        a = RationalFunction([Fraction(1)], [Fraction(1), Fraction(1)])
        b = RationalFunction([Fraction(1)], [Fraction(1), Fraction(1)])
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_trim_removes_trailing_zeros(self):
        rf = fit(
            [
                (Fraction(0), Fraction(1)),
                (Fraction(1), Fraction(1)),
                (Fraction(2), Fraction(1)),
            ],
            2,
            0,
        )
        # Numerator should collapse to the constant 1 because higher terms
        # are exactly zero.
        self.assertEqual(rf.numerator, (Fraction(1),))


if __name__ == "__main__":
    unittest.main()
