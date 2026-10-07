"""Independent magnetic-sublevel normalization for the shared rate kernel."""

import pytest
from sympy import Rational
from sympy.physics.wigner import wigner_3j

from atomx.constants import ALPHA, AU_RATE_PER_S
from atomx.properties.transitions.rates import einstein_A_eL, einstein_A_prefactor


@pytest.mark.parametrize("rank,factor", [(1, 4 / 3), (2, 1 / 15), (3, 8 / 4725)])
def test_analytic_multipole_prefactors(rank, factor):
    omega = 0.375
    expected = factor * (ALPHA * omega) ** (2 * rank + 1) * AU_RATE_PER_S
    assert einstein_A_prefactor(omega, rank) == pytest.approx(expected, rel=2e-15)


@pytest.mark.parametrize("initial_j2,final_j2,rank", [(1, 1, 1), (3, 1, 1), (3, 5, 2), (4, 2, 1)])
def test_fixed_magnetic_strength_matches_reduced_rate(initial_j2, final_j2, rank):
    # Reference is SymPy's Wigner-Eckart sum, independent of atomx angular code.
    reduced_amplitude, omega = -0.7, 0.375
    for initial_m2 in range(-initial_j2, initial_j2 + 1, 2):
        strength = sum(
            float(wigner_3j(
                Rational(final_j2, 2), rank, Rational(initial_j2, 2),
                -Rational(final_m2, 2), q, Rational(initial_m2, 2),
            )) ** 2 * reduced_amplitude ** 2
            for final_m2 in range(-final_j2, final_j2 + 1, 2)
            for q in range(-rank, rank + 1)
        )
        assert strength == pytest.approx(reduced_amplitude ** 2 / (initial_j2 + 1), rel=2e-15)
        assert einstein_A_prefactor(omega, rank) * strength == pytest.approx(
            einstein_A_eL(omega, reduced_amplitude ** 2, rank, initial_j2), rel=2e-15)
