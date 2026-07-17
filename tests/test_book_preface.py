"""Executable claims from the book's preface (docs/book/vol1/frontmatter/preface.md).

The preface promises: "The book has a test suite, and it fails when a claim
fails to verify — starting with the claims in this preface."  This file is
that gate for the preface.  Run:  pytest -m book

Companion script: docs/book/vol1/frontmatter/calc/preface_numbers.py
"""
import numpy as np
import pytest

pytestmark = pytest.mark.book


def test_hydrogen_dirac_1s_eleven_digits():
    # Preface: "an engine that cannot reproduce the analytic Dirac ground
    # state to eleven digits has no business computing tungsten."
    # Both engines must deliver the point-nucleus H 1s eigenvalue to 1e-11
    # relative against the Sommerfeld formula.
    from atomx import Atom
    from atomx.constants import ALPHA, C

    exact = C ** 2 * (np.sqrt(1.0 - ALPHA ** 2) - 1.0)  # -0.5000066565953 Eh
    for numerics in ("bspline", "shooting"):
        # The claim is against the PURE analytic Dirac eigenvalue —
        # run correction-free (breit/qed default True in Atom).
        result = Atom(Z=1, nuclear="point", numerics=numerics,
                      breit=False, qed=False).solve()
        e = result.orbital_energies["1s1/2"]
        assert abs(e / exact - 1.0) < 1e-11, numerics


def test_bound_g_factor_few_parts_in_ten_billion():
    # Preface: "a bound electron's magnetic moment, computed and measured,
    # agrees to a few parts in ten billion."
    # g(28Si13+) exp: Sturm et al., PRL 107, 023002 (2011):
    #   1.995 348 958 7 (5)(3)(8)
    # theory quoted in the same comparison:
    #   1.995 348 958 0 (17)
    g_exp = 1.9953489587
    g_theory = 1.9953489580
    rel = abs(g_exp - g_theory) / g_theory
    assert 1e-10 < rel < 1e-9  # "a few parts in ten billion", not one, not ten


def test_optical_clock_nineteenth_digit():
    # Preface: "optical clocks uncertain only in the nineteenth digit."
    # Aeppli et al., PRL 133, 023401 (2024): 8.1e-19 total systematic (Sr).
    u_sys = 8.1e-19
    assert 1e-19 <= u_sys < 1e-18


@pytest.mark.slow
def test_tungsten_same_call():
    # Preface: "The call that solves hydrogen solves tungsten unchanged."
    from atomx import Atom

    result = Atom(Z=74).solve()
    assert result.converged
    assert len(result.subshells) > 0
