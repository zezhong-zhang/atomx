"""Smoke tests for second-order MBPT on a closed-shell DHF reference."""

from __future__ import annotations

import math

import pytest

from atomx import Atom
from atomx.methods.mbpt import (
    MBPT2Result,
    mbpt2_energy,
    reduced_C,
    _allowed_k_range,
)


# ── Reduced matrix-element sanity checks ────────────────────────────────


def test_reduced_C_diagonal_s():
    """⟨s_{1/2} ‖ C^0 ‖ s_{1/2}⟩ = -√2 (Grant phase convention)."""
    val = reduced_C(-1, -1, 0)
    assert math.isclose(val, -math.sqrt(2.0), abs_tol=1e-12)


def test_reduced_C_dipole_s_to_p32():
    """⟨s_{1/2} ‖ C^1 ‖ p_{3/2}⟩ = 2/√3."""
    val = reduced_C(-1, -2, 1)
    assert math.isclose(val, 2.0 / math.sqrt(3.0), abs_tol=1e-12)


def test_reduced_C_parity_violation():
    """Parity-forbidden multipoles return exactly zero."""
    # ⟨s_{1/2} ‖ C^1 ‖ s_{1/2}⟩ violates parity (l_a+k+l_b = 1, odd).
    assert reduced_C(-1, -1, 1) == 0.0


def test_allowed_k_range_s_to_p():
    assert _allowed_k_range(-1, -2) == [1]


def test_allowed_k_range_d_to_d():
    # d_{3/2}↔d_{5/2}: triangle |3/2-5/2|..|3/2+5/2|=1..4, parity even → 2,4
    ks = _allowed_k_range(-3, +2)
    assert ks == [2, 4]


# ── End-to-end MBPT(2) on a tiny closed-shell atom ──────────────────────


@pytest.mark.slow
def test_mbpt2_helium_negative_and_decreasing_with_basis():
    """He MBPT(2) is negative and converges (in magnitude) as the
    partial-wave basis grows — characteristic atomic MP2 behaviour.

    Marked slow because each evaluation builds a virtual basis per κ.
    """
    atom = Atom(Z=2, config="1s2", breit=False, qed=False)
    atom.solve()
    res = atom._result

    r_small = mbpt2_energy(res, kappa_max=2, n_virt_max=15, multipole_max=2)
    r_large = mbpt2_energy(res, kappa_max=3, n_virt_max=25, multipole_max=4)

    assert isinstance(r_small, MBPT2Result)
    assert r_small.correlation_energy < 0.0
    assert r_large.correlation_energy < r_small.correlation_energy  # more negative
    # Sanity bound: He MP2 must be > -0.05 Ha for any reasonable basis
    assert r_large.correlation_energy > -0.05
    # And the partial-wave tail is at most a few times the converged value
    assert r_large.correlation_energy < -0.005


def test_mbpt2_open_shell_rejected():
    """Open-shell references trigger a clear error."""
    atom = Atom(Z=3, config="1s2 2s1", breit=False, qed=False)
    atom.solve()
    res = atom._result
    with pytest.raises(ValueError, match="closed-shell"):
        mbpt2_energy(res, kappa_max=1, n_virt_max=5)
