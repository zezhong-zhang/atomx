"""Smoke tests for second-order MBPT on a closed-shell DHF reference."""

from __future__ import annotations

import math

import pytest

from atomx import Atom
from atomx.methods.mbpt import (
    MBPT2Result,
    _allowed_k_range,
    mbpt2_energy,
    reduced_C,
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
    atom = Atom(Z=2, config="1s2", engine="bspline", corrections="none")
    atom.solve()
    res = atom._result

    r_small = mbpt2_energy(res, kappa_max=2, n_virt_max=15, multipole_max=2)
    r_large = mbpt2_energy(res, kappa_max=3, n_virt_max=25, multipole_max=4)

    assert isinstance(r_small, MBPT2Result)
    assert r_small.correlation_energy < 0.0
    assert r_large.correlation_energy < r_small.correlation_energy  # more negative
    # Physical ceiling: MP2 on a HF reference cannot over-correlate past the
    # exact correlation energy (-0.042044 Ha for He) — the total must stay
    # above the exact ground state (-2.903724 Ha).
    assert r_large.correlation_energy > -0.042044, (
        "MBPT(2) correlation exceeds the exact He correlation energy of "
        "-0.042044 Ha; the energy denominator is mixing two Hamiltonians"
    )
    # This truncated basis (kappa_max=3, 25 virtuals/kappa) recovers a few
    # mHa of the converged -0.036968 Ha; re-pinned 2026-09-19 when the
    # virtuals moved from V^{N-1} to V^N (see mbpt.build_virtual_basis).
    assert r_large.correlation_energy < -0.003


@pytest.mark.slow
def test_mbpt2_helium_matches_the_literature_value():
    """He MBPT(2) must reproduce the accepted correlation energy.

    Reviewed 2026-09-19: the virtuals were built in the V^{N-1} potential
    while the occupied ε came from the N-electron SCF, so the energy
    denominator mixed two Hamiltonians.  That gave E2 = -0.0496 Ha — 118 %
    of the exact correlation energy -0.042044 Ha — and a total energy
    8 mHa *below* the exact ground state, which is impossible for MP2 on a
    HF reference.  With V^N virtuals E2 = -0.036968 Ha, within 0.7 % of the
    literature MP2/CBS value (-0.0372 Ha; JCP 102, 6168 (1995);
    JCP 112, 10746 (2000)) and correctly bracketed by
    HF (-2.861680) < E_tot < exact (-2.903724).

    See docs/archive/2026-09/PHYSICS_NUMERICS_REVIEW_2026-09-19.md finding H4/M6.
    """
    atom = Atom(Z=2, config="1s2", engine="bspline", corrections="none")
    atom.solve()
    res = atom._result
    r = mbpt2_energy(res, kappa_max=4)

    assert r.correlation_energy == pytest.approx(-0.036968, abs=5e-4)
    e_tot = res.total_energy + r.correlation_energy
    assert -2.903724 < e_tot < -2.861680, (
        f"MBPT(2) total {e_tot:.6f} Ha is outside the HF/exact bracket"
    )
    # Not merely inside the bracket: within 1 % of the literature value.
    assert r.correlation_energy == pytest.approx(-0.0372, rel=0.01)


def test_mbpt2_open_shell_rejected():
    """Open-shell references trigger a clear error."""
    atom = Atom(Z=3, config="1s2 2s1", engine="bspline", corrections="none")
    atom.solve()
    res = atom._result
    with pytest.raises(ValueError, match="closed-shell"):
        mbpt2_energy(res, kappa_max=1, n_virt_max=5)
