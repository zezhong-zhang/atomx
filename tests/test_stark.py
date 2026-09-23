"""External-field perturbative Stark + Stark-Zeeman tests.

Unit conversions are derived from the shipped CODATA table instead of
retyping literals (review M12 class):

* ``AU_FIELD_VPM = HARTREE_EV / BOHR_RADIUS_M`` — the atomic unit of
  electric field is ``E_h / (e a_0)`` in V/m;
* ``HARTREE_MHZ = AU_RATE_PER_S / (2 pi) / 1e6`` — ``nu = E/h``, and
  ``AU_RATE_PER_S = 1/tau_0 = E_h/hbar``.

Only the *measured* field (1e7 V/m) and the literature polarisability
(alpha(H 1s) = 4.5 a_0^3, exact) stay as literals.
"""

import numpy as np
import pytest

from atomx.constants import AU_RATE_PER_S, BOHR_RADIUS_M, HARTREE_EV
from atomx.properties.stark import (
    StarkZeemanShift,
    linear_stark_shift_hydrogenic,
    quadratic_stark_shift,
    stark_zeeman_table,
)

#: 1 a.u. of electric field, V/m (E_h / (e a_0)).
AU_FIELD_VPM = HARTREE_EV / BOHR_RADIUS_M
#: 1 hartree expressed as a frequency, MHz (nu = E/h).
HARTREE_MHZ = AU_RATE_PER_S / (2.0 * np.pi) / 1e6


def test_linear_stark_h_n2_textbook():
    """H n=2 splitting at F=1e7 V/m: ΔE = (3/2)·n·k·F·a₀ → ±384 GHz."""
    dE_plus = linear_stark_shift_hydrogenic(n=2, k=+1, F_Vpm=1e7, Z=1)
    dE_minus = linear_stark_shift_hydrogenic(n=2, k=-1, F_Vpm=1e7, Z=1)
    assert dE_plus == pytest.approx(-dE_minus, rel=1e-12)
    # 3/2·2·1·1e7 V/m, converted to atomic units and then to MHz.
    expected = 1.5 * 2 * 1 * (1e7 / AU_FIELD_VPM) * HARTREE_MHZ
    assert dE_plus == pytest.approx(expected, rel=1e-6)


def test_linear_stark_zero_at_k_zero():
    """k=0 sublevel of any n is unshifted at first order."""
    assert linear_stark_shift_hydrogenic(n=3, k=0, F_Vpm=1e8) == 0.0


def test_linear_stark_invalid():
    with pytest.raises(ValueError):
        linear_stark_shift_hydrogenic(n=0, k=0, F_Vpm=1e8)
    with pytest.raises(ValueError):
        linear_stark_shift_hydrogenic(n=2, k=2, F_Vpm=1e8)  # |k| not < n


def test_linear_stark_z_scaling():
    """Linear Stark scales as 1/Z (a₀ → a₀/Z for Z⁺-like ions)."""
    dE_h = linear_stark_shift_hydrogenic(n=2, k=1, F_Vpm=1e7, Z=1)
    dE_he = linear_stark_shift_hydrogenic(n=2, k=1, F_Vpm=1e7, Z=2)
    assert dE_he == pytest.approx(dE_h / 2.0, rel=1e-12)


def test_quadratic_stark_negative_for_alpha_pos():
    """Positive polarizability → negative shift at any F (force on dipole)."""
    dE = quadratic_stark_shift(alpha_a0_cubed=4.5, F_Vpm=1e6)
    assert dE < 0


def test_quadratic_stark_h_ground():
    """H 1s quadratic Stark shift is exactly −½·α·F² in MHz.

    α(H 1s) = 4.5 a₀³ is the exact hydrogenic value (the same number gated
    first-principles in ``test_review_m9_m13_gates.py``), so the shift is
    fixed by the closed form with no fit:

        ΔE = −½ · 4.5 · (1e7 V/m / 5.14220674763e11 V/m)² Ha
           = −5.5987 MHz.

    Measured 2026-09-25 at HEAD: −5.59871678619299 MHz.  The gate is the
    closed form at rel=1e-6; the residual 4.2e-8 is the two CODATA-2018
    conversion literals still hard-coded in ``stark.py`` (``_AU_FIELD_VPM``
    = 5.14220674763e11, ``2.41798934e8`` MHz/eV) against the shipped
    CODATA-2022 defaults.  The previous gate was ``−10 < dE < 0``, which
    admitted any small number (including −1e-9).
    """
    dE = quadratic_stark_shift(alpha_a0_cubed=4.5, F_Vpm=1e7)
    F_au = 1e7 / AU_FIELD_VPM
    expected = -0.5 * 4.5 * F_au * F_au * HARTREE_MHZ
    assert pytest.approx(expected, rel=1e-6) == dE, (
        f"quadratic Stark: measured {dE!r} MHz vs exact -alpha F^2/2 "
        f"{expected!r} MHz")


def test_quadratic_stark_F_squared_scaling():
    """ΔE ∝ F² — doubling F quadruples the shift."""
    dE_1 = quadratic_stark_shift(4.5, 1e6)
    dE_2 = quadratic_stark_shift(4.5, 2e6)
    assert dE_2 / dE_1 == pytest.approx(4.0, rel=1e-12)


def test_stark_zeeman_table_structure():
    """One entry per (m_J, B, F) tuple."""
    B = np.array([0.5, 1.0])
    F = np.array([0.0, 1e6])
    tbl = stark_zeeman_table(gJ=2.0, J=1.5,
                              alpha_a0_cubed=10.0,
                              B_Tesla_grid=B, F_Vpm_grid=F)
    n_m = int(2 * 1.5 + 1)   # m = -3/2..+3/2 = 4 values
    assert len(tbl) == n_m * len(B) * len(F)
    for entry in tbl:
        assert isinstance(entry, StarkZeemanShift)


def test_stark_zeeman_F_zero_recovers_pure_zeeman():
    """At F = 0, total shift = pure Zeeman."""
    tbl = stark_zeeman_table(gJ=2.0, J=0.5,
                              alpha_a0_cubed=10.0,
                              B_Tesla_grid=np.array([1.0]),
                              F_Vpm_grid=np.array([0.0]))
    for entry in tbl:
        assert entry.delta_E_stark_MHz == 0.0
        assert entry.delta_E_total_MHz == entry.delta_E_zeeman_MHz


def test_stark_zeeman_B_zero_recovers_pure_stark():
    """At B = 0, total shift = pure Stark."""
    tbl = stark_zeeman_table(gJ=2.0, J=0.5,
                              alpha_a0_cubed=10.0,
                              B_Tesla_grid=np.array([0.0]),
                              F_Vpm_grid=np.array([1e7]))
    for entry in tbl:
        assert entry.delta_E_zeeman_MHz == 0.0
        assert entry.delta_E_total_MHz == entry.delta_E_stark_MHz
