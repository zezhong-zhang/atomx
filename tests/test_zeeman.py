"""Tests for M1.6 — Zeeman / Landé g-factors.

Key validation targets
----------------------
* Non-relativistic Landé g against textbook values for s/p/d orbitals.
* Bound-Dirac g(H 1s) = (2/3)(1 + 2γ); for Z = 1 this is
  ``1.999964503…``  (γ = √(1 − α²)).
* Bound-electron g(H 1s) including Schwinger anomaly — matches
  CODATA ``2.002283845…`` to sub-ppm.
* ²⁸Si¹³⁺ Penning-trap reference (Sturm *et al.* 2014,
  ``g = 1.995348958``) — atomx lands to ~15 ppm without any further
  QED binding corrections.
"""

from __future__ import annotations

import numpy as np
import pytest

from atomx.constants import ALPHA
from atomx.scf.solver import DFSolver, Subshell
from atomx.zeeman import (
    BOHR_MAGNETON_MHZ_PER_TESLA,
    LandeFactor,
    ZeemanShift,
    lande_gJ_bound_single_electron,
    lande_gJ_hydrogenic,
    lande_gJ_nr,
    weak_field_shift_MHz,
    zeeman_shift_table,
)
from atomx.zeeman.lande import lande_factor


# ---------------------------------------------------------------------------
# Non-relativistic closed form
# ---------------------------------------------------------------------------

class TestLandeNR:
    @pytest.mark.parametrize(
        "l, j, expected",
        [
            (0, 0.5, 2.0),           # ²S₁/₂
            (1, 0.5, 2.0 / 3.0),     # ²P₁/₂
            (1, 1.5, 4.0 / 3.0),     # ²P₃/₂
            (2, 1.5, 4.0 / 5.0),     # ²D₃/₂
            (2, 2.5, 6.0 / 5.0),     # ²D₅/₂
            (3, 2.5, 6.0 / 7.0),     # ²F₅/₂
            (3, 3.5, 8.0 / 7.0),     # ²F₇/₂
        ],
    )
    def test_textbook_values(self, l, j, expected):
        assert lande_gJ_nr(l=l, j=j) == pytest.approx(expected, rel=1e-14)

    def test_rejects_nonpositive_j(self):
        with pytest.raises(ValueError, match="j must be positive"):
            lande_gJ_nr(l=0, j=0)


# ---------------------------------------------------------------------------
# Hydrogenic Dirac
# ---------------------------------------------------------------------------

class TestLandeDirac:
    def test_h_1s_matches_breit_formula(self):
        """For κ = −1, Z = 1: g = (2/3)(1 + 2γ), γ = √(1−α²)."""
        gamma = np.sqrt(1.0 - ALPHA ** 2)
        expected = (2.0 / 3.0) * (1.0 + 2.0 * gamma)
        g = lande_gJ_hydrogenic(kappa=-1, Z=1)
        assert g == pytest.approx(expected, rel=1e-14)

    def test_h_1s_leading_order_expansion(self):
        """g(H 1s) ≈ 2 − (2/3)(Zα)²  at leading order."""
        g = lande_gJ_hydrogenic(kappa=-1, Z=1)
        expected_leading = 2.0 - (2.0 / 3.0) * ALPHA ** 2
        # Both should be close to 2 minus a small quantity; the Dirac
        # exact vs leading-order differ at O((Zα)⁴).
        assert g == pytest.approx(expected_leading, rel=1e-7)

    def test_heavy_ion_reduction(self):
        """As Zα grows, g(s₁/₂) decreases.  For U⁹¹⁺ (Z=92) it drops
        well below 2."""
        g = lande_gJ_hydrogenic(kappa=-1, Z=92)
        assert g < 1.8
        # Outside strong-field diverge: (Zα)² ≈ 0.45 for U
        assert g > 1.0

    def test_rejects_supercritical_Z(self):
        """At Zα ≥ 1 no bound state for j = 1/2 → ValueError."""
        # Z ≈ 137 corresponds to Zα = 1
        with pytest.raises(ValueError, match="bound-state"):
            lande_gJ_hydrogenic(kappa=-1, Z=140)

    def test_p3_2_falls_back_to_nr(self):
        """κ = −2 (p3/2) uses LS value for now (generic Dirac
        correction not yet verified against tables)."""
        g = lande_gJ_hydrogenic(kappa=-2, Z=1)
        assert g == pytest.approx(4.0 / 3.0, rel=1e-14)


# ---------------------------------------------------------------------------
# Bound electron with anomalous moment (Penning-trap-level accuracy)
# ---------------------------------------------------------------------------

class TestBoundElectronG:
    def test_h_1s_matches_codata(self):
        """Dirac × (1 + a_e) for H 1s → 2.002283762, CODATA 2.002283845."""
        g = lande_gJ_bound_single_electron(kappa=-1, Z=1)
        assert g == pytest.approx(2.002283845, abs=2e-7)

    def test_si13_plus_landmark_measurement(self):
        """Sturm *et al.* 2014, Nature 506, 467 — g(²⁸Si¹³⁺) = 1.995348958(5).

        atomx reproduces this to ~15 ppm from Dirac + a_e alone; the
        residual is higher-order QED binding (two-loop + recoil QED)
        not yet modelled.
        """
        g = lande_gJ_bound_single_electron(kappa=-1, Z=14)
        # Stringent band: 30 ppm
        assert abs(g - 1.995348958) / 1.995348958 < 30e-6

    def test_anomaly_off_returns_dirac(self):
        g_bound = lande_gJ_bound_single_electron(
            kappa=-1, Z=1, include_anomalous_moment=False,
        )
        g_dirac = lande_gJ_hydrogenic(kappa=-1, Z=1)
        assert g_bound == pytest.approx(g_dirac, rel=1e-14)


# ---------------------------------------------------------------------------
# Landé-factor dataclass wrapper
# ---------------------------------------------------------------------------

class TestLandeFactor:
    def test_h_1s_all_three_rungs(self):
        """For H (low Z), ordering is  dirac < nr < bound:  the Breit
        reduction (2 − (2/3)(Zα)²) is overwhelmed by the +a_e Schwinger
        anomaly.  For Z ≳ 8 the ordering flips and bound < nr."""
        lf = lande_factor(kappa=-1, Z=1, label="1s1/2")
        assert isinstance(lf, LandeFactor)
        assert lf.nr == pytest.approx(2.0)
        assert lf.dirac == pytest.approx(
            (2.0 / 3.0) * (1.0 + 2.0 * np.sqrt(1.0 - ALPHA ** 2)), rel=1e-14,
        )
        assert lf.dirac < lf.nr       # Breit reduction
        assert lf.bound > lf.dirac    # (1 + a_e) boost
        assert lf.bound > lf.nr       # at Z = 1 the anomaly wins

    def test_si13_rung_ordering(self):
        """For Z = 14 the Breit reduction overcomes the Schwinger anomaly,
        so  dirac < bound < nr  (measurable in Penning-trap experiments)."""
        lf = lande_factor(kappa=-1, Z=14, label="1s1/2")
        assert lf.dirac < lf.bound < lf.nr


# ---------------------------------------------------------------------------
# Weak-field Zeeman shift
# ---------------------------------------------------------------------------

class TestZeemanShift:
    def test_bohr_magneton_mhz_per_tesla(self):
        """μ_B/h ≈ 1.39962 × 10⁴ MHz/T — CODATA 2018."""
        assert BOHR_MAGNETON_MHZ_PER_TESLA == pytest.approx(13996.24, abs=0.1)

    def test_weak_field_shift_single(self):
        """For g=2, m_J=1/2, B=1 T:  ΔE = μ_B × 1 T ≈ 13996 MHz."""
        shift = weak_field_shift_MHz(gJ=2.0, m_j=0.5, B_Tesla=1.0)
        assert shift == pytest.approx(13996.24, abs=0.1)

    def test_zeeman_shift_table_s_state(self):
        """²S₁/₂ at B = 1 T should give ±(g μ_B / 2) × B ≈ ±13996 MHz."""
        lf = lande_factor(kappa=-1, Z=1, label="1s1/2")
        z = zeeman_shift_table(lf, B_Tesla=1.0, g_source="bound")
        assert isinstance(z, ZeemanShift)
        assert set(z.shifts_MHz.keys()) == {-0.5, +0.5}
        assert z.shifts_MHz[+0.5] > 0
        assert z.shifts_MHz[-0.5] == pytest.approx(-z.shifts_MHz[+0.5], rel=1e-14)
        assert z.splitting_MHz == pytest.approx(z.shifts_MHz[0.5] * 2, rel=1e-14)

    def test_zeeman_shift_table_p3_2(self):
        """p3/2 has four sub-levels m_J ∈ {−3/2, −1/2, +1/2, +3/2}."""
        lf = lande_factor(kappa=-2, Z=1, label="2p3/2")
        z = zeeman_shift_table(lf, B_Tesla=0.1, g_source="nr")
        expected_m = {-1.5, -0.5, 0.5, 1.5}
        assert set(z.shifts_MHz.keys()) == expected_m
        # Shifts are linear in m_J
        shift_12 = z.shifts_MHz[0.5]
        shift_32 = z.shifts_MHz[1.5]
        assert shift_32 == pytest.approx(3.0 * shift_12, rel=1e-14)

    def test_invalid_g_source(self):
        lf = lande_factor(kappa=-1, Z=1)
        with pytest.raises(ValueError, match="g_source"):
            zeeman_shift_table(lf, B_Tesla=1.0, g_source="typo")


# ---------------------------------------------------------------------------
# SCFResult integration
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h_1s_result():
    shells = [Subshell(n=1, kappa=-1, occupation=1)]
    solver = DFSolver(Z=1, subshells=shells, ns=60, ks=9, exchange="hf")
    return solver, solver.solve(max_iter=50, tol=1e-8)


class TestSCFIntegration:
    def test_zeeman_factors_dict(self, h_1s_result):
        _, result = h_1s_result
        factors = result.zeeman_factors()
        assert "1s1/2" in factors
        lf = factors["1s1/2"]
        assert lf.bound == pytest.approx(2.002283845, abs=2e-7)

    def test_zeeman_factors_orbital_filter(self, h_1s_result):
        _, result = h_1s_result
        factors = result.zeeman_factors(orbitals=["1s1/2"])
        assert list(factors.keys()) == ["1s1/2"]

    def test_zeeman_shifts_convenience(self, h_1s_result):
        _, result = h_1s_result
        shifts = result.zeeman_shifts(B_Tesla=1.0)
        z = shifts["1s1/2"]
        assert z.splitting_MHz > 0
        assert z.B_Tesla == 1.0

    def test_anomaly_toggle_affects_factors(self, h_1s_result):
        _, result = h_1s_result
        with_ae = result.zeeman_factors(include_anomalous_moment=True)
        no_ae  = result.zeeman_factors(include_anomalous_moment=False)
        assert with_ae["1s1/2"].bound > no_ae["1s1/2"].bound
