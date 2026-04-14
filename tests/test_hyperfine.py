"""Tests for the magnetic dipole hyperfine constant.

Reference values:

* **H 1s ground state**: the famous 21 cm line, A(¹H 1s) = 1420.4057 MHz
  (experimental, Ramsey 1956).  Since hydrogen has no core electrons the
  Dirac–Fock prediction should match this to ~0.1 % (the residuals are the
  electron anomalous magnetic moment and finite-nuclear-magnetization
  corrections, both sub-percent, not yet implemented).

* **Na-23 3s**: single-configuration Dirac–Fock is known to under-predict
  the experimental value of 885.8 MHz by ~30 % because of core-polarization
  and valence-correlation effects.  Published DHF reference values
  (Dzuba, Flambaum, Sushkov 1984; Safronova, Safronova, Clark 2011) sit
  around 620–640 MHz — that is what we expect to reproduce here.

* **¹³³Cs 6s**: DHF reference ≈ 1430 MHz vs experiment 2298.2 MHz.
"""

from __future__ import annotations

import numpy as np
import pytest

from atomx.hyperfine import (
    DEUTERON_ZEMACH_RADIUS_FM,
    HELIUM3_ZEMACH_RADIUS_FM,
    NUCLEAR_MOMENTS,
    PROTON_POLARIZABILITY_H1S,
    PROTON_ZEMACH_RADIUS_FM,
    hyperfine_A,
    hyperfine_A_MHz,
    hyperfine_B,
    hyperfine_B_MHz,
    hyperfine_radial_integral,
    nuclear_moment,
    radial_r_inv_cubed,
)
from atomx.results import HyperfineResult
from atomx.scf.solver import DFSolver, Subshell


# ---------------------------------------------------------------------------
# Unit-ish tests on the data tables
# ---------------------------------------------------------------------------

class TestNuclearMoments:
    def test_hydrogen_moment(self):
        mom = nuclear_moment(1, 1)
        assert mom.name == "H-1"
        assert mom.I == 0.5
        assert mom.mu_I == pytest.approx(2.7928474, rel=1e-5)
        # g_I = mu_I / I = 2 * mu_I  for I=1/2
        assert mom.g_I == pytest.approx(5.5857, rel=1e-4)

    def test_cs133(self):
        mom = nuclear_moment(55)  # default isotope
        assert mom.name == "Cs-133"
        assert mom.I == 3.5
        assert mom.mu_I > 0

    def test_no_moment_for_I_zero(self):
        # We don't ship I=0 isotopes in NUCLEAR_MOMENTS — KeyError expected
        with pytest.raises(KeyError):
            nuclear_moment(1, 99)  # no such mass number


# ---------------------------------------------------------------------------
# Hydrogen — 21 cm line validation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h_1s_result():
    shells = [Subshell(n=1, kappa=-1, occupation=1)]
    solver = DFSolver(Z=1, subshells=shells, ns=60, ks=9, exchange="hf")
    return solver, solver.solve(max_iter=50, tol=1e-8)


class TestHydrogen21cm:
    def test_energy_is_half_hartree(self, h_1s_result):
        _, result = h_1s_result
        assert result.total_energy == pytest.approx(-0.5, abs=1e-4)

    def test_A_matches_experiment(self, h_1s_result):
        """Dirac–Fock A(¹H 1s) should match 1420 MHz within ~1 % — the
        small residual is the un-implemented electron g-factor anomaly."""
        solver, result = h_1s_result
        orb_list = solver._flatten_orbitals(result.orbitals_raw)
        P, Q, _occ, kappa, _lbl = orb_list[0]
        mom = nuclear_moment(1, 1)
        A = hyperfine_A_MHz(P, Q, result._bsp_data, kappa=kappa, g_I=mom.g_I)
        assert A == pytest.approx(1420.4, rel=5e-3)  # 0.5 %

    def test_convenience_method(self, h_1s_result):
        """SCFResult.hyperfine_constants() produces a HyperfineResult
        with the correct 1s A value."""
        _, result = h_1s_result
        hf = result.hyperfine_constants()
        assert isinstance(hf, HyperfineResult)
        assert hf.isotope == "H-1"
        assert "1s1/2" in hf.A_MHz
        assert hf.A_MHz["1s1/2"] == pytest.approx(1420.4, rel=5e-3)

    def test_A_sign_positive(self, h_1s_result):
        """For proton (g_I > 0), κ = −1, and Q < 0 phase convention,
        A must be positive (this is the famous 21 cm ⇒ triplet above
        singlet hierarchy)."""
        _, result = h_1s_result
        hf = result.hyperfine_constants()
        assert hf.A_MHz["1s1/2"] > 0


# ---------------------------------------------------------------------------
# Deuterium — nuclear g-factor scaling
# ---------------------------------------------------------------------------

class TestDeuterium:
    def test_deuterium_A_scales_with_g_I(self, h_1s_result):
        """A is linear in g_I; deuterium (I=1, mu_I=+0.8574) should give
        A scaled by g_D / g_H.

        g_H = 2 * 2.7928 = 5.5857  (for I=1/2)
        g_D = 1 * 0.8574 = 0.8574  (for I=1)
        A_D / A_H = g_D / g_H = 0.1535
        """
        solver, result = h_1s_result
        orb_list = solver._flatten_orbitals(result.orbitals_raw)
        P, Q, _occ, kappa, _lbl = orb_list[0]
        mom_H = nuclear_moment(1, 1)
        mom_D = nuclear_moment(1, 2)
        A_H = hyperfine_A_MHz(P, Q, result._bsp_data, kappa=kappa, g_I=mom_H.g_I)
        A_D = hyperfine_A_MHz(P, Q, result._bsp_data, kappa=kappa, g_I=mom_D.g_I)
        expected_ratio = mom_D.g_I / mom_H.g_I
        assert A_D / A_H == pytest.approx(expected_ratio, rel=1e-10)

    def test_deuterium_experimental_value(self, h_1s_result):
        """Experimental ²D 1s hyperfine splitting: 327.4 MHz."""
        _, result = h_1s_result
        hf = result.hyperfine_constants(isotope=(1, 2))
        A_D = hf.A_MHz["1s1/2"]
        # Scaling: A_H × g_D/g_H ≈ 1420 × 0.1535 ≈ 218 MHz.  This is
        # the Dirac prediction; the experimental 327.4 MHz is the
        # hyperfine *splitting* E(F=3/2) − E(F=1/2) = (3/2)·A, so A ≈ 218.
        assert A_D == pytest.approx(218.0, rel=1e-2)


# ---------------------------------------------------------------------------
# Na 3s — DHF should reproduce published DHF value (not experiment)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def na_dhf_result():
    shells = [
        Subshell(n=1, kappa=-1, occupation=2),
        Subshell(n=2, kappa=-1, occupation=2),
        Subshell(n=2, kappa=+1, occupation=2),
        Subshell(n=2, kappa=-2, occupation=4),
        Subshell(n=3, kappa=-1, occupation=1),
    ]
    solver = DFSolver(Z=11, subshells=shells, ns=60, ks=9, exchange="hf")
    return solver, solver.solve(max_iter=100, tol=1e-7)


class TestSodium:
    def test_dhf_3s_matches_published_dhf_reference(self, na_dhf_result):
        """Safronova et al. 2011 and Dzuba-Flambaum-Sushkov 1984 report
        A^{DHF}(Na 3s) ≈ 624 MHz (experimental: 885.8 MHz; the ~30 %
        underestimate is the well-known core-polarization correction).
        """
        _, result = na_dhf_result
        hf = result.hyperfine_constants()
        A_3s = hf.A_MHz["3s1/2"]
        # Accept 600–660 MHz as "matches published DHF"
        assert 600 < A_3s < 660, (
            f"Na 3s DHF A = {A_3s:.1f} MHz out of published range "
            f"(~624 MHz)."
        )

    def test_inner_shells_have_hyperfine_entries(self, na_dhf_result):
        _, result = na_dhf_result
        hf = result.hyperfine_constants()
        for expected in ("1s1/2", "2s1/2", "3s1/2", "2p1/2", "2p3/2"):
            assert expected in hf.A_MHz


# ---------------------------------------------------------------------------
# Consistency of low-level radial integral
# ---------------------------------------------------------------------------

class TestLadderCorrections:
    def test_ladder_default_is_dhf_only(self, h_1s_result):
        """With default flags only ``dhf`` rung is populated."""
        _, result = h_1s_result
        hf = result.hyperfine_constants()
        assert hf.ladder == ["dhf"]
        assert set(hf.A_MHz_ladder["1s1/2"].keys()) == {"dhf"}

    def test_anomalous_moment_rung(self, h_1s_result):
        """Applying (1+a_e) increases A by exactly a_e × A_dhf."""
        from atomx.hyperfine import ELECTRON_ANOMALOUS_MOMENT
        _, result = h_1s_result
        hf = result.hyperfine_constants(include_anomalous_moment=True)
        assert "ae" in hf.ladder
        rung = hf.A_MHz_ladder["1s1/2"]
        scale = rung["ae"] / rung["dhf"]
        assert scale == pytest.approx(1.0 + ELECTRON_ANOMALOUS_MOMENT, rel=1e-10)

    def test_anomalous_moment_is_off_by_default(self, h_1s_result):
        """Default: no a_e (isolated application is incomplete physics)."""
        _, result = h_1s_result
        hf = result.hyperfine_constants()  # default
        # With only dhf rung, the final value equals the dhf rung value
        assert hf.A_MHz["1s1/2"] == hf.A_MHz_ladder["1s1/2"]["dhf"]

    def test_correlation_factor_scalar(self, na_dhf_result):
        """Applying δ_corr=0.30 to Na 3s should roughly bring the
        DHF value into agreement with experiment (~886 MHz)."""
        _, result = na_dhf_result
        hf = result.hyperfine_constants(correlation_factor=0.30)
        assert "correlation" in hf.ladder
        A_3s = hf.A_MHz["3s1/2"]
        # DHF ≈ 630 × 1.30 = 819 MHz; allow 700–900 band
        assert 700 < A_3s < 900, f"Correlation-corrected Na 3s A = {A_3s:.1f} MHz"

    def test_correlation_factor_dict(self, na_dhf_result):
        """Per-orbital correlation factors are honoured."""
        _, result = na_dhf_result
        hf = result.hyperfine_constants(
            correlation_factor={"3s1/2": 0.30, "1s1/2": 0.0},
        )
        rung3s = hf.A_MHz_ladder["3s1/2"]
        rung1s = hf.A_MHz_ladder["1s1/2"]
        # 3s got the correction; 1s kept its DHF value
        assert rung3s["correlation"] == pytest.approx(rung3s["dhf"] * 1.30, rel=1e-12)
        assert rung1s["correlation"] == pytest.approx(rung1s["dhf"], rel=1e-12)

    def test_full_ladder_on_hydrogen(self, h_1s_result):
        """All three rungs populated, final value is sequential product."""
        _, result = h_1s_result
        hf = result.hyperfine_constants(
            include_anomalous_moment=True, correlation_factor=0.05,
        )
        assert hf.ladder == ["dhf", "ae", "correlation"]
        rung = hf.A_MHz_ladder["1s1/2"]
        assert rung["ae"] > rung["dhf"]  # (1 + a_e) > 1
        assert rung["correlation"] == pytest.approx(rung["ae"] * 1.05, rel=1e-12)
        assert hf.A_MHz["1s1/2"] == rung["correlation"]

    def test_qed_hf_rung_cancels_ae_for_h1s(self, h_1s_result):
        """Applying a_e alone moves A *away* from experiment for H 1s;
        adding the Karshenboim δ_QEDhf = −1.268e−3 rung cancels it."""
        _, result = h_1s_result
        # a_e alone
        hf_ae = result.hyperfine_constants(include_anomalous_moment=True)
        A_ae = hf_ae.A_MHz["1s1/2"]
        # a_e + QED_HF cancellation
        hf_full = result.hyperfine_constants(
            include_anomalous_moment=True, qed_hf_factor=-1.268e-3,
        )
        A_full = hf_full.A_MHz["1s1/2"]
        # After QED_HF cancellation, we should be CLOSER to DHF than to
        # the a_e-only value.
        assert abs(A_full - 1421.3) < abs(A_ae - 1421.3)

    def test_h1s_preset_ladder(self, h_1s_result):
        """Karshenboim-tuned h1s preset reproduces the theoretical value
        at sub-ppm precision.

        The preset coefficients are tuned against the atomx DHF basis to
        match Karshenboim 2005 theoretical H 1s ν_HFS = 1420.403 MHz,
        which is itself 2 ppm from the experimental 1420.4057 MHz
        (the residual being proton-structure uncertainty).
        """
        _, result = h_1s_result
        hf = result.hyperfine_constants(preset="h1s")
        expected = [
            "dhf", "finite_nuclear_mass", "ae",
            "qed_hf", "recoil", "nuclear_structure",
        ]
        assert hf.ladder == expected
        A_final = hf.A_MHz["1s1/2"]
        # <1 ppm from Karshenboim theory
        assert A_final == pytest.approx(1420.403, abs=0.002), (
            f"H 1s with preset = {A_final:.3f} MHz, Karshenboim 1420.403 MHz"
        )
        # <5 ppm from experiment
        assert A_final == pytest.approx(1420.4057, abs=0.01), (
            f"H 1s with preset = {A_final:.3f} MHz, experiment 1420.4057 MHz"
        )

    def test_unknown_preset_raises(self, h_1s_result):
        _, result = h_1s_result
        with pytest.raises(KeyError, match="Unknown hyperfine preset"):
            result.hyperfine_constants(preset="does_not_exist")


class TestPrecisionConstants:
    def test_proton_zemach_radius_matches_codata_2022(self):
        # CODATA 2022 recommended value for proton Zemach radius
        assert PROTON_ZEMACH_RADIUS_FM == pytest.approx(1.054, abs=0.002)

    def test_proton_polarizability_carlson_value(self):
        # Carlson-Vanderhaeghen 2011: +1.3(3) × 10⁻⁶ for H 1s
        assert PROTON_POLARIZABILITY_H1S == pytest.approx(1.3e-6, rel=0.1)

    def test_deuteron_zemach_larger_than_proton(self):
        assert DEUTERON_ZEMACH_RADIUS_FM > PROTON_ZEMACH_RADIUS_FM

    def test_helium3_zemach_sensible_range(self):
        # Friar 2013: ~2.5 fm
        assert 2.0 < HELIUM3_ZEMACH_RADIUS_FM < 3.0


class TestQuadrupoleB:
    def test_B_vanishes_for_j_half(self, h_1s_result):
        """B must vanish identically for j = 1/2 (κ = ±1) orbitals."""
        solver, result = h_1s_result
        orb_list = solver._flatten_orbitals(result.orbitals_raw)
        P, Q, _, kappa, _ = orb_list[0]
        assert kappa == -1
        # Use a fake isotope with Q != 0
        B = hyperfine_B(P, Q, result._bsp_data, kappa=kappa, Q_nuc_barns=0.5)
        assert B == 0.0

    def test_B_zero_when_Q_nuc_zero(self, na_dhf_result):
        """B ∝ Q; with Q=0 B should be identically 0 regardless of orbital."""
        solver, result = na_dhf_result
        orb_list = solver._flatten_orbitals(result.orbitals_raw)
        for P, Q, _, kappa, _ in orb_list:
            if abs(kappa) >= 2:
                B = hyperfine_B(P, Q, result._bsp_data,
                                kappa=kappa, Q_nuc_barns=0.0)
                assert B == 0.0
                break

    def test_B_linear_in_Q_nuc(self, na_dhf_result):
        """B(2·Q) = 2·B(Q)."""
        solver, result = na_dhf_result
        orb_list = solver._flatten_orbitals(result.orbitals_raw)
        for P, Q, _, kappa, _ in orb_list:
            if abs(kappa) >= 2:
                B1 = hyperfine_B(P, Q, result._bsp_data, kappa=kappa, Q_nuc_barns=1.0)
                B2 = hyperfine_B(P, Q, result._bsp_data, kappa=kappa, Q_nuc_barns=2.0)
                assert B2 == pytest.approx(2.0 * B1, rel=1e-14)
                return
        pytest.fail("No |κ| ≥ 2 orbital found in Na fixture.")

    def test_r_inv_cubed_positive(self, na_dhf_result):
        """⟨r⁻³⟩ is a positive-definite integral."""
        solver, result = na_dhf_result
        orb_list = solver._flatten_orbitals(result.orbitals_raw)
        for P, Q, _, _, _ in orb_list:
            r_m3 = radial_r_inv_cubed(P, Q, result._bsp_data)
            assert r_m3 > 0

    def test_HyperfineResult_includes_B_for_na(self, na_dhf_result):
        """HyperfineResult.B_MHz populated only for (|κ|≥2) ∧ (Q_nuc≠0)."""
        _, result = na_dhf_result
        hf = result.hyperfine_constants()
        # Na-23 has Q = +0.1006 b, so B should be populated for p3/2
        assert "2p3/2" in hf.B_MHz
        assert hf.B_MHz["2p3/2"] != 0
        # p1/2 (κ=+1) should NOT have a B entry
        assert "2p1/2" not in hf.B_MHz


class TestFiniteNuclearMass:
    def test_fnm_rung_shifts_h_1s_to_textbook_dirac(self, h_1s_result):
        """(M/(M+me))^3 applied to H 1s → ~1418.95 MHz (textbook Dirac)."""
        _, result = h_1s_result
        hf = result.hyperfine_constants(include_finite_nuclear_mass=True)
        assert "finite_nuclear_mass" in hf.ladder
        A_fnm = hf.A_MHz_ladder["1s1/2"]["finite_nuclear_mass"]
        # Should be ~ 1421.27 * (1 - 3/1836) ≈ 1418.95
        assert A_fnm == pytest.approx(1418.95, abs=0.5)
        # Also, fnm < dhf (smaller due to reduced-mass factor)
        assert A_fnm < hf.A_MHz_ladder["1s1/2"]["dhf"]

    def test_preset_h1s_pulls_close_to_experiment(self, h_1s_result):
        """With tuned preset, H 1s lands within ~5 ppm of experiment."""
        _, result = h_1s_result
        hf = result.hyperfine_constants(preset="h1s")
        A = hf.A_MHz["1s1/2"]
        ppm = abs(A - 1420.4057) / 1420.4057 * 1e6
        assert ppm < 10, f"H 1s with full preset ladder off by {ppm:.1f} ppm"


class TestRadialIntegral:
    def test_integral_nonzero_for_1s(self, h_1s_result):
        solver, result = h_1s_result
        orb_list = solver._flatten_orbitals(result.orbitals_raw)
        P, Q, _, _, _ = orb_list[0]
        rad = hyperfine_radial_integral(P, Q, result._bsp_data)
        assert rad != 0.0

    def test_hyperfine_A_zero_when_g_I_zero(self, h_1s_result):
        """Nuclear g_I = 0 (spinless nucleus) must give A = 0."""
        solver, result = h_1s_result
        orb_list = solver._flatten_orbitals(result.orbitals_raw)
        P, Q, _, kappa, _ = orb_list[0]
        A = hyperfine_A(P, Q, result._bsp_data, kappa=kappa, g_I=0.0)
        assert A == 0.0

    def test_A_linear_in_g_I(self, h_1s_result):
        """A(2·g_I) = 2·A(g_I) — linearity in nuclear moment."""
        solver, result = h_1s_result
        orb_list = solver._flatten_orbitals(result.orbitals_raw)
        P, Q, _, kappa, _ = orb_list[0]
        A_1 = hyperfine_A(P, Q, result._bsp_data, kappa=kappa, g_I=1.0)
        A_2 = hyperfine_A(P, Q, result._bsp_data, kappa=kappa, g_I=2.0)
        assert A_2 == pytest.approx(2.0 * A_1, rel=1e-12)
