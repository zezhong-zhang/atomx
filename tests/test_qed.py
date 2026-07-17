"""M4 tests: Breit interaction and QED corrections.

All corrections are perturbative (computed after SCF convergence).
Reference magnitudes from Desclaux (1973) and DBSR_HF.
"""

import numpy as np
import pytest
from atomx.methods.dirac_fock import DFSolver, helium_subshells, neon_subshells
from atomx.corrections.qed import uehling_potential, compute_se_energy, compute_vp_energy
from atomx.corrections.breit import compute_breit_energy
from atomx.constants import ALPHA, C


# ---- Shared fixtures ----

@pytest.fixture(scope="module")
def ne_hf_solved():
    """Ne DHF converged result with orbitals accessible."""
    solver = DFSolver(Z=10, subshells=neon_subshells(), ns=50, ks=9, exchange="hf")
    result = solver.solve(max_iter=40, tol=1e-5)
    orb_list = solver._flatten_orbitals(
        {sh.kappa: [] for sh in solver.subshells})
    # Re-flatten from the result — need to re-solve to get orbitals
    # Actually solve returns orbital_energies but not raw orbitals.
    # Use include_breit/qed flags instead.
    return solver, result


@pytest.fixture(scope="module")
def ne_full_result():
    """Ne DHF with Breit + QED corrections."""
    solver = DFSolver(Z=10, subshells=neon_subshells(), ns=50, ks=9,
                      exchange="hf", include_breit=True, include_qed=True)
    return solver.solve(max_iter=40, tol=1e-5)


@pytest.fixture(scope="module")
def he_full_result():
    """He DHF with Breit + QED corrections."""
    solver = DFSolver(Z=2, subshells=helium_subshells(), ns=40, ks=9,
                      exchange="hf", include_breit=True, include_qed=True)
    return solver.solve(max_iter=60, tol=1e-6)


# ---- Uehling potential tests ----

class TestUehlingPotential:
    def test_negative(self):
        """VP potential is attractive (negative) within Compton wavelength."""
        # VP decays as exp(-2cr); only nonzero for r << 1/(2c) ≈ 0.004 Bohr
        r = np.array([0.001, 0.005, 0.01])
        V = uehling_potential(r, Z=10)
        assert np.all(V < 0)

    def test_decays_with_r(self):
        """VP potential decays rapidly (exponentially) at large r."""
        V_near = uehling_potential(0.1, Z=10)
        V_far = uehling_potential(10.0, Z=10)
        assert abs(V_far) < abs(V_near) * 0.01

    def test_scales_with_Z(self):
        """VP potential is proportional to Z."""
        V_Z1 = uehling_potential(1.0, Z=1)
        V_Z10 = uehling_potential(1.0, Z=10)
        assert V_Z10 == pytest.approx(10 * V_Z1, rel=1e-10)


# ---- Self-energy tests ----

class TestSelfEnergy:
    def test_se_1s_positive_for_low_Z(self):
        """1s self-energy is positive (increases binding) for low Z."""
        orbs = [(np.zeros(50), np.zeros(50), 2, -1, "1s1/2")]
        E_SE = compute_se_energy(orbs, Z=2)
        # F(Zα) ≈ 10.6 for Z=2, so E_SE should be positive
        assert E_SE > 0

    def test_se_scales_as_Z4(self):
        """SE scales approximately as Z^4."""
        orbs1 = [(np.zeros(50), np.zeros(50), 2, -1, "1s1/2")]
        orbs2 = [(np.zeros(50), np.zeros(50), 2, -1, "1s1/2")]
        E1 = compute_se_energy(orbs1, Z=10)
        E2 = compute_se_energy(orbs2, Z=20)
        ratio = E2 / E1
        # Should scale as (20/10)^4 = 16, but F(Zα) also changes
        assert 10 < ratio < 25


# ---- Integration tests (Breit + QED in full calculation) ----

class TestBreitQEDIntegration:
    def test_he_breit_small(self, he_full_result):
        """He Breit correction should be tiny (~10^-5 Hartree)."""
        E_Breit = he_full_result.energy_components.get("E_Breit", 0)
        # He (Z=2) Breit is α²·Z⁴·<1/r_12>-ish, so ~ α²·16·1 ~ 8.5e-4 Ha
        assert abs(E_Breit) < 0.01, f"He Breit too large: {E_Breit}"

    def test_he_qed_small(self, he_full_result):
        """He QED corrections should be tiny."""
        E_QED = he_full_result.energy_components.get("E_QED_total", 0)
        assert abs(E_QED) < 0.01, f"He QED too large: {E_QED}"

    def test_he_total_energy_with_corrections(self, he_full_result):
        """He total energy with Breit+QED should be close to DHF value."""
        assert he_full_result.total_energy == pytest.approx(-2.862, abs=0.05)

    def test_ne_has_breit(self, ne_full_result):
        """Ne Breit correction should be nonzero and physically sized.

        For neutral Ne (Z=10, 10 electrons) the Breit interaction
        scales as α²·Z⁴·electron-density-factor and lies around
        +0.05 Ha (Grant 2007 tables for Ne I).  The previous cap of
        0.01 Ha was set when atomx's Breit was α²-suppressed (an old
        bug in corrections/breit.py); it is now physically correct.
        """
        E_Breit = ne_full_result.energy_components.get("E_Breit", 0)
        assert E_Breit != 0, "Ne Breit should be nonzero"
        # Tight pin (was a 100×-wide 0.005–0.5 band that hid regressions).
        # atomx's Ne Breit is deterministic at +0.0504 Ha (~+0.05, Grant
        # 2007). rel=5% catches the α²-suppression class of bug (the old
        # 100×-too-small regression) and any >5% drift; a deliberate Breit-
        # operator refinement should update this number, not slip through.
        # 2026-07: pin moved 0.0519 → 0.0504 when the Breit radial
        # quadrature switched to the canonical trapezoidal Y^k kernel
        # (numerics.quadrature.yk_on_grid); the old rectangle-cumulative
        # compute_yk double-counted the diagonal cell and inflated the
        # strongly-cancelling Breit total by ~3% (dense-grid converged
        # value 0.050429, independently confirmed by the shooting engine
        # at 0.050437 — see tests/test_shooting_corrections.py).
        assert E_Breit == pytest.approx(0.0504, rel=0.05), (
            f"Ne Breit {E_Breit:.5f} out of range (expected ~+0.0504)"
        )

    def test_ne_has_vp(self, ne_full_result):
        """Ne Uehling VP correction: deterministic -0.002355 Ha (was: nonzero).

        VP (Uehling vacuum polarization) is negative and small for Ne; pin
        it so sign/magnitude regressions trip instead of only ==0.
        """
        E_VP = ne_full_result.energy_components.get("E_VP", 0)
        assert E_VP == pytest.approx(-0.002355, rel=0.05), (
            f"Ne VP {E_VP:.6f} out of range (expected ~-0.002355)")

    def test_ne_has_se(self, ne_full_result):
        """Ne SE correction should be nonzero."""
        E_SE = ne_full_result.energy_components.get("E_SE", 0)
        assert E_SE != 0, "SE should be nonzero for Ne"

    def test_ne_total_with_corrections(self, ne_full_result):
        """Ne total with Breit+QED should still be close to DHF value."""
        # Corrections are small, so total shouldn't change much
        assert ne_full_result.total_energy == pytest.approx(-128.5, abs=2.0)

    def test_ne_correction_signs(self, ne_full_result):
        """VP is negative (attractive), SE is positive, Breit usually negative."""
        E = ne_full_result.energy_components
        E_VP = E.get("E_VP", 0)
        E_SE = E.get("E_SE", 0)
        assert E_VP < 0, f"VP should be negative: {E_VP}"
        assert E_SE > 0, f"SE should be positive: {E_SE}"
