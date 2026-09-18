"""M4 tests: Breit interaction and QED corrections.

All corrections are perturbative (computed after SCF convergence).
Reference magnitudes from Desclaux (1973) and DBSR_HF.
"""

import numpy as np
import pytest

from atomx.corrections.qed import (
    compute_se_energy,
    uehling_potential,
)
from atomx.methods.dirac_fock import DFSolver, helium_subshells, neon_subshells

# ---- Shared fixtures ----

@pytest.fixture(scope="module")
def ne_hf_solved():
    """Ne DHF converged result with orbitals accessible."""
    solver = DFSolver(Z=10, subshells=neon_subshells(), ns=50, ks=9, exchange="hf")
    result = solver.solve(max_iter=40, tol=1e-5)
    # The result carries orbital_energies but not raw orbitals; the
    # Breit/QED assertions below work through the energy_components dict.
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
        assert pytest.approx(10 * V_Z1, rel=1e-10) == V_Z10


# ---- Self-energy tests ----

class TestSelfEnergy:
    def test_se_1s_positive_for_low_Z(self):
        """1s self-energy shifts the total energy upward for low Z.

        Sign convention: `energy_components["E_SE"]` is a shift *added* to
        the Dirac-Coulomb total, so a positive value makes the atom **less**
        bound (the Lamb shift raises s levels).  The reverse was stated here
        before 2026-09-18.
        """
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
        assert pytest.approx(-0.002355, rel=0.05) == E_VP, (
            f"Ne VP {E_VP:.6f} out of range (expected ~-0.002355)")

    def test_ne_has_se(self, ne_full_result):
        """Ne SE correction should be nonzero."""
        E_SE = ne_full_result.energy_components.get("E_SE", 0)
        assert E_SE != 0, "SE should be nonzero for Ne"

    def test_ne_total_with_corrections(self, ne_full_result):
        """Ne total with Breit+QED is pinned (was a 2 Ha band).

        Measured 2026-09-18: Breit +0.050437, VP −0.002355, SE (mohr)
        +0.030218 ⇒ total −128.6136144 Ha.  The old `abs=2.0` band could not
        have caught a wrong correction; 0.05 Ha still tolerates the known
        per-engine spread (bspline vs shooting agree to ~2e-5 Ha).
        """
        assert ne_full_result.total_energy == pytest.approx(-128.6136, abs=0.05)

    def test_ne_correction_signs(self, ne_full_result):
        """VP is negative (attractive), SE is positive, Breit usually negative."""
        E = ne_full_result.energy_components
        E_VP = E.get("E_VP", 0)
        E_SE = E.get("E_SE", 0)
        assert E_VP < 0, f"VP should be negative: {E_VP}"
        assert E_SE > 0, f"SE should be positive: {E_SE}"


# ---- Total QED shift: the previously unpinned quantity ----

class TestQEDTotalShiftPin:
    """Pin the **total** QED shift a default `Atom(qed=True)` applies.

    Why this class exists (2026-09-18 review, C3): the self-energy is the
    largest correction atomx applies by default (Ne: +0.0302 Ha, 12x the
    Uehling VP term) and *no* test pinned the total shift — `test_qed.py`
    asserted `E_SE != 0`, an order-of-magnitude band for He and, for Ne, only
    that the total stayed within 2 Ha.  The only external anchor for the
    self-energy is the analytic hydrogenic Z=1 case
    (`tests/test_qedmod.py::TestMohrBackwardCompat`); a tabulated many-electron
    reference is still missing, which `docs/VERIFICATION.md` lists explicitly.
    These are therefore **self-regression pins**, not external validation:
    they make a silent change in the default correction stack fail loudly.
    """

    #: Z -> (label, pinned total shift in Ha, pinned SE, pinned VP)
    PINS = {
        10: ("Ne", +0.027862900, +0.030218071, -0.002355171),
        18: ("Ar", +0.330528077, +0.356042705, -0.025514628),
    }

    @pytest.mark.parametrize("Z", sorted(PINS))
    def test_default_qed_shift_is_pinned(self, Z):
        from atomx import Atom

        label, shift_ref, _se_ref, _vp_ref = self.PINS[Z]
        base = Atom(Z=Z, numerics="shooting", breit=False, qed=False)
        base.solve()
        with_qed = Atom(Z=Z, numerics="shooting", breit=False, qed=True)
        with_qed.solve()
        shift = with_qed.total_energy - base.total_energy
        assert shift == pytest.approx(shift_ref, rel=1e-3), (
            f"{label}: default QED shift {shift:+.9f} != {shift_ref:+.9f} Ha")

    @pytest.mark.parametrize("Z", sorted(PINS))
    def test_default_stack_is_self_energy_plus_uehling(self, Z):
        """The default stack is SE + Uehling VP — Wichmann-Kroll is opt-in.

        Worth pinning: WK is a *separate* adapter and is not included by
        default, so the default QED shift is smaller than the sum of the
        three terms by exactly the WK contribution.
        """
        from atomx import Atom
        from atomx.corrections.adapters import (
            SelfEnergyCorrection,
            UehlingVPCorrection,
            WichmannKrollVPCorrection,
        )
        from atomx.corrections.qed import resolve_se_method

        label, shift_ref, se_ref, vp_ref = self.PINS[Z]
        base = Atom(Z=Z, numerics="shooting", breit=False, qed=False)
        base.solve()
        result = base.result
        backend = resolve_se_method("auto", Z)
        se = SelfEnergyCorrection(backend=backend).evaluate(result).energy_shift
        vp = UehlingVPCorrection().evaluate(result).energy_shift
        wk = WichmannKrollVPCorrection().evaluate(result).energy_shift
        assert backend == "mohr", f"{label}: default SE backend is {backend!r}"
        assert se == pytest.approx(se_ref, rel=1e-3)
        assert vp == pytest.approx(vp_ref, rel=1e-3)
        assert se + vp == pytest.approx(shift_ref, rel=1e-3), (
            f"{label}: stack is not SE+VP (SE={se:+.6f}, VP={vp:+.6f}, "
            f"pinned total {shift_ref:+.9f})")
        assert wk != 0.0, "WK adapter should still compute a value"
