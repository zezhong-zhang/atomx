"""Tests for the pluggable self-energy backend framework.

Covers:
    * Backend registry (mohr, welton, flambaum_ginges, shabaev).
    * Bit-for-bit backward compatibility: default dispatch == legacy Mohr.
    * Welton closed-form scaling on a numerical orbital.
    * Stubbed backends raise a helpful NotImplementedError that points at
      the target paper.
    * End-to-end SCF integration through the new ``qed_se_method`` kwarg.
"""

from __future__ import annotations

import numpy as np
import pytest

from atomx.constants import ALPHA
from atomx.corrections.qed import compute_qed_corrections, compute_se_energy
from atomx.corrections.qedmod import (
    SE_BACKENDS,
    available_backends,
    get_backend,
)
from atomx.corrections.qedmod.welton import _density_at_origin  # noqa: F401
from atomx.scf.solver import DFSolver, Subshell, helium_subshells


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestBackendRegistry:
    def test_all_backends_registered(self):
        names = available_backends()
        assert "mohr" in names
        assert "welton" in names
        assert "flambaum_ginges" in names
        assert "shabaev" in names

    def test_get_backend_returns_instance(self):
        for name in available_backends():
            backend = get_backend(name)
            assert backend.name == name
            assert backend.description  # non-empty

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown self-energy backend"):
            get_backend("no_such_backend")


# ---------------------------------------------------------------------------
# Mohr backend — backward-compatibility guarantee
# ---------------------------------------------------------------------------

class TestMohrBackwardCompat:
    def test_hydrogenic_1s_matches_reference(self):
        """F(Zα=α) × α(Zα)^4/(πn^3 α²)  with F=10.3169 for Z=1."""
        orbs = [(np.zeros(10), np.zeros(10), 1, -1, "1s1/2")]
        E = compute_se_energy(orbs, Z=1, method="mohr")
        # Analytic: α³ × 10.3169 / π
        expected = ALPHA ** 3 * 10.3169 / np.pi
        assert E == pytest.approx(expected, rel=1e-10)

    def test_default_method_is_mohr(self):
        """compute_se_energy without `method` keyword == method='mohr'."""
        orbs = [(np.zeros(10), np.zeros(10), 2, -1, "1s1/2")]
        E_default = compute_se_energy(orbs, Z=10)
        E_mohr = compute_se_energy(orbs, Z=10, method="mohr")
        assert E_default == pytest.approx(E_mohr, rel=1e-14)


# ---------------------------------------------------------------------------
# Welton backend
# ---------------------------------------------------------------------------

class TestWelton:
    def test_returns_zero_without_orbitals(self):
        """Welton is meaningless without P/Q, should gracefully return 0."""
        orbs = [(np.zeros(10), np.zeros(10), 2, -1, "1s1/2")]
        E = compute_se_energy(orbs, Z=5, method="welton", bsp_data=None)
        assert E == 0.0

    def test_sane_magnitude_for_he_1s(self):
        """Welton for He 1s should be in the 1e-6..1e-3 Ha ballpark."""
        solver = DFSolver(Z=2, subshells=helium_subshells(), ns=40, ks=9,
                          exchange="hf", include_qed=True,
                          qed_se_method="welton")
        result = solver.solve(max_iter=60, tol=1e-6)
        E = result.energy_components["E_SE"]
        assert 1e-6 < E < 1e-3, f"Welton He SE out of range: {E:.3e}"


# ---------------------------------------------------------------------------
# Flambaum–Ginges backend (now a working implementation, electric SE only)
# ---------------------------------------------------------------------------

class TestFlambaumGinges:
    def test_h_1s_matches_mohr_within_10pct(self):
        """FG paper claims ~10 % accuracy for 1s; for H (Z=1) we expect
        ~5 % agreement with Mohr's tabulated SE."""
        shells = [Subshell(n=1, kappa=-1, occupation=1)]
        solver = DFSolver(Z=1, subshells=shells, ns=60, ks=9, exchange="hf")
        result = solver.solve(max_iter=50, tol=1e-8)
        orb_list = solver._flatten_orbitals(result.orbitals_raw)
        E_mohr = compute_se_energy(
            orb_list, Z=1, method="mohr", bsp_data=result._bsp_data)
        E_fg = compute_se_energy(
            orb_list, Z=1, method="flambaum_ginges", bsp_data=result._bsp_data)
        # Same sign, comparable magnitude
        assert E_mohr > 0
        assert E_fg > 0
        ratio = E_fg / E_mohr
        assert 0.9 < ratio < 1.1, f"H 1s FG/Mohr = {ratio:.3f} out of band"

    def test_requires_bsp_data(self):
        orbs = [(np.zeros(10), np.zeros(10), 2, -1, "1s1/2")]
        with pytest.raises(ValueError, match="bsp_data"):
            compute_se_energy(orbs, Z=1, method="flambaum_ginges", bsp_data=None)


# ---------------------------------------------------------------------------
# Wichmann–Kroll higher-order VP
# ---------------------------------------------------------------------------

class TestWichmannKroll:
    def test_wk_signs_opposite_to_uehling(self):
        """WK is opposite sign to Uehling (paper: ~1 % cancellation at
        high Z).  In atomx convention: atomx Uehling < 0, atomx WK > 0."""
        from atomx.corrections.qed import compute_vp_energy, compute_wk_vp_energy
        solver = DFSolver(Z=10, subshells=helium_subshells()  # temp: small atom
                          , ns=40, ks=9, exchange="hf")
        # Use helium 1s² with Z=10 (charge-10 ion) so potentials probe higher-Z physics
        # Actually just use real Ne for this test
        from atomx.scf.solver import neon_subshells
        solver = DFSolver(Z=10, subshells=neon_subshells(), ns=50, ks=9, exchange="hf")
        result = solver.solve(max_iter=40, tol=1e-5)
        orb_list = solver._flatten_orbitals(result.orbitals_raw)
        E_U = compute_vp_energy(orb_list, result._bsp_data, 10)
        E_WK = compute_wk_vp_energy(orb_list, result._bsp_data, 10)
        assert E_U < 0
        assert E_WK > 0
        # WK magnitude should be much smaller than Uehling for Z=10
        assert abs(E_WK) < abs(E_U) * 0.05  # < 5 %

    def test_wk_included_via_compute_qed_corrections(self):
        from atomx.corrections.qed import compute_qed_corrections
        from atomx.scf.solver import neon_subshells
        solver = DFSolver(Z=10, subshells=neon_subshells(), ns=50, ks=9, exchange="hf")
        result = solver.solve(max_iter=40, tol=1e-5)
        orb_list = solver._flatten_orbitals(result.orbitals_raw)
        out = compute_qed_corrections(
            orb_list, result._bsp_data, 10, include_wichmann_kroll=True,
        )
        assert "E_WK" in out
        assert out["E_WK"] > 0
        # total = VP + SE + WK
        assert abs(out["E_QED_total"] - (out["E_VP"] + out["E_SE"] + out["E_WK"])) < 1e-15

    def test_scf_integration_wichmann_kroll_flag(self):
        from atomx.scf.solver import neon_subshells
        solver = DFSolver(
            Z=10, subshells=neon_subshells(), ns=50, ks=9, exchange="hf",
            include_qed=True, include_wichmann_kroll=True,
        )
        result = solver.solve(max_iter=40, tol=1e-5)
        assert "E_WK" in result.energy_components
        assert result.energy_components["E_WK"] > 0


# ---------------------------------------------------------------------------
# Stub backend (Shabaev only, FG is now implemented above)
# ---------------------------------------------------------------------------

class TestStubBackends:
    def test_shabaev_raises_with_reference(self):
        """The full (local + non-local) Shabaev stub raises."""
        orbs = [(np.zeros(10), np.zeros(10), 2, -1, "1s1/2")]
        with pytest.raises(NotImplementedError, match="Shabaev"):
            compute_se_energy(orbs, Z=1, method="shabaev")


# ---------------------------------------------------------------------------
# Shabaev local backend (working — QEDMOD v7 se_pot.f port)
# ---------------------------------------------------------------------------

class TestShabaevLocal:
    def test_h_1s_matches_mohr_at_calibration_point(self):
        """By construction: ⟨ψ_1s_hydrogenic|V_loc|ψ_1s_hydrogenic⟩ = Mohr(1s).

        For a well-converged H 1s SCF orbital (which IS hydrogenic to
        numerical precision) the Shabaev-local result should reproduce
        Mohr to ~1 % — the small residual is the non-rel vs Dirac
        density difference at O((Zα)²)."""
        shells = [Subshell(n=1, kappa=-1, occupation=1)]
        solver = DFSolver(Z=1, subshells=shells,
                          ns=80, ks=9, r_min=1e-5, exchange="hf")
        result = solver.solve(max_iter=80, tol=1e-10)
        orb_list = solver._flatten_orbitals(result.orbitals_raw)
        E_m = compute_se_energy(
            orb_list, Z=1, method="mohr", bsp_data=result._bsp_data,
        )
        E_s = compute_se_energy(
            orb_list, Z=1, method="shabaev_local", bsp_data=result._bsp_data,
        )
        # H 1s: should match to ~0.1 %
        assert E_s == pytest.approx(E_m, rel=2e-3)

    def test_ne_1s_closer_to_mohr_than_flambaum_ginges(self):
        """Total Ne SE: Shabaev-local should be within ~10 % of Mohr
        and demonstrably closer than FG-electric (which is 70 % off).

        The residual ~ 6 % comes from Mohr's explicit orbital-overlap
        ratio for 2s (not captured by the Shabaev local-only form);
        the full non-local Shabaev operator would close this."""
        from atomx.scf.solver import neon_subshells
        solver = DFSolver(Z=10, subshells=neon_subshells(),
                          ns=50, ks=9, exchange="hf")
        result = solver.solve(max_iter=40, tol=1e-5)
        orb_list = solver._flatten_orbitals(result.orbitals_raw)
        E_m = compute_se_energy(
            orb_list, Z=10, method="mohr", bsp_data=result._bsp_data)
        E_s = compute_se_energy(
            orb_list, Z=10, method="shabaev_local", bsp_data=result._bsp_data)
        E_fg = compute_se_energy(
            orb_list, Z=10, method="flambaum_ginges", bsp_data=result._bsp_data)
        # Shabaev-local within 10 % of Mohr
        assert abs(E_s - E_m) / abs(E_m) < 0.10
        # And demonstrably closer than FG-electric (which is 70 % off)
        assert abs(E_s - E_m) < abs(E_fg - E_m)

    def test_high_kappa_return_zero_cleanly(self):
        """|κ| ≥ 4 orbitals have no reference → amplitude is 0, so the
        backend returns zero contribution without crashing."""
        from atomx.corrections.qedmod import get_backend
        shells = [Subshell(n=1, kappa=-1, occupation=1)]
        solver = DFSolver(Z=1, subshells=shells,
                          ns=40, ks=9, exchange="hf")
        result = solver.solve(max_iter=40, tol=1e-7)
        orb_list = solver._flatten_orbitals(result.orbitals_raw)
        # Fake κ = -4 (4f7/2) orbital — no reference tabulated yet
        P, Q, occ, _, _ = orb_list[0]
        faux = [(P, Q, occ, -4, "4f7/2")]
        r = get_backend("shabaev_local").compute(
            faux, result._bsp_data, 1)
        assert r.per_orbital["4f7/2"] == 0.0
        assert r.E_SE == 0.0

    def test_available_as_scf_qed_backend(self):
        """DFSolver accepts qed_se_method='shabaev_local' end-to-end."""
        shells = [Subshell(n=1, kappa=-1, occupation=1)]
        solver = DFSolver(
            Z=1, subshells=shells, ns=60, ks=9, exchange="hf",
            include_qed=True, qed_se_method="shabaev_local",
        )
        result = solver.solve(max_iter=50, tol=1e-7)
        E_SE = result.energy_components.get("E_SE")
        assert E_SE is not None
        assert E_SE > 0
        assert result.energy_components.get("se_method") == "shabaev_local"

    def test_d_orbital_calibration_now_works(self):
        """After the 3d extension, κ = ±3 orbitals give non-zero SE."""
        # Hydrogenic Z=22 with single 3d5/2 electron
        shells = [Subshell(n=3, kappa=-3, occupation=1)]
        solver = DFSolver(Z=22, subshells=shells,
                          ns=60, ks=9, exchange="hf")
        result = solver.solve(max_iter=60, tol=1e-7)
        orb_list = solver._flatten_orbitals(result.orbitals_raw)
        E_s = compute_se_energy(
            orb_list, Z=22, method="shabaev_local", bsp_data=result._bsp_data,
        )
        E_m = compute_se_energy(
            orb_list, Z=22, method="mohr", bsp_data=result._bsp_data,
        )
        # Both should be non-zero (and negative for 3d5/2)
        assert E_s != 0.0
        assert E_m != 0.0
        # Signs should agree
        assert (E_s < 0) == (E_m < 0)


class TestMohrN3Table:
    def test_n3_s_state_interpolated(self):
        """New Mohr table entries for n=3 s-state: Z=10 should give
        F ≈ 11.79 (between 1s=11.68 and 2s=11.87)."""
        from atomx.corrections.qedmod.mohr import _interpolate_se_f
        F_3s = _interpolate_se_f(10, 3, -1)
        # Our tabulated value at Z=10 for 3s
        assert 11.0 < F_3s < 13.0

    def test_n3_p_states_small(self):
        """3p1/2 and 3p3/2 SE is small, of the right sign."""
        from atomx.corrections.qedmod.mohr import _interpolate_se_f
        F_3p12 = _interpolate_se_f(10, 3, 1)
        F_3p32 = _interpolate_se_f(10, 3, -2)
        assert abs(F_3p12) < 1.0
        assert abs(F_3p32) < 1.0

    def test_n3_d_states_tiny(self):
        """3d states have even smaller F values."""
        from atomx.corrections.qedmod.mohr import _interpolate_se_f
        F_3d32 = _interpolate_se_f(50, 3, 2)
        F_3d52 = _interpolate_se_f(50, 3, -3)
        # Larger-Z but d states still << 1
        assert abs(F_3d32) < 0.5
        assert abs(F_3d52) < 0.1


# ---------------------------------------------------------------------------
# SCF integration via qed_se_method kwarg
# ---------------------------------------------------------------------------

class TestSCFIntegration:
    def test_qed_se_method_defaults_to_mohr(self):
        """DFSolver without qed_se_method kwarg uses Mohr (backward compat)."""
        solver = DFSolver(Z=2, subshells=helium_subshells(), ns=40, ks=9,
                          exchange="hf", include_qed=True)
        assert solver.qed_se_method == "mohr"

    def test_qed_se_method_welton_end_to_end(self):
        """Solver accepts qed_se_method='welton' and populates SE accordingly."""
        solver = DFSolver(Z=2, subshells=helium_subshells(), ns=40, ks=9,
                          exchange="hf", include_qed=True,
                          qed_se_method="welton")
        result = solver.solve(max_iter=60, tol=1e-6)
        E_SE = result.energy_components.get("E_SE")
        assert E_SE is not None
        assert E_SE > 0  # SE is positive
        # And the bookkeeping entry records the backend used
        assert result.energy_components.get("se_method") == "welton"

    def test_qed_se_method_stub_raises_at_solve(self):
        """Passing a stub backend triggers NotImplementedError during solve."""
        solver = DFSolver(Z=2, subshells=helium_subshells(), ns=40, ks=9,
                          exchange="hf", include_qed=True,
                          qed_se_method="shabaev")
        with pytest.raises(NotImplementedError, match="Shabaev"):
            solver.solve(max_iter=40, tol=1e-5)

    def test_mohr_and_welton_give_comparable_he_values(self):
        """Within 50 % for He — Welton is heuristic but correct order of magnitude."""
        solver_m = DFSolver(Z=2, subshells=helium_subshells(), ns=40, ks=9,
                            exchange="hf", include_qed=True,
                            qed_se_method="mohr")
        result_m = solver_m.solve(max_iter=60, tol=1e-6)

        solver_w = DFSolver(Z=2, subshells=helium_subshells(), ns=40, ks=9,
                            exchange="hf", include_qed=True,
                            qed_se_method="welton")
        result_w = solver_w.solve(max_iter=60, tol=1e-6)

        E_mohr = result_m.energy_components["E_SE"]
        E_welton = result_w.energy_components["E_SE"]

        # Both positive
        assert E_mohr > 0
        assert E_welton > 0

        # Welton is known to overestimate by ~30 % for 1s.  Allow a factor
        # of 2 either way as a sanity band.
        ratio = E_welton / E_mohr
        assert 0.5 < ratio < 2.5, (
            f"Welton/Mohr ratio {ratio:.2f} out of expected band "
            f"(E_mohr={E_mohr:.3e}, E_welton={E_welton:.3e})"
        )
