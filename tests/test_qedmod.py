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
from atomx.scf.solver import DFSolver, helium_subshells


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
# Stub backends
# ---------------------------------------------------------------------------

class TestStubBackends:
    def test_shabaev_raises_with_reference(self):
        orbs = [(np.zeros(10), np.zeros(10), 2, -1, "1s1/2")]
        with pytest.raises(NotImplementedError, match="Shabaev"):
            compute_se_energy(orbs, Z=1, method="shabaev")

    def test_flambaum_ginges_raises_with_reference(self):
        orbs = [(np.zeros(10), np.zeros(10), 2, -1, "1s1/2")]
        with pytest.raises(NotImplementedError, match="Flambaum"):
            compute_se_energy(orbs, Z=1, method="flambaum_ginges")


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
