"""M3 tests: SCF solver — both Slater (local) and HF (non-local) exchange.

Reference values (Desclaux 1973, Dirac-Hartree-Fock):
- He total energy: -2.862 Hartree
- Ne total energy: -128.547 Hartree

Ne HF tests share a single solver to avoid repeating the expensive SCF.
"""

import numpy as np
import pytest
from atomx.methods.dirac_fock import DFSolver, Subshell, helium_subshells, neon_subshells
from atomx.angular.coefficients import exchange_lambda, exchange_k_range, exchange_coefficient


# ---- Shared fixtures (run SCF ONCE, reuse across tests) ----

@pytest.fixture(scope="module")
def he_hf_result():
    """He DHF result — computed once for the module."""
    solver = DFSolver(Z=2, subshells=helium_subshells(), ns=40, ks=9, exchange="hf")
    return solver.solve(max_iter=60, tol=1e-6)


@pytest.fixture(scope="module")
def ne_hf_result():
    """Ne DHF result — computed once for the module."""
    solver = DFSolver(Z=10, subshells=neon_subshells(), ns=50, ks=9, exchange="hf")
    return solver.solve(max_iter=40, tol=1e-5)


# ---- Angular coefficient tests ----

class TestAngularCoeffs:
    def test_ss_exchange_k0(self):
        assert exchange_k_range(-1, -1) == [0]

    def test_sp_exchange_k1(self):
        assert 1 in exchange_k_range(-1, 1)

    def test_sp32_exchange(self):
        k_vals = exchange_k_range(-1, -2)
        assert 1 in k_vals
        assert 0 not in k_vals

    def test_lambda_nonzero(self):
        assert exchange_lambda(-1, -1, 0) > 0

    def test_exchange_coeff_negative(self):
        assert exchange_coefficient(-1, -1, 0, 2) < 0


# ---- Slater exchange tests ----

class TestSCFSlater:
    def test_helium_slater_converges(self):
        solver = DFSolver(Z=2, subshells=helium_subshells(), ns=40, ks=9,
                          exchange="slater")
        result = solver.solve(max_iter=50, tol=1e-5)
        assert result.converged

    def test_helium_slater_energy_range(self):
        solver = DFSolver(Z=2, subshells=helium_subshells(), ns=40, ks=9,
                          exchange="slater")
        result = solver.solve(max_iter=50, tol=1e-5)
        assert -3.5 < result.total_energy < -2.0

    def test_helium_slater_energy_components(self):
        solver = DFSolver(Z=2, subshells=helium_subshells(), ns=40, ks=9,
                          exchange="slater")
        result = solver.solve(max_iter=50, tol=1e-5)
        E = result.energy_components
        assert E["E_1e"] < 0
        assert E["E_J"] > 0
        # E_K is the HF-style exchange magnitude evaluated on the Slater
        # orbitals (matching ``_compute_total_energy``'s HF convention, which
        # is what shooting's ``_expval`` also does for its DFS orbitals).
        # For closed-shell He 1s², there is no same-spin exchange pair, so
        # ``E_K == 0``.  The old buggy LDA-integral assertion ``E_K > 0``
        # reflected the pre-refactor convention; the positive-magnitude
        # property of ``E_K`` still holds for multi-shell atoms (see
        # ``tests/test_dfsolver_slater.py``).
        assert E["E_K"] >= 0


# ---- HF exchange tests (shared fixtures) ----

class TestSCFHartreeFock:
    def test_helium_hf_converges(self, he_hf_result):
        assert he_hf_result.converged
        assert he_hf_result.iterations < 30

    def test_helium_hf_total_energy(self, he_hf_result):
        assert he_hf_result.total_energy == pytest.approx(-2.862, abs=0.05), \
            f"He DHF E={he_hf_result.total_energy:.4f}"

    def test_helium_hf_1s_energy(self, he_hf_result):
        E_1s = he_hf_result.orbital_energies["1s1/2"]
        assert E_1s == pytest.approx(-0.918, abs=0.05)

    def test_neon_hf_converges(self, ne_hf_result):
        assert ne_hf_result.converged, \
            f"Ne: residual={ne_hf_result.residual:.2e}, iter={ne_hf_result.iterations}"
        assert ne_hf_result.iterations < 30

    def test_neon_hf_total_energy(self, ne_hf_result):
        # Tight accuracy pin (was ±1.5 Ha, ~useless). atomx's Ne DHF total
        # energy is deterministic and fully basis-converged (identical to 8
        # digits from ns=50 up to ns=150) at -128.69197 Ha, within the
        # accepted point-nucleus DHF range (~-128.692). abs=1e-3 catches any
        # >1 mHa (0.027 eV) accuracy regression while tolerating grid/tol
        # details. Confirmed bit-identical across fresh processes 2026-07-10.
        assert ne_hf_result.total_energy == pytest.approx(-128.6920, abs=1.0e-3), \
            f"Ne DHF E={ne_hf_result.total_energy:.5f} (expected ~-128.69197)"

    def test_neon_hf_orbital_ordering(self, ne_hf_result):
        assert ne_hf_result.orbital_energies["1s1/2"] < ne_hf_result.orbital_energies["2s1/2"]

    def test_neon_hf_1s_energy(self, ne_hf_result):
        # Tight pin (was ±2.0 Ha). Deterministic converged 1s eigenvalue
        # -32.81748 Ha; abs=1e-2 catches any >10 mHa regression.
        assert ne_hf_result.orbital_energies["1s1/2"] == pytest.approx(-32.8175, abs=1.0e-2)

    def test_neon_hf_energy_components(self, ne_hf_result):
        E = ne_hf_result.energy_components
        assert E["E_1e"] < 0
        assert E["E_J"] > 0   # direct Coulomb (F^0)
        assert E["E_K"] > 0   # exchange (F^{k>0}, G^k)
        assert E["E_2e"] == pytest.approx(E["E_J"] - E["E_K"])
