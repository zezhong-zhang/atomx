"""M3 tests: SCF solver — both Slater (local) and HF (non-local) exchange.

Reference values (point nucleus, Dirac-Hartree-Fock, no Breit/QED), with the
source named per number:

- He DHF total energy: **-2.86181332 Ha** (DBSR_HF, ``benchmarks/
  EXTERNAL_VERIFICATION.md`` row "Z=2 He"; atomx reproduces it to 2.3e-10).
  The *non-relativistic* HF limit is a different number, -2.8616800 Ha
  (Froese Fischer; the value ``tests/test_mbpt2.py`` sandwiches He with).
- Ne DHF total energy: **-128.69192580 Ha** (DBSR_HF ``an=10 term=AV
  mbreit=0``, same file; quoted in
  ``tests/test_atom_surface_equivalence.py:190``), i.e. the classic
  Desclaux 1973 value -128.6919 to 7 digits
  (``docs/PHYSICS_NUMERICS_REVIEW_2026-09-19.md``).

This module previously cited "Desclaux 1973 DHF: Ne total energy -128.547
Hartree".  -128.5471 is the **non-relativistic HF limit** of Ne, not its DHF
energy, and it misses the assertion at ``test_neon_hf_total_energy`` by
0.145 Ha — 145x that test's abs=1e-3 tolerance.  The value the test actually
gates is atomx's own point-nucleus DHF total, -128.69196949 Ha (3.05e-5 from
the DBSR_HF reference above; the ns=50 physics grid is what sets that gap).

Ne HF tests share a single solver to avoid repeating the expensive SCF.
"""

import pytest

from atomx.angular.coefficients import (
    exchange_coefficient,
    exchange_k_range,
    exchange_lambda,
)
from atomx.methods.dirac_fock import DFSolver, helium_subshells, neon_subshells

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

    @pytest.mark.snapshot
    def test_helium_hf_total_energy(self, he_hf_result):
        # Tight pin (was ±0.05 Ha ≈ 1.4 eV). Deterministic, basis-converged
        # He DHF total -2.861813 Ha, consistent with the DBSR_HF reference
        # -2.86181332 (benchmarks/EXTERNAL_VERIFICATION.md). abs=1 mHa
        # catches real regressions.
        assert he_hf_result.total_energy == pytest.approx(-2.8618, abs=1.0e-3), \
            f"He DHF E={he_hf_result.total_energy:.5f} (expected ~-2.86181)"

    @pytest.mark.snapshot
    def test_helium_hf_1s_energy(self, he_hf_result):
        # Tight pin (was ±0.05). Deterministic 1s eigenvalue -0.917991 Ha.
        E_1s = he_hf_result.orbital_energies["1s1/2"]
        assert E_1s == pytest.approx(-0.91799, abs=1.0e-2)

    def test_neon_hf_converges(self, ne_hf_result):
        assert ne_hf_result.converged, \
            f"Ne: residual={ne_hf_result.residual:.2e}, iter={ne_hf_result.iterations}"
        assert ne_hf_result.iterations < 30

    @pytest.mark.snapshot
    def test_neon_hf_total_energy(self, ne_hf_result):
        # Tight accuracy pin (was ±1.5 Ha, ~useless). atomx's Ne DHF total
        # energy is deterministic and fully basis-converged (identical to 8
        # digits from ns=50 up to ns=150) at -128.69197 Ha, 3.05e-5 Ha from
        # the DBSR_HF point-nucleus DHF reference -128.69192580 Ha.
        # abs=1e-3 catches any >1 mHa (0.027 eV) accuracy regression while
        # tolerating grid/tol details.  Confirmed bit-identical across fresh
        # processes 2026-07-10.
        #
        # The SAME Ne ground state is pinned four ways in this suite; they are
        # four different physics questions, not four disagreeing answers:
        #   -128.69197  point nucleus, HF exchange,  B-spline DFSolver  <- here
        #               (external check: DBSR_HF -128.69192580)
        #   -128.6299783  FERMI nucleus, HF exchange, B-spline
        #               (test_atom_surface_equivalence.py:32/105)
        #   -128.623390   point nucleus, Slater alpha=1.5, shooting
        #               (test_slater_integral_slag6.py:146; Salvat DHFS.f)
        #   -128.623437   Fermi nucleus, Slater alpha=1.5, B-spline
        #               (test_exchange_model_is_honoured.py:9)
        # The 0.062 Ha jump from the first to the second is the finite
        # nucleus; the 0.068 Ha from HF to Slater alpha=1.5 is the local
        # exchange model.
        assert ne_hf_result.total_energy == pytest.approx(-128.6920, abs=1.0e-3), \
            f"Ne DHF E={ne_hf_result.total_energy:.5f} (expected ~-128.69197)"

    def test_neon_hf_orbital_ordering(self, ne_hf_result):
        assert ne_hf_result.orbital_energies["1s1/2"] < ne_hf_result.orbital_energies["2s1/2"]

    @pytest.mark.snapshot
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
