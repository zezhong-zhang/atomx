"""Large-grid Coulomb factorization preserves the converged DHF state."""

import pytest

import atomx.numerics.bspline.convol as convol
from atomx.methods.dirac_fock import DFSolver, neon_subshells


@pytest.mark.slow
def test_large_basis_exchange_matches_full_cell_bank(monkeypatch):
    def neon():
        return DFSolver(Z=10, subshells=neon_subshells(), ns=256, ks=9,
                        r_max=50, exchange="hf")

    monkeypatch.setattr(convol, "SEMISEPARABLE_EXCHANGE_NS", 10**9)
    monkeypatch.setattr(convol, "SEMISEPARABLE_DIRECT_NS", 10**9)
    full = neon().solve(max_iter=25, tol=1e-5)
    monkeypatch.setattr(convol, "SEMISEPARABLE_EXCHANGE_NS", 256)
    monkeypatch.setattr(convol, "SEMISEPARABLE_DIRECT_NS", 256)
    factored_solver = neon()
    factored = factored_solver.solve(max_iter=25, tol=1e-5)

    assert full.converged and factored.converged
    assert factored.iterations == full.iterations
    assert abs(factored.total_energy - full.total_energy) < 1e-9
    assert any(len(key) == 3 and key[-1] == "semiseparable"
               for key in factored_solver._convol_cache._cache)


@pytest.mark.slow
def test_mixed_exchange_avoids_full_banks_in_batch_scf(monkeypatch):
    def neon():
        return DFSolver(Z=10, subshells=neon_subshells(), ns=320, ks=9,
                        r_max=50, exchange="hf", kinetic_balance=True)

    monkeypatch.setattr(convol, "SEMISEPARABLE_MIXED_EXCHANGE_NS", 10**9)
    full_solver = neon()
    full = full_solver.solve(max_iter=3, tol=1e-5)
    # Release the ns=320 full-basis banks (shared process-wide on this
    # grid) before the compact solve, so both do not sit in memory at once.
    full_solver._convol_cache.release_shared_banks()

    monkeypatch.setattr(convol, "SEMISEPARABLE_MIXED_EXCHANGE_NS", 320)
    compact_solver = neon()
    compact = compact_solver.solve(max_iter=3, tol=1e-5)
    assert abs(compact.total_energy - full.total_energy) < 1e-9
    assert not any(len(key) == 3 and key[-1] == "T"
                   for key in compact_solver._convol_cache._cache)
