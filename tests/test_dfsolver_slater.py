"""Regression tests for the B-spline DFSolver with local Slater exchange.

Prior to this fix the B-spline DFS path had three independent bugs that
together gave Ne ~2× too-deep binding (–225 Ha vs shooting's –128.62 Ha):

1. ``V_x`` used ``ρ = P² + Q²`` (radial density per ``dr``) where the
   Slater formula ``V_x = -α (3 ρ_vol / π)^{1/3}`` wants the volumetric
   charge density ``ρ_vol = (P²+Q²) / (4π r²)``.
2. No Latter tail correction: pure LDA ``V_x`` decays too fast, so
   the neutral-atom asymptote ``V_tot → -1/r`` is lost, valence
   orbitals stop being bound, and the SCF collapses.
3. No potential-mixing damping in the Slater SCF loop: shooting mixes
   the potential (not the orbitals) with ``factor = 0.2``; without it,
   the B-spline loop root-swaps between iterations.

These tests pin the closed-shell energies against atomx's own shooting
backend (Salvat-style ODE DHFS) at 10⁻³ relative.  Shooting is the
canonical reference for local-exchange atoms in atomx, and the
matching ``α_x = 1.5`` is the Slater 1951 original value that both
backends share.
"""

import pytest

from atomx import Atom
from atomx.config.jj_coupling import generate_jj_average
from atomx.methods.dirac_fock import DFSolver
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.knots import make_physics_knots


SLATER_ALPHA = 1.5   # matches shooting's DHFS convention


def _dfsolver_slater(Z: int, config: str, alpha_x: float = SLATER_ALPHA,
                     max_iter: int = 200, tol: float = 1e-8):
    """Build a physics-grid DFSolver pinned to the shooting α convention."""
    subshells, _ = generate_jj_average(config, averaging="statistical")
    subshells = [s for s in subshells if s.occupation > 1e-10]
    solver = DFSolver(
        Z=Z, subshells=subshells, exchange="slater", alpha_x=alpha_x,
        starting_potential=("tfda" if Z >= 20 else None),
    )
    t_phys, ns_phys = make_physics_knots(ks=solver.ks, Z=Z, nuclear="point")
    solver.t = t_phys
    solver.bsp_data = prepare_bspline_data(
        t_phys, ns_phys, solver.ks, kinetic_balance=True
    )
    solver.ns = ns_phys
    solver._precompute()
    return solver.solve(max_iter=max_iter, tol=tol)


@pytest.fixture(scope="module")
def ne_shooting_total_energy():
    return Atom(Z=10, config="1s2 2s2 2p6",
                exchange="shooting").solve().total_energy


@pytest.fixture(scope="module")
def ar_shooting_total_energy():
    return Atom(Z=18, config="1s2 2s2 2p6 3s2 3p6",
                exchange="shooting").solve().total_energy


class TestBsplineSlaterClosedShell:
    def test_neon_slater_converges(self):
        result = _dfsolver_slater(10, "1s2 2s2 2p6")
        assert result.converged

    def test_neon_slater_matches_shooting(self, ne_shooting_total_energy):
        result = _dfsolver_slater(10, "1s2 2s2 2p6")
        rel = abs(result.total_energy - ne_shooting_total_energy) / abs(
            ne_shooting_total_energy
        )
        assert rel < 1e-3, (
            f"Ne Slater B-spline E={result.total_energy:.4f} vs shooting "
            f"E={ne_shooting_total_energy:.4f} (rel={rel:.2e})"
        )

    def test_argon_slater_converges(self):
        result = _dfsolver_slater(18, "1s2 2s2 2p6 3s2 3p6")
        assert result.converged

    def test_argon_slater_matches_shooting(self, ar_shooting_total_energy):
        result = _dfsolver_slater(18, "1s2 2s2 2p6 3s2 3p6")
        rel = abs(result.total_energy - ar_shooting_total_energy) / abs(
            ar_shooting_total_energy
        )
        assert rel < 1e-3, (
            f"Ar Slater B-spline E={result.total_energy:.4f} vs shooting "
            f"E={ar_shooting_total_energy:.4f} (rel={rel:.2e})"
        )

    def test_neon_slater_energy_components_signs(self):
        """Slater E_K is the positive magnitude |E_x|; the three
        Coulomb pieces always satisfy E_1e<0, E_J>0, E_K>=0.
        """
        result = _dfsolver_slater(10, "1s2 2s2 2p6")
        E = result.energy_components
        assert E["E_1e"] < 0
        assert E["E_J"] > 0
        assert E["E_K"] >= 0


class TestDFSolverKineticBalanceOption:
    """RKB is a user-selectable option on DFSolver; default preserves the
    historical unbalanced basis exactly (regression-safe)."""

    def _ne(self, **kw):
        from atomx.methods.dirac_fock import neon_subshells
        s = DFSolver(Z=10, subshells=neon_subshells(), ns=50, ks=9,
                     exchange="hf", **kw)
        return s, s.solve(max_iter=80, tol=1e-8)

    def test_default_is_unbalanced(self):
        s, _ = self._ne()
        assert s.kinetic_balance is False
        assert s.bsp_data["ksp"] == s.bsp_data["ks"]  # ksp == ksq == ks

    def test_rkb_uses_reduced_order_p_basis(self):
        s, _ = self._ne(kinetic_balance=True)
        assert s.kinetic_balance is True
        assert s.bsp_data["ksp"] == s.bsp_data["ks"] - 1  # P order = ks-1
        assert s.bsp_data["ksq"] == s.bsp_data["ks"]      # Q order = ks

    def test_rkb_ground_state_energy_matches_unbalanced(self):
        """RKB removes spurious states without shifting the DHF energy:
        Ne agrees to ~1e-5 (here machine-precision-close)."""
        _, r0 = self._ne()
        _, r1 = self._ne(kinetic_balance=True)
        rel = abs(r0.total_energy - r1.total_energy) / abs(r0.total_energy)
        assert rel < 1e-4
