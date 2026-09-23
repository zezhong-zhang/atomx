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

These tests gate the closed-shell energies two ways:

1. **External oracle (B).** ``test_neon_slater_matches_salvat_dhfs`` compares
   the B-spline Slater SCF to Salvat's independently written Fortran
   ``radial/DHFS`` program (α = 1.5, point nucleus, 1000-point grid,
   R_max = 75 a₀, Ne 1s² 2s² 2p⁶ → −128.623390 Ha).  Run 2026-09-22:
   ``./DHFS < Z010.in`` gives ``Total binding energy = -1.2862339E+02 Eh``
   for the point nucleus.  atomx B-spline: −128.6234601, i.e. 7.0e-5 Ha
   apart.
2. **Cross-engine (C).** ``test_neon/argon_slater_matches_shooting`` compares
   the B-spline Galerkin SCF to the shooting ODE solver.  The two engines
   share the jj-average angular coefficients and the α_x = 1.5 value; they
   share no radial kernel (Galerkin B-spline matrices vs the Salvat-style
   ODE integrator), so an error in either kernel shows up.  Gates are
   absolute, sized at ~2x the measured agreement: Ne 1.049e-4 Ha
   (rel 8.16e-7), Ar 1.538e-5 Ha (rel 2.91e-8).

The docstring's bug history is a 2x error (−225 vs −128.6 Ha), which is what
a 1e-3 *relative* gate could catch and nothing finer; the absolute gates
above are 6000x and 34000x tighter.
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
    # Correction-free: the DFSolver reference below runs bare Slater
    # SCF, so the shooting side must too (breit/qed now default True).
    return Atom(
                Z=10,
                config="1s2 2s2 2p6",
                engine="shooting",
                exchange="slater",
                corrections="none",
           ).solve().total_energy


@pytest.fixture(scope="module")
def ar_shooting_total_energy():
    return Atom(
                Z=18,
                config="1s2 2s2 2p6 3s2 3p6",
                engine="shooting",
                exchange="slater",
                corrections="none",
           ).solve().total_energy


class TestBsplineSlaterClosedShell:
    def test_neon_slater_converges(self):
        result = _dfsolver_slater(10, "1s2 2s2 2p6")
        assert result.converged

    def test_neon_slater_matches_shooting(self, ne_shooting_total_energy):
        result = _dfsolver_slater(10, "1s2 2s2 2p6")
        diff = abs(result.total_energy - ne_shooting_total_energy)
        # Measured 1.0492e-4 Ha (rel 8.16e-7) on 2026-09-22 with the
        # half-weight Y^k rule; after the interpolatory partial-interval
        # fix (2026-09-23) 2.779e-5 Ha — the B-spline side moved 3.8x closer
        # to the shooting engine.  Gate 2.0e-4 (was 0.129 Ha, 1226x looser,
        # before the 2026-09-22 audit).
        assert diff < 2.0e-4, (
            f"Ne Slater B-spline E={result.total_energy:.6f} vs shooting "
            f"E={ne_shooting_total_energy:.6f} (abs={diff:.3e} Ha)"
        )

    def test_neon_slater_matches_salvat_dhfs(self):
        """External oracle: Salvat's ``radial/DHFS``, α = 1.5, point nucleus.

        ``~/code/ASAP/radial/DHFS/Z010.in`` (Ne 1s² 2s² 2p⁶, α = 1.5,
        point nucleus, 1000-point grid, R_max = 75 a₀) prints
        ``Total binding energy = -1.2862339E+02 Eh``; the same number is
        recorded in ``benchmarks/results_shooting.csv`` and used by
        ``tests/test_slater_integral_slag6.py``.  It is a *different code*
        (independent Fortran implementation of the same DHFS model), not an
        atomx output.  Measured atomx B-spline Slater: −128.6234601 Ha,
        7.0e-5 Ha away; after the interpolatory partial-interval rule
        (2026-09-23) −128.6233830 Ha, 7.0e-6 Ha away — a 10x improvement
        from removing the Y^k quadrature bias.  Gate 5.0e-5 Ha = 7x the
        measurement.
        """
        result = _dfsolver_slater(10, "1s2 2s2 2p6")   # point nucleus
        salvat_point = -128.623390
        diff = abs(result.total_energy - salvat_point)
        assert diff < 5.0e-5, (
            f"Ne Slater B-spline E={result.total_energy:.7f} Ha vs Salvat "
            f"radial/DHFS {salvat_point:.6f} Ha (abs={diff:.3e} Ha)"
        )

    def test_argon_slater_converges(self):
        result = _dfsolver_slater(18, "1s2 2s2 2p6 3s2 3p6")
        assert result.converged

    def test_argon_slater_matches_shooting(self, ar_shooting_total_energy):
        result = _dfsolver_slater(18, "1s2 2s2 2p6 3s2 3p6")
        diff = abs(result.total_energy - ar_shooting_total_energy)
        # Measured 1.538e-5 Ha (rel 2.91e-8) on 2026-09-22 with the
        # half-weight Y^k rule; 1.882e-4 Ha after the interpolatory
        # partial-interval fix (2026-09-23).  The pre-fix tight agreement
        # was the Y^k quadrature bias (2.8e-5-class per integral)
        # coincidentally tracking the shooting engine's own discretization;
        # the fixed B-spline side is pinned independently (Salvat DHFS
        # agrees 10x better after the fix — test_neon_slater_matches_salvat_
        # dhfs — and the Y^k construction itself is gated against closed
        # forms at rel 1e-8), so the residual is the genuine
        # engine-vs-engine discretization gap, which is element-dependent.
        # Gate 5.0e-4 = 2.7x the measurement; the pre-fix 5e-5 gate encoded
        # the defective agreement.
        assert diff < 5.0e-4, (
            f"Ar Slater B-spline E={result.total_energy:.6f} vs shooting "
            f"E={ar_shooting_total_energy:.6f} (abs={diff:.3e} Ha)"
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
