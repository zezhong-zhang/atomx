"""Relativistic Dirac-Hartree-Fock method (B-spline basis).

Solves the four-component Dirac equation self-consistently.  Unlike
the Breit-Pauli approach, relativity is treated variationally: scalar
relativistic corrections, spin-orbit coupling, and the Darwin term all
emerge naturally from the Dirac equation structure.

The orbital wavefunctions have two radial components:
    P(r) — the large (upper) component
    Q(r) — the small (lower) component, typically |Q| ~ (v/c)|P|

Uses jj-coupling: each electron has a well-defined total angular
momentum j = l +/- 1/2, characterized by the Dirac quantum number kappa.
"""

from __future__ import annotations

from dataclasses import dataclass

from atomx.methods.base import Method
from atomx.types import Exchange, NuclearModel, StartingPotential, TheoryLevel


@dataclass
class DiracFock(Method):
    """Relativistic Dirac-Fock self-consistent field method.

    Parameters
    ----------
    exchange : Exchange
        Electron exchange approximation:
        - ``Exchange.HARTREE_FOCK`` (default): exact non-local exchange
          via Y^k Slater integrals.  Accurate but slower.
        - ``Exchange.SLATER``: local density exchange V_x ~ -rho^{1/3}.
          Faster but approximate.
    breit : bool
        Include the Breit interaction — magnetic interaction between
        electrons and retardation of the Coulomb interaction.  Important
        for inner-shell energies of heavy atoms (Z > 30).
    qed : bool
        Include QED corrections: vacuum polarization (Uehling potential)
        and electron self-energy (Lamb shift from Mohr tables).
    bspline_order : int
        B-spline polynomial order (default 9, i.e. degree 8).
        Higher order gives better accuracy but larger matrices.
    starting_potential : StartingPotential or None
        Approximate potential for the initial SCF guess.
        Auto-selects TFDA for Z >= 20 if None.
    """
    exchange: Exchange = Exchange.HARTREE_FOCK
    breit: bool = False
    qed: bool = False
    bspline_order: int = 9
    starting_potential: StartingPotential = None

    @property
    def theory_level(self) -> TheoryLevel:
        if self.qed:
            return TheoryLevel.DIRAC_FOCK_QED
        if self.breit:
            return TheoryLevel.DIRAC_FOCK_BREIT
        if self.exchange == Exchange.SLATER:
            return TheoryLevel.DIRAC_FOCK_SLATER
        return TheoryLevel.DIRAC_HARTREE_FOCK

    def solve(self, Z, subshells, tol=1e-5, max_iter=40,
              jj_configs=None, use_physics_grid=True, ns=113, **kwargs):
        """Run Dirac-Fock SCF calculation."""
        from atomx.scf.solver import DFSolver

        starting = None
        if self.starting_potential is not None:
            starting = self.starting_potential.value
        elif Z >= 20:
            starting = "tfda"

        solver = DFSolver(
            Z=Z, subshells=subshells,
            ns=ns, ks=self.bspline_order,
            exchange=self.exchange.value,
            include_breit=self.breit,
            include_qed=self.qed,
            nuclear=self.nuclear.value,
            starting_potential=starting,
            jj_configs=jj_configs)

        if use_physics_grid:
            from atomx.bspline.knots import make_physics_knots
            from atomx.bspline.basis import prepare_bspline_data
            knots, ns_grid = make_physics_knots(
                ks=self.bspline_order, Z=Z, nuclear=self.nuclear.value)
            bsp_data = prepare_bspline_data(
                knots, ns_grid, self.bspline_order, kinetic_balance=True)
            solver.t = knots
            solver.bsp_data = bsp_data
            solver.ns = ns_grid
            solver._precompute()

        result = solver.solve(max_iter=max_iter, tol=tol)
        result._theory_level = self.theory_level.value
        result._method_name = self.name
        if not result._bsp_data:
            result._bsp_data = solver.bsp_data
        return result
