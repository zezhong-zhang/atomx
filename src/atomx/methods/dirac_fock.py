"""Relativistic Dirac-Hartree-Fock method (B-spline basis).

Solves the Dirac equation self-consistently with either non-local
Hartree-Fock exchange or local Slater exchange.  Optionally includes
Breit interaction and QED corrections.

Uses jj-coupling naturally — spin-orbit effects emerge from the
Dirac equation structure rather than being added perturbatively.
"""

from __future__ import annotations

from dataclasses import dataclass

from atomx.methods.base import Method
from atomx.types import Exchange, NuclearModel, TheoryLevel


@dataclass
class DiracFock(Method):
    """Relativistic Dirac-Fock self-consistent field.

    The Dirac equation includes all relativistic effects variationally:
    scalar relativistic corrections, spin-orbit coupling, and the
    Darwin term are all embedded in the four-component formalism.

    Parameters
    ----------
    exchange : Exchange
        Exchange approximation:
        - HARTREE_FOCK (default): non-local, accurate
        - SLATER: local density approximation, faster
    breit : bool
        Include Breit (Gaunt) interaction — magnetic and retardation
        corrections to the Coulomb interaction.
    qed : bool
        Include QED corrections: vacuum polarization (Uehling)
        and self-energy (Mohr tables).
    basis_order : int
        B-spline order (default 9).
    starting_potential : str or None
        ``"tfda"`` or ``"gsz"`` for faster SCF convergence.
    """
    exchange: Exchange = Exchange.HARTREE_FOCK
    breit: bool = False
    qed: bool = False
    basis_order: int = 9
    starting_potential: str = None

    @property
    def theory_level(self) -> TheoryLevel:
        if self.qed:
            return TheoryLevel.DHF_QED
        if self.breit:
            return TheoryLevel.DHF_BREIT
        if self.exchange == Exchange.SLATER:
            return TheoryLevel.DIRAC_FOCK_SLATER
        return TheoryLevel.DIRAC_HARTREE_FOCK

    def solve(self, Z, subshells, tol=1e-5, max_iter=40,
              jj_configs=None, use_physics_grid=True, ns=113, **kwargs):
        """Run Dirac-Fock SCF calculation.

        This dispatches to the existing DFSolver with the appropriate
        parameters set from this method object.
        """
        from atomx.scf.solver import DFSolver

        starting = self.starting_potential
        if starting is None and Z >= 20:
            starting = "tfda"

        solver = DFSolver(
            Z=Z, subshells=subshells,
            ns=ns, ks=self.basis_order,
            exchange=self.exchange.value,
            include_breit=self.breit,
            include_qed=self.qed,
            nuclear=self.nuclear.value,
            starting_potential=starting,
            jj_configs=jj_configs)

        if use_physics_grid:
            from atomx.bspline.knots import make_physics_knots
            from atomx.dirac.matrix import prepare_bspline_data
            t_phys, ns_phys = make_physics_knots(
                ks=self.basis_order, Z=Z, nuclear=self.nuclear.value)
            data = prepare_bspline_data(
                t_phys, ns_phys, self.basis_order, kinetic_balance=True)
            solver.t = t_phys
            solver.bsp_data = data
            solver.ns = ns_phys
            solver._precompute()

        result = solver.solve(max_iter=max_iter, tol=tol)

        # Tag with theory level
        result._theory_level = self.theory_level.value
        result._method_name = self.name
        if not result._bsp_data:
            result._bsp_data = solver.bsp_data
        return result
