"""Non-relativistic Hartree-Fock method (Schrodinger equation + exchange).

Solves the radial Schrodinger equation self-consistently with local
Slater exchange.  Uses LS-coupling (Russell-Saunders): the angular
quantum numbers are (n, l) without j-splitting.

This is the standard non-relativistic many-electron method where the
Pauli exclusion principle is enforced through antisymmetric wavefunctions,
giving rise to exchange interactions between electrons.
"""

from __future__ import annotations

from dataclasses import dataclass

from atomx.methods.base import Method
from atomx.types import CouplingScheme, NuclearModel, TheoryLevel


@dataclass
class HartreeFock(Method):
    """Non-relativistic Hartree-Fock with local Slater exchange.

    Solves the Schrodinger equation self-consistently using the ODE
    shooting method.  Each orbital sees the mean-field potential from
    all other electrons (Hartree potential) plus a local approximation
    to the exchange interaction (Slater exchange).

    Parameters
    ----------
    nuclear : NuclearModel
        Nuclear charge distribution.  POINT is most consistent with
        non-relativistic theory.
    slater_exchange_alpha : float
        Slater exchange scaling parameter.  alpha=1.0 gives the
        free-electron gas exchange.  alpha=2/3 is the Kohn-Sham value.
    coupling : CouplingScheme
        Angular momentum coupling scheme (LS by default).
    """
    slater_exchange_alpha: float = 1.0
    coupling: CouplingScheme = CouplingScheme.LS

    @property
    def theory_level(self) -> TheoryLevel:
        return TheoryLevel.HARTREE_FOCK

    def solve(self, Z, subshells, tol=1e-5, max_iter=40, **kwargs):
        """Run non-relativistic Hartree-Fock-Slater via ODE shooting."""
        from atomx.shooting.solver import ShootingSolver

        solver = ShootingSolver(
            Z=Z, subshells=subshells,
            n_grid=1000, r_max=75.0,
            alpha_x=self.slater_exchange_alpha,
            equation='schrodinger')
        result = solver.solve(max_iter=max_iter, tol=tol)
        result._equation = "schrodinger"
        return result
