"""Non-relativistic Hartree-Fock (Schrodinger equation + exchange).

Uses LS-coupling.  Orbital quantum numbers are (n, l) with no j-splitting.
Wraps the ODE shooting solver with ``equation='schrodinger'``.
"""

from __future__ import annotations

from dataclasses import dataclass

from atomx.methods.base import Method
from atomx.types import CouplingScheme, NuclearModel, TheoryLevel


@dataclass
class HartreeFock(Method):
    """Non-relativistic Hartree-Fock with local Slater exchange.

    Solves the Schrodinger equation self-consistently using ODE shooting.

    Parameters
    ----------
    alpha_x : float
        Slater exchange parameter.  alpha_x=1.0 gives the free-electron
        gas exchange; alpha_x=2/3 is the Kohn-Sham value.
    coupling : CouplingScheme
        Angular momentum coupling (LS by default).
    """
    alpha_x: float = 1.0
    coupling: CouplingScheme = CouplingScheme.LS

    @property
    def theory_level(self) -> TheoryLevel:
        return TheoryLevel.HARTREE_FOCK

    def solve(self, Z, subshells, tol=1e-5, max_iter=40, **kwargs):
        """Run non-relativistic HF-Slater via ODE shooting."""
        from atomx.shooting.solver import ShootingSolver

        solver = ShootingSolver(
            Z=Z, subshells=subshells,
            n_grid=1000, r_max=75.0,
            alpha_x=self.alpha_x,
            equation='schrodinger')
        result = solver.solve(max_iter=max_iter, tol=tol)
        result._equation = "schrodinger"
        return result
