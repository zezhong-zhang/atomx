"""Non-relativistic Hartree-Fock method (Schrodinger equation + exchange).

Uses LS-coupling (Russell-Saunders coupling). The angular quantum numbers
are (n, l) without j-splitting — all electrons in a given (n, l) subshell
are degenerate.

This wraps the ODE shooting solver with ``equation='schrodinger'``.
"""

from __future__ import annotations

from dataclasses import dataclass

from atomx.methods.base import Method
from atomx.types import CouplingScheme, NuclearModel, TheoryLevel


@dataclass
class HartreeFock(Method):
    """Non-relativistic Hartree-Fock with local Slater exchange.

    Solves the Schrodinger equation self-consistently using the
    shooting method with Slater's local exchange approximation.

    Students see: how exchange modifies orbital energies compared to
    the independent-particle model, all within a non-relativistic framework.

    Parameters
    ----------
    nuclear : NuclearModel
        Nuclear model.  Point nucleus is most consistent with
        non-relativistic theory.
    alpha_x : float
        Slater exchange parameter (default 2/3 * 3/2 = 1.0).
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

        # Tag result with theory level
        result._equation = "schrodinger"
        return result
