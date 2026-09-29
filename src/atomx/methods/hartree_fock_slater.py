"""Non-relativistic Hartree–Fock–Slater with local density exchange.

Uses LS-coupling.  Orbital quantum numbers are (n, l) with no j-splitting.
Wraps the ODE shooting solver with ``equation='schrodinger'``.
"""

from __future__ import annotations

from dataclasses import dataclass

from atomx.methods.base import Method
from atomx.types import CouplingScheme, TheoryLevel


@dataclass
class HartreeFockSlater(Method):
    """Non-relativistic HF–Slater with local exchange.

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

    def solve(self, Z, subshells, tol=1e-5, max_iter=40, *,
              n_grid=1000, r_max=75.0, r_inner=1e-7, screening_potential=None):
        """Run non-relativistic HF-Slater via ODE shooting."""
        if self.coupling != CouplingScheme.LS:
            raise NotImplementedError('this local-reference method supports LS coupling only')
        from atomx.numerics.shooting.local import ShootingSolver

        solver = ShootingSolver(
            Z=Z, subshells=subshells,
            n_grid=n_grid, r_max=r_max, r_inner=r_inner,
            nuclear=getattr(self.nuclear, "value", self.nuclear),
            screening_potential=screening_potential,
            alpha_x=self.alpha_x,
            equation='schrodinger')
        result = solver.solve(max_iter=max_iter, tol=tol)
        from dataclasses import replace
        result.calculation_spec = replace(result.calculation_spec, method="HartreeFockSlater")
        result.method = "HartreeFockSlater"
        result._equation = "schrodinger"
        return result
