"""Multi-Configuration Dirac-Hartree-Fock method.

Captures electron correlation by mixing multiple configuration
state functions (CSFs) via CI diagonalization with orbitals
optimized for the weighted average of states.
"""

from __future__ import annotations

from dataclasses import dataclass

from atomx.methods.base import Method
from atomx.types import Correlation, NuclearModel, TheoryLevel


@dataclass
class MCDHF(Method):
    """Multi-Configuration Dirac-Hartree-Fock.

    Generates CSFs from a reference configuration via single and/or
    double excitations, then solves the MCDHF equations self-consistently.

    Parameters
    ----------
    correlation : Correlation
        SINGLES or SINGLES_DOUBLES excitation level.
    breit : bool
        Include Breit interaction.
    qed : bool
        Include QED corrections.
    basis_order : int
        B-spline order (default 9).
    """
    correlation: Correlation = Correlation.SINGLES_DOUBLES
    breit: bool = False
    qed: bool = False
    basis_order: int = 9

    @property
    def theory_level(self) -> TheoryLevel:
        return TheoryLevel.MCDHF

    def solve(self, Z, config_strings, tol=1e-5, max_iter=40,
              ns=113, **kwargs):
        """Run MCDHF calculation.

        Parameters
        ----------
        Z : int
            Nuclear charge.
        config_strings : list of str
            Configuration strings to include.
        """
        from atomx.mcdhf.config import make_csf_list
        from atomx.mcdhf.solver import MCDHFSolver

        csfs = make_csf_list(Z, config_strings)
        solver = MCDHFSolver(
            Z=Z, csfs=csfs, ns=ns, ks=self.basis_order,
            include_breit=self.breit, include_qed=self.qed)
        return solver.solve(max_iter=max_iter, scf_tol=tol)
