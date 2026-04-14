"""Multi-Configuration Dirac-Hartree-Fock (MCDHF) method.

Goes beyond single-configuration Dirac-Fock by mixing multiple
configuration state functions (CSFs) to capture electron correlation.

The MCDHF wavefunction is a linear combination:
    |Psi_alpha> = sum_r c_r^alpha |Phi_r>

where |Phi_r> are CSFs built from a common set of orbitals, and the
mixing coefficients c_r^alpha and orbitals are simultaneously optimized
via the average-level approach.
"""

from __future__ import annotations

from dataclasses import dataclass

from atomx.methods.base import Method
from atomx.types import Correlation, NuclearModel, TheoryLevel


@dataclass
class MCDHF(Method):
    """Multi-Configuration Dirac-Hartree-Fock.

    Generates CSFs from a reference configuration via single and/or
    double excitations into virtual orbitals, then iteratively:
    1. Optimizes shared orbitals for the weighted average of all states
    2. Diagonalizes the CI matrix to get state energies and mixing

    Parameters
    ----------
    correlation : Correlation
        Excitation level for CSF generation:
        - ``Correlation.SINGLES``: single excitations only.
        - ``Correlation.SINGLES_DOUBLES`` (default): single + double.
    breit : bool
        Include Breit interaction in the CI Hamiltonian.
    qed : bool
        Include QED corrections.
    bspline_order : int
        B-spline polynomial order (default 9).
    """
    correlation: Correlation = Correlation.SINGLES_DOUBLES
    breit: bool = False
    qed: bool = False
    bspline_order: int = 9

    @property
    def theory_level(self) -> TheoryLevel:
        return TheoryLevel.MCDHF

    def solve(self, Z, config_strings, tol=1e-5, max_iter=40,
              ns=113, **kwargs):
        """Run MCDHF calculation.

        Parameters
        ----------
        Z : int
            Atomic number (nuclear charge).
        config_strings : list of str
            Configuration strings (e.g. ["1s2 2s2", "1s2 2p2"]).
        tol : float
            SCF convergence tolerance (Hartree).
        max_iter : int
            Maximum number of MCDHF macro-iterations.
        """
        from atomx.mcdhf.config import make_csf_list
        from atomx.mcdhf.solver import MCDHFSolver

        csfs = make_csf_list(Z, config_strings)
        solver = MCDHFSolver(
            Z=Z, csfs=csfs, ns=ns, ks=self.bspline_order,
            include_breit=self.breit, include_qed=self.qed)
        return solver.solve(max_iter=max_iter, scf_tol=tol)
