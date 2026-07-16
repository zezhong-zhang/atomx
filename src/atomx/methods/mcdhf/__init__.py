"""Multi-Configuration Dirac-Hartree-Fock (MCDHF).

Average-Level MCDHF: shared orbitals optimised for the weighted
average of configurations, then CI diagonalisation for state energies.

This package holds
    - :class:`MCDHF` — the user-facing :class:`~atomx.methods.Method`
      selectable via ``Atom.solve(MCDHF(...))``;
    - :class:`MCDHFSolver` — the iterative driver (orbital-opt loop +
      :func:`~atomx.methods.ci.ci_solve`);
    - :func:`make_csf_list`, :class:`CSF` — CSF generation utilities;
    - :func:`build_ci_matrix` — CI matrix assembly.

The CI diagonalisation proper lives one level up in
:mod:`atomx.methods.ci` so that pure-CI calculations on frozen DHF
orbitals (and perturbative correlation methods that reuse CI
machinery) can call it without importing the full MCDHF driver.
"""

from __future__ import annotations

from dataclasses import dataclass

from atomx.methods.base import Method
from atomx.methods.mcdhf.config import (
    CSF,
    count_j_resolved_csfs,
    make_csf_list,
)
from atomx.methods.mcdhf.matrix import build_ci_matrix
from atomx.methods.mcdhf.solver import MCDHFResult, MCDHFSolver
from atomx.types import Correlation, TheoryLevel


@dataclass
class MCDHF(Method):
    """Multi-Configuration Dirac–Hartree–Fock.

    Generates CSFs from a reference configuration via single and/or
    double excitations into virtual orbitals, then iteratively:

    1. Optimises shared orbitals for the weighted average of all
       states (Guerra average-level approach).
    2. Diagonalises the CI matrix to get state energies and mixing
       coefficients (via :func:`atomx.methods.ci.ci_solve`).

    Parameters
    ----------
    correlation : Correlation
        Excitation level for CSF generation.

        - ``Correlation.SINGLES`` — single excitations only.
        - ``Correlation.SINGLES_DOUBLES`` (default) — single + double.
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

    def solve(self, Z, config_strings, tol=1e-5, max_iter=40, ns=113, **kwargs):
        """Run MCDHF calculation.

        Parameters
        ----------
        Z : int
            Atomic number (nuclear charge).
        config_strings : list of str
            Configuration strings, e.g. ``["1s2 2s2", "1s2 2p2"]``.
        tol : float
            SCF convergence tolerance (Hartree).
        max_iter : int
            Maximum number of MCDHF macro-iterations.
        """
        csfs = make_csf_list(Z, config_strings)
        solver = MCDHFSolver(
            Z=Z, csfs=csfs, ns=ns, ks=self.bspline_order,
            include_breit=self.breit, include_qed=self.qed,
        )
        return solver.solve(max_iter=max_iter, scf_tol=tol)


__all__ = [
    "CSF",
    "MCDHF",
    "MCDHFResult",
    "MCDHFSolver",
    "build_ci_matrix",
    "count_j_resolved_csfs",
    "make_csf_list",
]
