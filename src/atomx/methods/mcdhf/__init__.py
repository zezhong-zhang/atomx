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
        Include the Breit interaction **as an operator in the CI
        Hamiltonian**: the exact Gaunt term
        (:mod:`atomx.methods.mcdhf.breit_ci`) is built in the m-scheme
        determinant basis of every ``(2J, parity)`` block, so it shifts
        levels *differentially* and produces fine structure.  Measured
        on He-like C V ``["1s2", "1s1 2p1"]``: the 1s2p 3P fine
        structure moves from the Dirac-Coulomb 13.74 cm-1 to 123.29
        cm-1 (NIST ASD v5.12 observed: 123.29 cm-1).
    qed : bool
        Include QED corrections (self-energy + vacuum polarisation).
        QED is a monopole correction and *is* legitimately a per-orbital
        scalar: each orbital ``i`` carries its own shift ``eps_i`` and
        state ``k`` receives ``sum_i q_i^(k) eps_i`` with the
        CI-weighted orbital occupations ``q_i^(k)`` — so states with
        different occupancies get different QED shifts.  The Breit
        interaction is *not* treated that way: its spin-other-orbit and
        spin-spin parts are not an orbital sum, which is why it needs
        the CI operator above.
    breit_mode : str
        ``"operator"`` (default) — the Gaunt operator in the CI blocks.
        ``"scalar"`` — the historical comparison path: the
        configuration-average number from
        the exact Gaunt operator, evaluated
        once on the converged Guerra-averaged orbitals and added to
        every CI state identically.  It cannot split levels and exists
        only so the old numbers stay reproducible.
    bspline_order : int
        B-spline polynomial order (default 9).

    Notes
    -----
    The frequency-independent **retardation** term (the second half of
    the Breit interaction, ``+1/2[alpha.alpha/r - (alpha.r)(alpha.r)/r^3]``)
    is *not* implemented.  It is not silently zeroed: the operator's
    provenance records it and
    :data:`atomx.methods.mcdhf.breit_ci.RETARDATION_NOTE` carries its
    measured size (~1-4 % of the Gaunt total on closed-shell Ne, where
    DBSR_HF's full Breit is the reference).
    """

    correlation: Correlation = Correlation.SINGLES_DOUBLES
    breit: bool = False
    qed: bool = False
    breit_mode: str = "operator"
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
            breit_mode=self.breit_mode,
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
