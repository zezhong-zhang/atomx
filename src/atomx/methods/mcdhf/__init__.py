"""Multi-Configuration Dirac-Hartree-Fock (MCDHF).

The B-spline route starts from Guerra-averaged DHF orbitals, then minimizes
the ground-state correlated CI energy in its shared active space. Excited
states are diagonalized on that same orbital set; their principal labels
are not guaranteed to remain spectroscopic after ground-state optimization.

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

from atomx.config.csf import (
    CSF,
    count_j_resolved_csfs,
    make_csf_list,
)
from atomx.methods.base import Method
from atomx.methods.mcdhf.solver import MCDHFResult, MCDHFSolver
from atomx.numerics.shooting.electron_space import ShootingSpace
from atomx.physics.csf.matrix import build_ci_matrix
from atomx.types import Correlation, TheoryLevel


@dataclass
class MCDHF(Method):
    """Multi-Configuration Dirac–Hartree–Fock.

    Generates CSFs from a reference configuration via single and/or
    double excitations into virtual orbitals, then iteratively:

    1. Builds a Guerra average-level DHF orbital seed, then optimises
       shared orbitals for the correlated ground-state CI energy.
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
        (:mod:`atomx.physics.csf.breit_ci`) is built in the m-scheme
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
    orbital_preconditioner : {"spectral", "shooting"}
        Experimental shooting uses radial Green solves to precondition the
        full correlated gradient, retaining B-spline energy and constraint
        checks. It does not select the configuration-average shooting engine.

    Notes
    -----
    Static Breit is the default. ``breit_frequency="orbital_differences"``
    with an explicit ``breit_orbital_energies`` map selects real transverse
    exchange in RCI after Dirac-Coulomb orbital optimization. The energy
    map is fixed; finite-frequency orbital relaxation is not included.
    """

    correlation: Correlation = Correlation.SINGLES_DOUBLES
    breit: bool = False
    qed: bool = False
    breit_mode: str = "operator"
    breit_frequency: str = "static"
    breit_orbital_energies: dict | None = None
    bspline_order: int = 9
    orbital_preconditioner: str = "spectral"
    engine: str = "bspline"
    shooting_space: ShootingSpace | None = None

    @property
    def theory_level(self) -> TheoryLevel:
        return TheoryLevel.MCDHF

    def solve(self, Z, config_strings, tol=1e-5, max_iter=40, ns=113, *,
              r_max=None, use_physics_grid=True, grid_hi=None, grid_he=None,
              grid_hmax=None, screening_potential=None):
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
        if not use_physics_grid and any(
                value is not None for value in (grid_hi, grid_he, grid_hmax)):
            raise ValueError('grid shape controls require use_physics_grid=True')
        csfs = make_csf_list(Z, config_strings)
        stack = None
        if self.breit_frequency != 'static' or self.breit_orbital_energies is not None:
            from atomx.corrections.qed import resolve_se_method
            from atomx.corrections.stack import CorrectionStack
            stack = CorrectionStack(
                include_breit=self.breit, breit_frequency=self.breit_frequency,
                breit_orbital_energies=self.breit_orbital_energies,
                se_backend=resolve_se_method('auto', Z) if self.qed else None,
                include_vp=self.qed)
        solver = MCDHFSolver(
            Z=Z, csfs=csfs, ns=ns, ks=self.bspline_order,
            include_breit=self.breit, include_qed=self.qed,
            breit_mode=self.breit_mode,
            correction_stack=stack,
            orbital_preconditioner=self.orbital_preconditioner,
            engine=self.engine, shooting_space=self.shooting_space,
            nuclear=self.nuclear.value,
            r_max=r_max, use_physics_grid=use_physics_grid,
            grid_hi=grid_hi, grid_he=grid_he, grid_hmax=grid_hmax,
            screening_potential=screening_potential,
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
