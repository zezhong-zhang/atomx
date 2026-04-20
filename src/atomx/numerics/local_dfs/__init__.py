"""Local Dirac-Fock-Slater SCF support.

Shared glue for :class:`atomx.methods.dirac_fock.DFSolver` and
:class:`atomx.numerics.shooting.solver.ShootingSolver` when the caller
picks a local-exchange flavour (``exchange='slater'`` or
``exchange='fac'``). The core functional lives in
:mod:`atomx.physics.local_exchange`; this sub-package holds the
supporting numerics — Gu's tabulated parameters and (in a follow-up
task) the FAC-style asymptotic-potential fix.
"""

from atomx.numerics.local_dfs.hxs_tables import (  # noqa: F401
    NHXN,
    SLATER_ALPHA_LDA,
    hxs_row,
)
