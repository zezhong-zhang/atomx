"""Numerical representations for radial atomic problems.

The back-ends here are interchangeable choices for *how* the
radial Schrödinger or Dirac problem is discretised; they are
independent of *what* equation is being solved (which lives in
:mod:`atomx.physics`) and of *which* variational level is applied
(which lives in :mod:`atomx.methods`).

Sub-packages
------------
:mod:`atomx.numerics.bspline`
    B-spline basis (Zatsarinny / Froese Fischer DBSR_HF style).

:mod:`atomx.numerics.shooting`
    ODE shooting on a non-uniform grid (Salvat RADIAL / DHFS style).

:mod:`atomx.numerics.yk`
    Shared Y^k potential mathematics — the single home for the
    interpolatory partial-interval construction used by the dense
    Gauss-node (B-spline Galerkin) path.
"""
