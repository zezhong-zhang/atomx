"""AtomX: High-performance relativistic atomic structure calculations.

B-spline Dirac-Fock solver with Numba-accelerated kernels and SciPy LAPACK.

Quick start:
    from atomx import Atom
    ne = Atom(Z=10)
    ne.solve()
    ne.summary()
"""

__version__ = "0.1.0"

from atomx.atom import Atom
