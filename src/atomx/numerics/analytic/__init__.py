"""Closed-form (analytic) radial wavefunctions.

Hydrogenic Dirac and Schrödinger orbitals that admit a closed-form
expression on a numerical grid live here.  They are "numerics" in
the axis sense — a concrete way to evaluate a wavefunction — but
they are exact rather than variational, so they have no SCF loop.

Modules
-------
:mod:`atomx.numerics.analytic.hydrogenic_dirac`
    Dirac point-nucleus (P, Q) components and the
    ``⟨r · exp(−cr)⟩`` matrix element used by the Shabaev-local
    self-energy backend.
"""

from atomx.numerics.analytic.hydrogenic_dirac import (  # noqa: F401
    hydrogenic_dirac_wavefunction,
    hydrogenic_r_exp_integral_dirac,
)
