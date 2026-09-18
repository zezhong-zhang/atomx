"""What equation is being solved.

This sub-package holds the *physics* axis of an atomic calculation —
the radial Hamiltonian / Dirac operator structure, the rest-mass
shift convention, perturbative operator definitions — independently
of the chosen numerical representation (:mod:`atomx.numerics`) and
the variational level (:mod:`atomx.methods`).

The canonical :class:`Equation` protocol is defined in
:mod:`atomx.physics.base`; concrete equations live in the sibling
modules.

Current equations
-----------------
:class:`~atomx.physics.schrodinger.Schrodinger`
    One-component non-relativistic radial equation.
:class:`~atomx.physics.dirac.Dirac`
    Two-component (P, Q) relativistic radial equation with κ-coupling.
:mod:`atomx.physics.breit_pauli`
    Perturbative operators (mass-velocity, Darwin, spin-orbit) for the
    ``BreitPauli`` method — the non-relativistic-limit reduction of
    the Dirac equation.

Two-electron interaction kernels live where they are evaluated: the
Coulomb/Slater radial machinery in :mod:`atomx.numerics.bspline` and
:func:`atomx.numerics.bspline.slater_yk.compute_slater_rk`, and the Breit
angular coefficients in :mod:`atomx.corrections.breit`.
"""

from atomx.physics.base import Equation  # noqa: F401
from atomx.physics.dirac import (  # noqa: F401
    Dirac,
    hydrogenic_dirac_energy,
    list_hydrogenic_states,
)
from atomx.physics.schrodinger import Schrodinger  # noqa: F401
