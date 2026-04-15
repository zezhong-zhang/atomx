"""Autoionization, Auger rates, and dielectronic recombination (DR).

This module implements the three companion processes that govern the
population and decay of doubly-excited (autoionizing) atomic states:

1. **Auger / autoionization** — radiationless decay of a doubly-
   excited state via ejection of one electron to the continuum.  The
   rate is a two-electron Coulomb matrix element with one bound and
   one continuum orbital.

2. **Dielectronic recombination** (DR) — the inverse of Auger, where
   a free electron is captured into a doubly-excited state that
   subsequently decays radiatively to a singly-bound configuration.
   The DR rate coefficient combines Auger and radiative widths via
   Burgess' (1965) detailed-balance formula.

3. **Resonant photoionization** — photoabsorption with a resonance
   structure from autoionizing intermediate states (handled in
   :mod:`atomx.scattering.photoionization`).

Public API
----------
:class:`AutoionizationState`
    Dataclass holding ``(E_resonance, J, Γ_Auger, A_radiative, ...)``
    for a single resonance.

:func:`auger_rate_two_electron`
    Compute Γ_Auger from a pair of bound orbitals + one continuum
    orbital via the two-electron Coulomb matrix element (Slater
    integrals) and the appropriate angular coupling.

:func:`dr_rate_coefficient`
    Detailed-balance DR rate coefficient α^DR(T_e) from a list of
    :class:`AutoionizationState` resonances.

Reference
---------
M. F. Gu, *Astrophys. J. Suppl.* **153**, 389 (2004) — FAC's approach.
A. Burgess, *Astrophys. J.* **139**, 776 (1965) — first DR theory.
D. W. Savin, *Astrophys. J.* **562**, 1259 (2001) — benchmark Fe XVI DR.
"""

from atomx.scattering.autoionization.auger import (
    AutoionizationState,
    auger_rate_two_electron,
)
from atomx.scattering.dielectronic_recombination.solver import (
    DRRateCoefficient,
    dr_rate_coefficient,
)

__all__ = [
    "AutoionizationState",
    "DRRateCoefficient",
    "auger_rate_two_electron",
    "dr_rate_coefficient",
]
