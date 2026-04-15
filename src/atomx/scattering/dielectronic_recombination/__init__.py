"""Dielectronic recombination (DR).

Two-step resonant capture:

1. Free electron is captured, forming a doubly-excited intermediate
   autoionising state (resonance).
2. The state stabilises radiatively before autoionising back.

The DR rate coefficient is a sum over intermediate resonances weighted
by radiative-to-total branching ratios computed by
:func:`atomx.scattering.autoionization.auger.auger_rates`.
"""

from atomx.scattering.dielectronic_recombination.solver import *  # noqa: F401,F403
