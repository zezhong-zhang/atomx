"""Zeeman effect: Landé g-factors and weak-field energy shifts.

For an atomic state with good ``|J, m_J⟩`` quantum numbers in a weak
external magnetic field **B**,

    ΔE_Zeeman = g_J · μ_B · B · m_J.                                  (1)

The Landé factor ``g_J`` is computed in three increasingly accurate
layers:

1. **Non-relativistic closed form** (LS-coupling):

       g_J = 1 + [j(j+1) − l(l+1) + s(s+1)] / (2 j (j+1)),
                 with g_s = 2, g_l = 1.

2. **Bound-Dirac (Breit 1928)** for a single electron in a Coulomb
   field:

       g_J = (2/3) · [1 + 2 √(1 − (Zα)²)],    (for j = 1/2, κ = −1)

   giving a ~(Zα)²/3 relativistic reduction.

3. **Electron anomalous moment** (Schwinger + loops):
   multiply the result by ``(1 + a_e)`` with ``a_e = 1.15965218e-3``.
   This is the leading QED correction that brings the bound-electron g
   into agreement with high-precision Penning-trap measurements
   (Sturm, Wagner, *et al.*).

Public API
----------
>>> from atomx import Atom
>>> result = Atom(Z=1).solve(...)
>>> g = result.zeeman_gJ("1s1/2")
>>> shifts = result.zeeman_shift(B_Tesla=1.0)

Strong-field (Paschen–Back) regime, quadratic (diamagnetic) Zeeman, and
finite-nuclear-mass corrections to the bound-electron g are TODOs for a
follow-up.
"""

from atomx.properties.zeeman.lande import (
    BOHR_MAGNETON_MHZ_PER_TESLA,
    LandeFactor,
    lande_gJ_bound_single_electron,
    lande_gJ_hydrogenic,
    lande_gJ_nr,
)
from atomx.properties.zeeman.shift import (
    ZeemanShift,
    weak_field_shift_MHz,
    zeeman_shift_table,
)

__all__ = [
    "BOHR_MAGNETON_MHZ_PER_TESLA",
    "LandeFactor",
    "ZeemanShift",
    "lande_gJ_bound_single_electron",
    "lande_gJ_hydrogenic",
    "lande_gJ_nr",
    "weak_field_shift_MHz",
    "zeeman_shift_table",
]
