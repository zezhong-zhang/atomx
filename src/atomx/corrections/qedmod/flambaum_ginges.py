"""Flambaum–Ginges local radiative potential (stub).

The Flambaum–Ginges radiative potential is a local operator on the radial
coordinate that reproduces the one-electron QED self-energy to good
accuracy.  It splits into low- and high-frequency pieces plus a magnetic
form-factor term:

    V_rad(r) = V_SE^low(r) + V_SE^high(r) + V_magnetic(r)  [ + V_WK(r) ]

Key references
--------------
V. V. Flambaum, J. S. M. Ginges,
    "Radiative potential and calculation of QED radiative corrections to
    energy levels and electromagnetic amplitudes in many-electron atoms",
    Phys. Rev. A 72, 052115 (2005).
J. S. M. Ginges, J. C. Berengut,
    "Atomic many-body effects and Lamb shifts in alkali metals",
    Phys. Rev. A 93, 052509 (2016).
J. S. M. Ginges, J. C. Berengut,
    "QED radiative corrections and many-body effects in atoms",
    J. Phys. B 49, 095001 (2016).

Accuracy: 0.1–1 % of the exact hydrogenic SE for Z ≤ 92, including
orbitals with n ≥ 3 that the Mohr table does not cover.  Widely used in
AMBiT, dmd, and modern GRASP extensions.

Implementation notes
--------------------
The local potential is kappa-dependent through the magnetic piece.  Once
implemented, the backend will:
    1. Build V_rad(r) on the B-spline grid (per-orbital kappa).
    2. Sandwich: ``E_a = ∫ [P² + Q²] V_rad(r) dr``.
    3. Optionally provide the potential itself for self-consistent QED,
       so the Fock matrix can pick it up during SCF iterations.
"""

from __future__ import annotations

from atomx.corrections.qedmod.base import SEBackend, SEResult, register_backend


@register_backend
class FlambaumGingesBackend(SEBackend):
    """Flambaum–Ginges (2005) radiative potential — **stub**.

    Reserves the ``"flambaum_ginges"`` name in the backend registry and
    documents the target paper.  The actual potential formulas will be
    filled in in a follow-up iteration.
    """

    name = "flambaum_ginges"
    description = (
        "Flambaum–Ginges (PRA 72, 052115, 2005) local radiative "
        "potential — not yet implemented."
    )

    def compute(self, orbitals, bsp_data, Z) -> SEResult:
        raise NotImplementedError(
            "Flambaum–Ginges radiative potential is not yet implemented. "
            "Use backend 'mohr' (default) or 'welton' for now. "
            "Target reference: Flambaum & Ginges, PRA 72, 052115 (2005); "
            "Ginges & Berengut, J. Phys. B 49, 095001 (2016)."
        )
