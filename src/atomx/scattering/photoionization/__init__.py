"""Photoionization cross sections, with resonance decoration.

:mod:`atomx.scattering.photoionization.direct` ships the smooth non-resonant Dirac
single-channel cross section ``σ_bg(ω)``.  This companion module
layers autoionization resonances on top of that background using the
Fano (Beutler-Fano) profile:

    σ(ε) = σ_bg(ε) · (q + ε)² / (1 + ε²),   ε = 2(E − E_res)/Γ        (1)

A ``ρ²`` (correlation) factor and multiple-channel formulation are
included per Fano's original 1961 paper.

Public API
----------
:class:`FanoResonance`
    ``(E_res_eV, Γ_total_eV, q, rho_sq)`` parameters for one resonance.

:func:`fano_profile`
    Dimensionless Fano factor  (q + ε)² / (1 + ε²)  as a function of
    photon energy.

:func:`apply_resonances`
    Given a background cross-section array and a list of
    :class:`FanoResonance`, returns the resonance-decorated XS on
    the same energy grid.

References
----------
U. Fano, *Phys. Rev.* **124**, 1866 (1961).
M. F. Gu, *Astrophys. J.* **582**, 1241 (2003) — FAC PI calculations.
"""

from atomx.scattering.photoionization.direct import photoionization_cross_section
from atomx.scattering.photoionization.fano import (
    FanoResonance,
    apply_resonances,
    fano_profile,
)

__all__ = [
    "photoionization_cross_section",
    "FanoResonance",
    "apply_resonances",
    "fano_profile",
]
