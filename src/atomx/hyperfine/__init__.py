"""Hyperfine structure: magnetic dipole (A) and electric quadrupole (B).

For a single Dirac orbital with quantum number κ, j the magnetic-dipole
hyperfine constant is

    A_nj = 2 α g_I (m_e/m_p) · [κ / j(j+1)] · ⟨P Q / r²⟩     [Hartree]

with ``⟨P Q / r²⟩ = ∫ P(r) Q(r) / r² dr`` the hyperfine radial integral
over the large (P) and small (Q) Dirac components, and ``g_I = μ_I / I``
the nuclear g-factor in units of the nuclear magneton μ_N.

Public API
----------
>>> from atomx.hyperfine import hyperfine_A, hyperfine_A_MHz
>>> hyperfine_A_MHz(P, Q, bsp_data, kappa=-1, Z=1)   # H 1s ≈ 1420 MHz

See Also
--------
atomx.hyperfine.nuclear_moments : tabulated μ_I, I, A for common isotopes.
"""

from atomx.hyperfine.dipole import (
    ELECTRON_ANOMALOUS_MOMENT,
    HARTREE_MHZ,
    hyperfine_A,
    hyperfine_A_MHz,
    hyperfine_radial_integral,
)
from atomx.hyperfine.nuclear_moments import (
    NUCLEAR_MOMENTS,
    NuclearMoment,
    default_isotope,
    g_factor,
    nuclear_moment,
)

__all__ = [
    "ELECTRON_ANOMALOUS_MOMENT",
    "HARTREE_MHZ",
    "hyperfine_A",
    "hyperfine_A_MHz",
    "hyperfine_radial_integral",
    "NUCLEAR_MOMENTS",
    "NuclearMoment",
    "default_isotope",
    "g_factor",
    "nuclear_moment",
]
