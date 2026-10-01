"""Hyperfine structure: magnetic dipole (A) and electric quadrupole (B).

For a single Dirac orbital with quantum number κ, j the magnetic-dipole
hyperfine constant is

    A_nj = α g_I (m_e/m_p) · [κ / j(j+1)] · ⟨P Q / r²⟩     [Hartree]

(with no factor 2: for I = j = 1/2 this A *is* the F=1 ↔ F=0 splitting,
ΔE_HFS = A.  A factor 2 belongs only to the H = 2A I·J convention;
atomx uses H = A I·J — see
:mod:`atomx.properties.hyperfine.dipole`, review 2026-09-19 finding M9.)

with ``⟨P Q / r²⟩ = ∫ P(r) Q(r) / r² dr`` the hyperfine radial integral
over the large (P) and small (Q) Dirac components, and ``g_I = μ_I / I``
the nuclear g-factor in units of the nuclear magneton μ_N.

Public API
----------
>>> from atomx.properties.hyperfine import hyperfine_A, hyperfine_A_MHz
>>> hyperfine_A_MHz(P, Q, bsp_data, kappa=-1, Z=1)   # H 1s ≈ 1420 MHz

See Also
--------
atomx.properties.hyperfine.nuclear_moments : tabulated μ_I, I, A for common isotopes.
"""

from atomx.properties.hyperfine.dipole import (
    ELECTRON_ANOMALOUS_MOMENT,
    HARTREE_MHZ,
    hyperfine_A,
    hyperfine_A_MHz,
    hyperfine_radial_integral,
)
from atomx.properties.hyperfine.nuclear_moments import (
    NUCLEAR_MOMENTS,
    NuclearMoment,
    default_isotope,
    g_factor,
    nuclear_moment,
)
from atomx.properties.hyperfine.quadrupole import (
    BARN_BOHR2,
    hyperfine_B,
    hyperfine_B_MHz,
    radial_r_inv_cubed,
)

__all__ = [
    # dipole
    "ELECTRON_ANOMALOUS_MOMENT",
    "HARTREE_MHZ",
    "hyperfine_A",
    "hyperfine_A_MHz",
    "hyperfine_radial_integral",
    # quadrupole
    "BARN_BOHR2",
    "hyperfine_B",
    "hyperfine_B_MHz",
    "radial_r_inv_cubed",
    # nuclear data
    "NUCLEAR_MOMENTS",
    "NuclearMoment",
    "default_isotope",
    "g_factor",
    "nuclear_moment",
]
