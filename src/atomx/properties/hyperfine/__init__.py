"""Hyperfine structure: magnetic dipole (A) and electric quadrupole (B).

For a single Dirac orbital with quantum number κ, j the magnetic-dipole
hyperfine constant is

    A_nj = 2 α g_I (m_e/m_p) · [κ / j(j+1)] · ⟨P Q / r²⟩     [Hartree]

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
from atomx.properties.hyperfine.qed_corrections import (
    DEUTERON_POLARIZABILITY_D1S,
    DEUTERON_ZEMACH_RADIUS_FM,
    HELIUM3_POLARIZABILITY,
    HELIUM3_ZEMACH_RADIUS_FM,
    PROTON_POLARIZABILITY_H1S,
    PROTON_ZEMACH_RADIUS_FM,
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
    # QED corrections to hyperfine operator — precision constants
    "DEUTERON_POLARIZABILITY_D1S",
    "DEUTERON_ZEMACH_RADIUS_FM",
    "HELIUM3_POLARIZABILITY",
    "HELIUM3_ZEMACH_RADIUS_FM",
    "PROTON_POLARIZABILITY_H1S",
    "PROTON_ZEMACH_RADIUS_FM",
    # nuclear data
    "NUCLEAR_MOMENTS",
    "NuclearMoment",
    "default_isotope",
    "g_factor",
    "nuclear_moment",
]
