"""Per-isotope nuclear-structure constants (Zemach radii, polarisabilities).

Citation
--------
- CODATA 2022 recommended values (proton Zemach radius reconciling
  Mainz 2011 elastic-scattering and muonic-H spectroscopy).
- J. Friar & I. Sick, *Phys. Rev. C* **88**, 034003 (2013) — ³He
  Zemach moment.
- Friar, Payne, Preedom (2005) — deuteron Zemach radius.
- C. E. Carlson, V. Nazaryan, K. Griffioen, *Phys. Rev. Lett.* **107**,
  020401 (2011) — proton polarisability contribution to H 1s
  hyperfine splitting.
- Khriplovich et al. — deuteron polarisability.

Units
-----
Zemach radii are in femtometres (fm).  Polarisability corrections are
dimensionless fractional shifts (δ_pol) of the hyperfine splitting.
"""

from __future__ import annotations

from dataclasses import dataclass


CITATION = (
    "CODATA 2022; Friar & Sick, PRC 88, 034003 (2013); "
    "Carlson, Nazaryan, Griffioen, PRL 107, 020401 (2011)."
)


@dataclass(frozen=True)
class NuclearStructureConstants:
    """Nuclear-structure inputs for the hyperfine correction ladder.

    Attributes
    ----------
    zemach_radius_fm
        Zemach moment  r_Z  in femtometres.
    polarisability_fraction
        Fractional correction δ_pol to the hyperfine splitting due to
        nuclear polarisability.
    """

    zemach_radius_fm: float
    polarisability_fraction: float


TABLE: dict[str, NuclearStructureConstants] = {
    # Proton / ¹H
    "h1s":    NuclearStructureConstants(1.054, 1.3e-6),
    # Deuteron / ²H
    "d1s":    NuclearStructureConstants(2.593, 1.7e-6),
    # ³He
    "he3_1s": NuclearStructureConstants(2.528, 3.1e-6),
}
