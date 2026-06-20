"""T,P-violating electric dipole moment (EDM) enhancement factors.

Atomic EDMs probe fundamental CP violation.  The relevant atomic
quantities are enhancement factors that multiply underlying
T,P-odd sources:

- **Electron EDM** (``d_e``) — enhancement factor ``R_e`` in
  paramagnetic atoms/molecules (e.g. Tl, HfF⁺, ThO).
- **Nuclear Schiff moment** (``S_N``) — enhancement factor
  ``K_S`` in diamagnetic atoms (e.g. ¹²⁹Xe, ¹⁹⁹Hg, ²²⁵Ra).
- **Tensor-pseudotensor** and **scalar-pseudoscalar** electron–nucleon
  interactions.

These are computed as sum-over-states / Sternheimer matrix elements
of the T,P-odd operators on a relativistic SCF reference with
correlation included through Coupled-Cluster or all-order MBPT.

Current modules
---------------
:mod:`atomx.prop.edm.scaling`
    Closed-form Sandars and Dzuba-Flambaum scaling formulas.
:mod:`atomx.prop.edm.sternheimer`
    Sternheimer sum-over-states calculation of the electron-EDM
    enhancement factor R_e using the existing V^{N-1} virtual basis.

References
----------
- E. D. Commins, *Adv. At. Mol. Opt. Phys.* **40**, 1 (1999).
- P. G. H. Sandars, *Phys. Lett.* **14**, 194 (1965) — the original
  enhancement-factor concept.
- I. B. Khriplovich & S. K. Lamoreaux, *CP Violation Without
  Strangeness* (1997).
- V. A. Dzuba, V. V. Flambaum, *Phys. Rev. A* **80**, 062509 (2009)
  — review of atomic EDM calculations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from atomx.prop.edm.scaling import (
    electron_edm_enhancement_sandars,
    schiff_moment_enhancement_scaling,
    edm_summary,
)
from atomx.prop.edm.sternheimer import (
    SternheimerEDMResult,
    sternheimer_edm_enhancement,
    compute,
)


@dataclass
class EDMResult:
    """T,P-odd enhancement factors (scaffold)."""

    electron_edm_enhancement: float = 0.0
    schiff_moment_enhancement_cm: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def schiff_moment_enhancement(self) -> float:
        """Backward-compatible alias for the Schiff enhancement in cm."""
        return self.schiff_moment_enhancement_cm


__all__ = [
    "EDMResult",
    "SternheimerEDMResult",
    "sternheimer_edm_enhancement",
    "compute",
    "edm_summary",
    "electron_edm_enhancement_sandars",
    "schiff_moment_enhancement_scaling",
]
