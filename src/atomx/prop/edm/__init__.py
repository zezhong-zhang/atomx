"""T,P-violating electric dipole moment (EDM) enhancement factors.  SCAFFOLD.

Planned scope
-------------
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
    edm_summary,
    electron_edm_enhancement_sandars,
    schiff_moment_enhancement_scaling,
)


@dataclass
class EDMResult:
    """T,P-odd enhancement factors (scaffold)."""

    electron_edm_enhancement: float = 0.0
    schiff_moment_enhancement: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def schiff_moment_enhancement_cm(self) -> float:
        """Schiff-moment enhancement in cm."""
        return self.schiff_moment_enhancement


def compute(
    *,
    Z: int | None = None,
    A: int | None = None,
    n_eff: float = 2.0,
    J: float = 0.5,
    L: int = 0,
    valence: str = "s",
) -> EDMResult:
    """Evaluate closed-form EDM enhancement scaling factors."""
    if Z is None or A is None:
        raise NotImplementedError(
            "edm ab-initio enhancement factors are a planned observable; "
            "pass Z and A to use closed-form scaling estimates."
        )
    summary = edm_summary(
        Z=Z,
        A=A,
        n_eff=n_eff,
        J=J,
        L=L,
        valence=valence,
    )
    return EDMResult(
        electron_edm_enhancement=summary["R_electron_edm"],
        schiff_moment_enhancement=summary["K_schiff_cm"],
        provenance=summary,
    )


__all__ = [
    "EDMResult",
    "compute",
    "edm_summary",
    "electron_edm_enhancement_sandars",
    "schiff_moment_enhancement_scaling",
]
