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


@dataclass
class EDMResult:
    """T,P-odd enhancement factors (scaffold)."""

    electron_edm_enhancement: float = 0.0
    schiff_moment_enhancement: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)


def compute(*_args, **_kwargs) -> EDMResult:
    """Evaluate T,P-odd enhancement factors.  Not implemented yet.

    Tracking issue: *atomx#edm*.
    """
    raise NotImplementedError(
        "EDM enhancement factors are a planned observable — see "
        "prop/edm/ scaffold (tracking issue atomx#edm)."
    )


__all__ = ["EDMResult", "compute"]
