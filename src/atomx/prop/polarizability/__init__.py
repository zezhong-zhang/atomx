"""Static and dynamic dipole polarisability  α(ω).  SCAFFOLD.

Planned scope
-------------
- Sum-over-states form at low cost:
  α(ω) = 2 Σ_n (E_n − E_0) |⟨0|d|n⟩|² / [(E_n − E_0)² − ω²].
- RPA / TDHF response — numerically accurate; calls
  :func:`atomx.methods.rpa.compute` once implemented.
- Sternheimer self-consistent linear-response form for atoms where
  RPA overcounts core–valence interactions.
- Black-body radiation shift from a Planck-weighted integral of
  α(ω).

References
----------
- J. Mitroy, M. S. Safronova, C. W. Clark,
  *J. Phys. B* **43**, 202001 (2010) — review of polarisability
  calculations in atoms.
- A. Derevianko, S. G. Porsev, K. Beloy, *Phys. Rev. A* **82**,
  030901(R) (2010) — BBR-shift computation for clock atoms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class PolarizabilityResult:
    """Static / dynamic dipole polarisability (scaffold)."""

    static: Optional[float] = None
    dynamic_omega: Optional[np.ndarray] = None
    dynamic_alpha: Optional[np.ndarray] = None
    provenance: Dict[str, Any] = field(default_factory=dict)


def compute(*_args, **_kwargs) -> PolarizabilityResult:
    """Evaluate α(ω) for a given result.  Not implemented yet.

    Tracking issue: *atomx#polarizability*.
    """
    raise NotImplementedError(
        "Polarisability evaluation is a planned observable — see "
        "prop/polarizability/ scaffold (tracking issue atomx#polarizability)."
    )


__all__ = ["PolarizabilityResult", "compute"]
