"""Bound-electron Landé g_J factors.  SCAFFOLD.

Planned scope
-------------
- Non-relativistic g_J (already available via
  :func:`atomx.prop.zeeman.lande.lande_gJ_nr`).
- Dirac bound-electron contribution  g_D(Z, n, κ).
- QED corrections:
  - Schwinger (single-loop) anomalous moment ``a_e``.
  - Two-loop QED  (α/π)² Fit to Sturm et al., *Phys. Rev. Lett.*
    **107**, 023002 (2011).
  - Vacuum polarisation and self-energy on the bound-electron vertex.
- Nuclear-size and nuclear-recoil (electron-to-nucleus mass ratio)
  corrections, needed for Penning-trap g-factor measurements that
  constrain α and the electron mass.

References
----------
- V. M. Shabaev, *Phys. Rep.* **356**, 119 (2002).
- S. Sturm *et al.*, *Nature* **506**, 467 (2014) — precision g-factor
  measurement on ¹²C⁵⁺.
- A. Czarnecki, *Phys. Rev. A* **61**, 052508 (2000) — two-loop
  self-energy correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class GFactorResult:
    """Bound-electron g_J result (scaffold)."""

    g_J: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)


def compute(*_args, **_kwargs) -> GFactorResult:
    """Bound-electron Landé g_J with QED corrections.  Not implemented yet.

    Tracking issue: *atomx#g_factor*.
    """
    raise NotImplementedError(
        "Bound-electron g-factor is a planned observable — see "
        "prop/g_factor/ scaffold (tracking issue atomx#g_factor)."
    )


__all__ = ["GFactorResult", "compute"]
