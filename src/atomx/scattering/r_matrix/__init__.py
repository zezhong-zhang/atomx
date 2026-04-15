"""R-matrix close-coupling scattering.  SCAFFOLD.

Planned scope
-------------
Systematic continuum treatment via the internal-external region
split of the R-matrix method:

- Inside ``r ≤ a``: the full configuration-interaction problem in a
  square-integrable basis (B-splines natural fit).
- Outside ``r > a``: the continuum is described by asymptotic
  coupled partial waves matched at the boundary through the R-matrix.

Gives the unified framework for:
- Electron-impact excitation & ionisation cross-sections.
- Photoionisation with resonances.
- Dielectronic recombination with inter-resonance interference.
- Low-energy scattering phase shifts.

References
----------
- P. G. Burke, *R-Matrix Theory of Atomic Collisions* (Springer,
  2011) — the canonical reference.
- K. A. Berrington, W. B. Eissner, P. H. Norrington, *Comput. Phys.
  Commun.* **92**, 290 (1995) — relativistic Dirac R-matrix code
  (DARC).
- N. R. Badnell *et al.*, *J. Phys. B* **29**, 3683 (1996) —
  relativistic close-coupling for heavy atoms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class RMatrixResult:
    """Output of an R-matrix calculation (scaffold)."""

    cross_sections_Mbarn: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)


def compute(*_args, **_kwargs) -> RMatrixResult:
    """Close-coupling R-matrix solve.  Not implemented yet.

    Tracking issue: *atomx#r_matrix*.
    """
    raise NotImplementedError(
        "R-matrix close-coupling is a planned module — see "
        "scattering/r_matrix/ scaffold (tracking issue atomx#r_matrix)."
    )


__all__ = ["RMatrixResult", "compute"]
