"""Parity non-conservation (PNC) amplitudes.  SCAFFOLD.

Planned scope
-------------
Matrix element of the weak-interaction Hamiltonian

    H_W  =  (G_F / √8) · Q_W · γ₅ · ρ_nuc(r)

between opposite-parity atomic states, producing the dominant PNC
observable in heavy alkali atoms (Cs, Yb, Fr) — the ``E_PNC`` matrix
element that multiplies the weak charge ``Q_W``.

Deliverables
------------
- Nuclear-spin-independent (NSI) amplitude ``E_PNC_NSI``.
- Nuclear-spin-dependent (NSD) amplitude from the nuclear anapole
  moment.
- Basis correlation corrections via Sternheimer / RPA / all-order
  coupled-cluster machinery.

References
----------
- M.-A. Bouchiat & C. Bouchiat, *Rep. Prog. Phys.* **60**, 1351
  (1997).
- V. A. Dzuba, V. V. Flambaum, M. G. Kozlov, *Phys. Rev. Lett.*
  **80**, 3390 (1998) — Cs ⁶S→⁷S amplitude at the per-cent level.
- C. S. Wood *et al.*, *Science* **275**, 1759 (1997) — Boulder Cs
  measurement.
- B. M. Roberts, V. A. Dzuba, V. V. Flambaum, *Annu. Rev. Nucl.
  Part. Sci.* **65**, 63 (2015) — review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class PNCResult:
    """PNC matrix element result (scaffold)."""

    E_PNC_NSI: float = 0.0
    E_PNC_NSD: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)


def compute(*_args, **_kwargs) -> PNCResult:
    """Evaluate the PNC matrix element.  Not implemented yet.

    Tracking issue: *atomx#pnc*.
    """
    raise NotImplementedError(
        "Parity-non-conservation amplitudes are a planned observable "
        "— see prop/pnc/ scaffold (tracking issue atomx#pnc)."
    )


__all__ = ["PNCResult", "compute"]
