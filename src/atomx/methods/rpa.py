"""Random-phase approximation / TDHF linear response.  SCAFFOLD.

Planned scope
-------------
Frequency-dependent linear response of a DHF reference via the
RPA (also: TDHF) eigenvalue problem

    ⎛ A   B ⎞⎛ X ⎞ =  ω ⎛ 1    0⎞ ⎛X⎞
    ⎝ B*  A*⎠⎝ Y ⎠      ⎝ 0  −1 ⎠ ⎝Y⎠

with the A / B blocks built from particle–hole Slater matrix
elements.  Dominant deliverables:

- Static dipole polarisability α(0) from the (ω = 0) response.
- Dynamic polarisability α(ω) over a photon-energy sweep —
  directly feeds :mod:`atomx.prop.polarizability`.
- Black-body radiation shift (integral of α(ω) against Planck's law).

References
----------
- A. L. McLachlan & M. A. Ball, *Rev. Mod. Phys.* **36**, 844 (1964).
- P. Ring & P. Schuck, *The Nuclear Many-Body Problem* (1980),
  Chapter 8 — the canonical exposition of the RPA matrix form.
- M. S. Safronova, W. R. Johnson, A. Derevianko,
  *Phys. Rev. A* **60**, 4476 (1999) — polarisabilities for
  alkali atoms via RPA + MBPT.

Required inputs (present in atomx)
----------------------------------
- DHF reference + virtuals (occupied + particle–hole pairs).
- Two-body Slater integrals (:mod:`atomx.numerics.bspline.convol`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class RPAResult:
    """Outcome of an RPA calculation (scaffold)."""

    excitation_energies: np.ndarray = field(default_factory=lambda: np.zeros(0))
    transition_moments: np.ndarray = field(default_factory=lambda: np.zeros(0))
    static_polarizability: Optional[float] = None
    provenance: Dict[str, Any] = field(default_factory=dict)


def compute(*_args, **_kwargs) -> RPAResult:
    """Diagonalise the RPA matrix and extract response properties.

    Not implemented yet — tracking issue: *atomx#rpa*.
    """
    raise NotImplementedError(
        "RPA linear response is a planned method — see methods/rpa.py "
        "scaffold (tracking issue atomx#rpa)."
    )


__all__ = ["RPAResult", "compute"]
