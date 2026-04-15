"""Coupled-cluster correlation methods.  SCAFFOLD.

Planned scope
-------------
CCSD and CCSD(T) on a DHF reference, along with Fock-space CC for
open-shell atoms.  The CCSD amplitude equations are

    ⟨Φ_i^a | H̄ | Φ_0⟩         = 0   (T₁)
    ⟨Φ_{ij}^{ab} | H̄ | Φ_0⟩   = 0   (T₂)

with  H̄ = e^{−T} H e^{T}, solved iteratively.  (T) adds the leading
perturbative triples correction.

References
----------
- R. J. Bartlett & M. Musiał, *Rev. Mod. Phys.* **79**, 291 (2007).
- H. S. Nataraj, B. K. Sahoo, B. P. Das, D. Mukherjee,
  *Phys. Rev. Lett.* **101**, 033002 (2008) — relativistic CCSD for
  clock transitions.
- E. Eliav & U. Kaldor, *J. Chem. Phys.* **103**, 3310 (1995) —
  Fock-space CC for the relativistic problem.

Required inputs (present in atomx)
----------------------------------
- DHF reference + virtual orbitals on the B-spline basis.
- Two-body Slater integrals
  (:mod:`atomx.numerics.bspline.convol`).

Storage
-------
For ``N`` occupied × ``M`` virtual orbitals the T₂ tensor has
shape ``(N, N, M, M)`` — a B-spline basis of ``ns ≈ 100`` and a few
core orbitals gives ~10⁶–10⁸ entries, so a sparse / integral-direct
implementation is planned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class CoupledClusterResult:
    """Outcome of a CCSD / CCSD(T) calculation (scaffold)."""

    method: str = "CCSD"
    correlation_energy: float = 0.0
    triples_correction: float = 0.0
    n_iterations: int = 0
    provenance: Dict[str, Any] = field(default_factory=dict)


def compute(*_args, **_kwargs) -> CoupledClusterResult:
    """Solve the CC amplitude equations.  Not implemented yet.

    Tracking issue: *atomx#ccsd*.
    """
    raise NotImplementedError(
        "Coupled-cluster is a planned method — see "
        "methods/coupled_cluster.py scaffold (tracking issue atomx#ccsd)."
    )


__all__ = ["CoupledClusterResult", "compute"]
