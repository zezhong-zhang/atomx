"""Many-body perturbation theory (MBPT) on top of DHF.  SCAFFOLD.

Planned scope
-------------
Second-order MBPT — the leading correlation correction to a
single-determinant Dirac–Hartree–Fock reference.  Diagrammatically,

    E_MBPT(2) = ¼ Σ_{ab, rs}  |⟨ab‖rs⟩|² / (ε_a + ε_b − ε_r − ε_s)

with (a, b) occupied and (r, s) virtual Dirac orbitals.  Third- and
all-order MBPT-SD in the no-virtual-pair approximation are the
natural follow-ups.

References
----------
- I. Lindgren & J. Morrison, *Atomic Many-Body Theory* (2nd ed., 1986).
- V. A. Dzuba, V. V. Flambaum, M. G. Kozlov, *Phys. Rev. A* **54**,
  3948 (1996) — all-order method for heavy alkali atoms.
- W. R. Johnson, *Atomic Structure Theory* (2007), Chapter 11.

Required inputs (already present in atomx)
------------------------------------------
- Converged DHF orbitals on a B-spline basis (``DFSolver.solve()``).
- Slater-integral two-body machinery
  (:mod:`atomx.numerics.bspline.convol`).
- Angular recoupling (:mod:`atomx.angular`).

This module is a placeholder; see the tracking issue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class MBPTResult:
    """Outcome of an MBPT(2) calculation (scaffold)."""

    order: int = 2
    correlation_energy: float = 0.0
    per_orbital: Dict[str, float] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)


def compute(*_args, **_kwargs) -> MBPTResult:
    """Evaluate second-order MBPT correlation energy.  Not implemented yet.

    Tracking issue: *atomx#mbpt2*.
    """
    raise NotImplementedError(
        "MBPT(2) is a planned method — see methods/mbpt.py scaffold "
        "(tracking issue atomx#mbpt2)."
    )


__all__ = ["MBPTResult", "compute"]
