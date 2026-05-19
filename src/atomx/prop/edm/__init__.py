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
    schiff_moment_enhancement_cm: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def schiff_moment_enhancement(self) -> float:
        """Backward-compatible alias for the Schiff enhancement in cm."""
        return self.schiff_moment_enhancement_cm


def electron_edm_enhancement_sandars(
    *,
    Z: int,
    n_eff: float = 1.0,
    J: float = 0.5,
    L: int = 0,
) -> float:
    """Closed-form Sandars-style electron-EDM enhancement scaling."""
    if Z <= 0:
        raise ValueError("Z must be positive")
    if n_eff <= 0.0:
        raise ValueError("n_eff must be positive")
    if J <= 0.0:
        raise ValueError("J must be positive")
    if L < 0:
        raise ValueError("L must be non-negative")

    angular = 1.0 / (2 * L + 1)
    return 0.004 * (float(Z) ** 3) * angular / (n_eff ** 3 * J * (J + 1.0))


def schiff_moment_enhancement_scaling(
    *,
    Z: int,
    A: int,
    valence: str = "s",
) -> float:
    """Dzuba-Flambaum order-of-magnitude Schiff-moment scaling in cm."""
    if Z <= 0:
        raise ValueError("Z must be positive")
    if A <= 0:
        raise ValueError("A must be positive")
    valence_scale = {
        "s": 1.0,
        "p": 0.35,
        "closed": 0.8,
    }
    if valence not in valence_scale:
        raise ValueError(f"unknown valence {valence!r}")
    return 4.0e-17 * valence_scale[valence] * (float(Z) / 80.0) ** 4


def edm_summary(
    *,
    Z: int,
    A: int,
    n_eff: float = 1.0,
    J: float = 0.5,
    L: int = 0,
    valence: str = "s",
) -> Dict[str, Any]:
    """Return the EDM scaling diagnostics as a plain dictionary."""
    return {
        "Z": Z,
        "A": A,
        "n_eff": n_eff,
        "J": J,
        "L": L,
        "valence": valence,
        "R_electron_edm": electron_edm_enhancement_sandars(
            Z=Z, n_eff=n_eff, J=J, L=L
        ),
        "K_schiff_cm": schiff_moment_enhancement_scaling(
            Z=Z, A=A, valence=valence
        ),
    }


def compute(
    *,
    Z: int | None = None,
    A: int | None = None,
    n_eff: float = 1.0,
    J: float = 0.5,
    L: int = 0,
    valence: str = "s",
) -> EDMResult:
    """Evaluate closed-form EDM enhancement scaling factors."""
    if Z is None or A is None:
        raise NotImplementedError(
            "edm.compute() needs explicit Z and A for the current "
            "closed-form scaling scaffold."
        )
    return EDMResult(
        electron_edm_enhancement=electron_edm_enhancement_sandars(
            Z=Z, n_eff=n_eff, J=J, L=L
        ),
        schiff_moment_enhancement_cm=schiff_moment_enhancement_scaling(
            Z=Z, A=A, valence=valence
        ),
        provenance={
            "model": "Sandars electron EDM + Dzuba-Flambaum Schiff scaling",
            "Z": Z,
            "A": A,
            "n_eff": n_eff,
            "J": J,
            "L": L,
            "valence": valence,
        },
    )


__all__ = [
    "EDMResult",
    "compute",
    "edm_summary",
    "electron_edm_enhancement_sandars",
    "schiff_moment_enhancement_scaling",
]
