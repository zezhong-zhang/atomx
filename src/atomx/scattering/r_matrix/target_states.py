"""Hydrogenic target-state orbitals for the e + He+ close-coupling problem.

Phase 2 of HYP-188 freezes the He+ target electron in one of the
analytic Z=2 hydrogenic states {1s, 2s, 2p}; the scattered electron
moves in the static + exchange potential of that target electron. The
target orbitals are the non-relativistic Coulomb radial functions

    P_{nl}(r) = r R_{nl}(r),    R_{nl} normalized via int R^2 r^2 dr = 1,

so int P_{nl}^2 dr = 1. Energies follow the hydrogenic formula
``epsilon_nl = -Z^2 / (2 n^2)`` in Hartree (l-degenerate at the
non-relativistic level used here).

Reference: any standard QM text; e.g. Bethe & Salpeter (1957) eq. (3.16).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class TargetState:
    """A single hydrogenic target orbital + its energy."""

    label: str
    n: int
    l: int
    Z: float
    energy_au: float
    P: Callable[[np.ndarray], np.ndarray]  # P(r) = r R(r), vectorised in r


def _P_1s(Z: float) -> Callable[[np.ndarray], np.ndarray]:
    norm = 2.0 * Z ** 1.5

    def P(r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        return norm * r * np.exp(-Z * r)

    return P


def _P_2s(Z: float) -> Callable[[np.ndarray], np.ndarray]:
    norm = Z ** 1.5 / (2.0 * np.sqrt(2.0))

    def P(r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        return norm * r * (2.0 - Z * r) * np.exp(-0.5 * Z * r)

    return P


def _P_2p(Z: float) -> Callable[[np.ndarray], np.ndarray]:
    norm = Z ** 2.5 / (2.0 * np.sqrt(6.0))

    def P(r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        return norm * r * r * np.exp(-0.5 * Z * r)

    return P


def heplus_target_states(Z: float = 2.0) -> dict[str, TargetState]:
    """Return the {1s, 2s, 2p} target states of He+ at nuclear charge ``Z``.

    ``Z`` defaults to 2 (He+); the function is parameterised on Z so the
    same builders can drive unit tests on hydrogenic ions of other charges.
    """
    e_1 = -0.5 * Z * Z
    e_2 = -0.5 * Z * Z / 4.0
    return {
        "1s": TargetState("1s", n=1, l=0, Z=Z, energy_au=e_1, P=_P_1s(Z)),
        "2s": TargetState("2s", n=2, l=0, Z=Z, energy_au=e_2, P=_P_2s(Z)),
        "2p": TargetState("2p", n=2, l=1, Z=Z, energy_au=e_2, P=_P_2p(Z)),
    }
