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

import math

import numpy as np
from scipy.special import eval_genlaguerre

from atomx.scattering.two_electron.states import TwoElectronTerm


@dataclass(frozen=True)
class TargetState:
    """A target state for close-coupling expansions.

    By default this represents a one-electron orbital (e.g. He+).  The optional
    ``two_electron_term`` field upgrades it to an LS-coupled two-electron term
    such as ``1s2p ^1P``.  In that case ``l`` is the total target orbital
    angular momentum ``L_t`` and ``S_t`` is the total target spin.
    """

    label: str
    n: int
    l: int
    Z: float
    energy_au: float
    P: Callable[[np.ndarray], np.ndarray]  # P(r) = r R(r), vectorised in r
    S_t: float = 0.5
    n_electrons: int = 1
    two_electron_term: TwoElectronTerm | None = None

    @property
    def residual_charge(self) -> float:
        """Net charge of the target state (Z - N_electrons).

        This is the asymptotic Coulomb charge seen by the scattered electron,
        e.g. +1 for He+ and 0 for neutral H or He.
        """
        return self.Z - self.n_electrons


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


def _P_nl(n: int, l: int, Z: float) -> Callable[[np.ndarray], np.ndarray]:
    """General normalized hydrogenic radial function P_{nl}(r) = r R_{nl}(r).

    Uses the closed-form expression

        P_{nl}(r) = sqrt( Z/n^2 * (n-l-1)! / (n+l)! )
                    * x^{l+1} * L_{n-l-1}^{2l+1}(x) * exp(-x/2)

    with x = 2 Z r / n.
    """
    if not (0 <= l < n):
        raise ValueError(f"invalid hydrogenic quantum numbers n={n}, l={l}")
    norm = np.sqrt(
        float(Z) / (n * n)
        * math.factorial(n - l - 1)
        / math.factorial(n + l)
    )
    alpha = 2 * l + 1
    k = n - l - 1

    def P(r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        x = 2.0 * Z * r / n
        # eval_genlaguerre handles x=0 correctly for k>=0.
        lag = eval_genlaguerre(k, alpha, x)
        return norm * np.power(x, l + 1) * lag * np.exp(-0.5 * x)

    return P


def heplus_target_states(Z: float = 2.0, *, n_max: int = 2) -> dict[str, TargetState]:
    """Return hydrogenic He+ target states up to principal quantum number ``n_max``.

    For ``n_max=2`` the returned set is the historical {1s, 2s, 2p} used by
    the Burke 1971 resonance tests.  Larger values expand the close-coupling
    target basis with 3s, 3p, 3d, ... states.  This is the minimal change
    needed to improve the high-energy tail of the 1s -> 2p excitation cross
    section against FAC.

    ``Z`` defaults to 2 (He+); the function is parameterised on Z so the
    same builders can drive unit tests on hydrogenic ions of other charges.
    """
    states: dict[str, TargetState] = {}
    for n in range(1, n_max + 1):
        energy = -0.5 * Z * Z / (n * n)
        for l in range(n):
            label = f"{n}{'spdfgh'[l]}"
            if n <= 2:
                # Keep the existing hand-written builders for the default basis
                # to avoid any accidental regression in the Burke tests.
                if label == "1s":
                    P = _P_1s(Z)
                elif label == "2s":
                    P = _P_2s(Z)
                elif label == "2p":
                    P = _P_2p(Z)
                else:
                    continue
            else:
                P = _P_nl(n, l, Z)
            states[label] = TargetState(
                label, n=n, l=l, Z=Z, energy_au=energy, P=P, S_t=0.5,
                n_electrons=1,
            )
    return states


def helike_target_states(
    Z: float = 2.0, *, n_max: int = 2
) -> dict[str, TargetState]:
    """Return LS-coupled two-electron target states for He-like ions.

    This wraps the two-electron term layer so the R-matrix channel builder can
    consume real He-like target terms (e.g. ``1s^2 ^1S``, ``1s2p ^1P/3P``)
    instead of one-electron hydrogenic orbitals.  It is the entry point for the
    two-electron close-coupling path.
    """
    from atomx.scattering.two_electron.states import helike_terms

    terms = helike_terms(Z=Z, n_max=n_max)
    return {
        term.label: TargetState(
            label=term.label,
            n=term.n2,
            l=term.L,
            Z=Z,
            energy_au=term.energy_au,
            P=term.P2,
            S_t=float(term.S),
            n_electrons=2,
            two_electron_term=term,
        )
        for term in terms
    }
