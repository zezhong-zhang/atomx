"""LS-coupled two-electron target states for He-like ions.

The layer is intentionally narrow: it supports ``1s^2`` and ``1s n l``
configurations, which are sufficient for the R-matrix CE benchmark and for the
core-hole + excited-electron DR resonances that dominate Fe XVII.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from atomx.scattering.r_matrix.target_states import TargetState
from atomx.scattering.two_electron import coulomb


@dataclass(frozen=True)
class TwoElectronTerm:
    """An LS-coupled two-electron term.

    Attributes
    ----------
    label
        Human-readable label such as ``"1s2p ^1P"``.
    n1, l1, n2, l2
        Principal and orbital quantum numbers of the two electrons.
    S, L
        Total spin and orbital angular momentum quantum numbers.
    parity
        ``+1`` or ``-1``.
    energy_au
        Term energy in Hartree (including one-electron + Coulomb parts).
    P1, P2
        Callable radial functions ``P(r) = r R(r)`` for the two electrons.
    """

    label: str
    n1: int
    l1: int
    n2: int
    l2: int
    S: int
    L: int
    parity: int
    energy_au: float
    P1: Callable[[np.ndarray], np.ndarray]
    P2: Callable[[np.ndarray], np.ndarray]

    @property
    def configuration(self) -> str:
        """Configuration string, e.g. ``1s.2p``."""
        syms = "spdfgh"
        return f"{self.n1}{syms[self.l1]}.{self.n2}{syms[self.l2]}"

    def to_target_state(self, Z: float) -> TargetState:
        """Wrap as a one-entry ``TargetState`` for the current R-matrix API.

        The returned object carries the two-electron term as its label and the
        term energy.  This is a temporary bridge; Milestone 1 will generalise
        ``TargetState`` to hold the term directly.
        """
        return TargetState(
            label=self.label,
            n=max(self.n1, self.n2),
            l=self.L,
            Z=Z,
            energy_au=self.energy_au,
            P=self.P1,
        )


def _orbital_label(n: int, l: int) -> str:
    return f"{n}{'spdfgh'[l]}"


def helike_terms(
    Z: float,
    *,
    n_max: int = 2,
    n_grid: int = 2000,
    r_max: float | None = None,
) -> tuple[TwoElectronTerm, ...]:
    """Generate LS-coupled two-electron terms for He-like ions.

    Parameters
    ----------
    Z
        Nuclear charge (e.g. 6 for C V, 10 for Ne IX).
    n_max
        Maximum principal quantum number of the excited electron.
    n_grid, r_max
        Parameters passed to the radial Slater-integral evaluation.

    Returns
    -------
    tuple of TwoElectronTerm
        Sorted by energy.  Includes the ``1s^2 ^1S`` ground term and all
        ``1s n l ^1L`` / ``^3L`` terms with ``2 <= n <= n_max``.
    """
    from atomx.scattering.r_matrix.target_states import heplus_target_states

    orbitals = heplus_target_states(Z=Z, n_max=n_max)

    # 1s ground-state energy and radial function.
    s1s = orbitals["1s"]
    e_1s = s1s.energy_au
    P_1s = s1s.P

    terms: list[TwoElectronTerm] = []

    # 1s^2 ^1S closed shell.
    e_1s2 = coulomb.term_energy_1s2(
        P_1s, e_1s, n_points=n_grid, r_max=r_max,
    )
    terms.append(TwoElectronTerm(
        label="1s^2 ^1S",
        n1=1, l1=0, n2=1, l2=0,
        S=0, L=0, parity=+1,
        energy_au=e_1s2,
        P1=P_1s, P2=P_1s,
    ))

    # 1s n l terms.
    for n in range(2, n_max + 1):
        for l in range(n):
            orbital_label = _orbital_label(n, l)
            st = orbitals[orbital_label]
            e_nl = st.energy_au
            P_nl = st.P
            for S in (0, 1):
                multiplicity = 2 * S + 1
                term_label = f"1s{orbital_label} ^{multiplicity}{'spdfgh'[l].upper()}"
                e_term = coulomb.term_energy_1s_nl(
                    P_1s, P_nl, e_1s, e_nl, l, S,
                    n_points=n_grid, r_max=r_max,
                )
                terms.append(TwoElectronTerm(
                    label=term_label,
                    n1=1, l1=0, n2=n, l2=l,
                    S=S, L=l, parity=(-1) ** l,
                    energy_au=e_term,
                    P1=P_1s, P2=P_nl,
                ))

    return tuple(sorted(terms, key=lambda t: t.energy_au))
