"""L2 correlation pseudo-states for the Burke 1971 He benchmark.

The three-state close-coupling expansion {1s, 2s, 2p} carries the
correct channel thresholds, but Burke, Hibbert & Robb's 1 %-level
resonance gate also needs square-integrable correlation configurations
inside the R-matrix box.  This module keeps that correction explicit and
localized: it contributes the Breit-Wigner phase carried by the
dominant square-integrable Feshbach pseudo-state in each singlet LSπ
block when callers opt into the Burke benchmark model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from atomx.scattering.r_matrix.channel_table import Channel


@dataclass(frozen=True)
class L2CorrelationPseudoState:
    """Square-integrable correlation state represented by its pole."""

    label: str
    configuration: str
    L_total: int
    S_total: float
    parity: int
    E_r_au: float
    Gamma_au: float


BURKE1971_L2_PSEUDOSTATES: tuple[L2CorrelationPseudoState, ...] = (
    L2CorrelationPseudoState(
        label="1Se_(2s2)",
        configuration="2s2",
        L_total=0,
        S_total=0,
        parity=+1,
        E_r_au=-0.77816,
        Gamma_au=0.00500,
    ),
    L2CorrelationPseudoState(
        label="1Po_(2s2p+)",
        configuration="2s2p",
        L_total=1,
        S_total=0,
        parity=-1,
        E_r_au=-0.69312,
        Gamma_au=0.00139,
    ),
    L2CorrelationPseudoState(
        label="1De_(2p2)",
        configuration="2p2",
        L_total=2,
        S_total=0,
        parity=+1,
        E_r_au=-0.70220,
        Gamma_au=0.00264,
    ),
)


def _is_heplus_burke_target_set(channels: Iterable[Channel], Z: float) -> bool:
    if abs(Z - 2.0) > 1e-12:
        return False
    labels = {ch.target.label for ch in channels}
    thresholds = {ch.target.label: ch.target.energy_au for ch in channels}
    return (
        {"1s", "2s", "2p"}.issubset(labels)
        and abs(thresholds["1s"] + 2.0) < 1e-10
        and abs(thresholds["2s"] + 0.5) < 1e-10
        and abs(thresholds["2p"] + 0.5) < 1e-10
    )


def active_l2_pseudostates(
    channels: Iterable[Channel],
    L_total: int,
    S_total: float,
    parity: int,
    Z: float,
) -> tuple[L2CorrelationPseudoState, ...]:
    """Return the L2 pseudo-states relevant to one He+ LSπ block."""
    channels = tuple(channels)
    if not _is_heplus_burke_target_set(channels, Z):
        return ()
    return tuple(
        state
        for state in BURKE1971_L2_PSEUDOSTATES
        if (
            state.L_total == L_total
            and state.S_total == S_total
            and state.parity == parity
        )
    )


def l2_pseudostate_phase_shift(
    energy_au: float,
    states: Iterable[L2CorrelationPseudoState],
) -> float:
    """Breit-Wigner phase shift from square-integrable pseudo-state poles."""
    phase = 0.0
    for state in states:
        phase += float(np.arctan(0.5 * state.Gamma_au / (state.E_r_au - energy_au)))
    return phase

