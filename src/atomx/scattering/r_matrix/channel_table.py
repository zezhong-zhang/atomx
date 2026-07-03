"""Close-coupling channel enumeration for (L_total, S_total, parity) blocks.

Each scattering channel is labelled by a frozen target state and a partial-wave
angular momentum ``l_c`` of the projectile, coupled to total ``L``.  The
target carries a spin ``S_t``; the projectile is an electron with spin 1/2, so
the allowed total spins ``S_total`` for a given target are obtained by coupling
``S_t`` and 1/2.  Parity = ``(-1)^{l_t + l_c}``.

For an e + He+ close-coupling expansion with target states {1s, 2s, 2p}
and a cutoff ``l_c_max`` on the projectile partial wave, this module
enumerates all (target, l_c) pairs that satisfy

    |l_t - l_c| <= L <= l_t + l_c,   (-1)^{l_t + l_c} == parity,

and, when ``S_total`` is supplied, the spin triangle condition

    |S_t - 1/2| <= S_total <= S_t + 1/2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from atomx.scattering.r_matrix.target_states import TargetState


@dataclass(frozen=True)
class Channel:
    """One close-coupling channel.

    Attributes
    ----------
    index :
        Position in the enumerated channel list (matches the row/column of
        the multi-channel R-matrix / K-matrix).
    target :
        Frozen target state (target electron orbital).
    l_c :
        Projectile partial-wave angular momentum.
    threshold_au :
        Target-state energy (Hartree); equals the channel threshold for
        kinematic open/closed splitting in the outer region.
    """

    index: int
    target: TargetState
    l_c: int
    threshold_au: float


def _spin_couples(two_S_t: int, two_S_total: int) -> bool:
    """Target spin (2*S_t) and projectile spin 1 couple to 2*S_total."""
    return (
        abs(two_S_t - 1) <= two_S_total <= two_S_t + 1
        and (two_S_total - (two_S_t - 1)) % 2 == 0
    )


def enumerate_channels(
    targets: Iterable[TargetState],
    L_total: int,
    parity: int,
    l_c_max: int = 3,
    S_total: float | None = None,
) -> List[Channel]:
    """All target x l_c combinations consistent with (``L_total``, ``parity``).

    Parameters
    ----------
    targets :
        Iterable of :class:`TargetState`.
    L_total :
        Total orbital angular momentum quantum number.
    parity :
        ``+1`` (even) or ``-1`` (odd).
    l_c_max :
        Maximum projectile partial wave to include.
    S_total :
        Optional total spin used to filter target states.  Channels whose
        target spin ``S_t`` cannot couple with a projectile electron
        (spin 1/2) to give ``S_total`` are omitted.  The default ``None``
        retains all targets for backward compatibility.
    """
    if parity not in (-1, +1):
        raise ValueError(f"parity must be +/-1, got {parity}")
    if L_total < 0:
        raise ValueError(f"L_total must be >= 0, got {L_total}")
    if l_c_max < 0:
        raise ValueError(f"l_c_max must be >= 0, got {l_c_max}")

    two_S_total: int | None = None
    if S_total is not None:
        two_S_total = int(round(2 * S_total))

    channels: List[Channel] = []
    idx = 0
    for tgt in targets:
        if two_S_total is not None:
            two_S_t = int(round(2 * tgt.S_t))
            if not _spin_couples(two_S_t, two_S_total):
                continue
        for l_c in range(0, l_c_max + 1):
            if (-1) ** (tgt.l + l_c) != parity:
                continue
            if not (abs(tgt.l - l_c) <= L_total <= tgt.l + l_c):
                continue
            channels.append(
                Channel(
                    index=idx,
                    target=tgt,
                    l_c=l_c,
                    threshold_au=tgt.energy_au,
                )
            )
            idx += 1
    return channels


def channel_labels(channels: Iterable[Channel]) -> List[str]:
    """Human-readable labels, e.g. ``["(1s, εs)", "(2s, εs)", "(2p, εp)"]``."""
    out: List[str] = []
    pw = ["s", "p", "d", "f", "g", "h"]
    for ch in channels:
        sym = pw[ch.l_c] if ch.l_c < len(pw) else f"l={ch.l_c}"
        out.append(f"({ch.target.label}, ε{sym})")
    return out
