"""Close-coupling channel enumeration for (L_total, S_total, parity) blocks.

Each scattering channel is labelled by a frozen target orbital ``(n_t, l_t)``
and a partial-wave angular momentum ``l_c`` of the projectile, coupled to
total ``L``. Spin couples to total ``S``. Parity = ``(-1)^{l_t + l_c}``.

For an e + He+ close-coupling expansion with target states {1s, 2s, 2p}
and a cutoff ``l_c_max`` on the projectile partial wave, this module
enumerates all (target, l_c) pairs that satisfy

    |l_t - l_c| <= L <= l_t + l_c,   (-1)^{l_t + l_c} == parity.
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


def enumerate_channels(
    targets: Iterable[TargetState],
    L_total: int,
    parity: int,
    l_c_max: int = 3,
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
    """
    if parity not in (-1, +1):
        raise ValueError(f"parity must be +/-1, got {parity}")
    if L_total < 0:
        raise ValueError(f"L_total must be >= 0, got {L_total}")
    if l_c_max < 0:
        raise ValueError(f"l_c_max must be >= 0, got {l_c_max}")

    channels: List[Channel] = []
    idx = 0
    for tgt in targets:
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
