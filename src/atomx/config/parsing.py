"""Non-relativistic configuration-string parsing.

One parser, used by both the MCDHF CSF enumerator
(:mod:`atomx.methods.mcdhf.config`) and the jj-averaging layer
(:mod:`atomx.config.jj_coupling`).  Both previously did

    ``n = int(token[0]); l = l_map[token[1]]; q = int(token[2:])``

which fails for a two-digit principal quantum number (``"10s1"`` parses
``token[1] == "0"`` and raises ``KeyError: '0'``) and for the implicit
unit occupation (``"1s2 2s"`` reads ``token[2:] == ""`` and raises
``ValueError: invalid literal for int()``).  See
``docs/PHYSICS_NUMERICS_REVIEW_2026-09-19.md`` finding M2.
"""

from __future__ import annotations

import re
from typing import List, Tuple

#: Spectroscopic letter -> orbital angular momentum.
L_MAP: dict[str, int] = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6}

#: ``<n><letter><occupation?>`` — occupation defaults to 1.
_SHELL_RE = re.compile(r"^(\d+)([spdfghi])(\d*)$")


def parse_nr_shells(config_str: str) -> List[Tuple[int, int, int]]:
    """Parse a configuration string into ``(n, l, occupation)`` triples.

    Parameters
    ----------
    config_str : str
        Whitespace-separated shells in non-relativistic notation, e.g.
        ``"1s2 2s2 2p6"``.  The occupation may be omitted, in which case
        it is 1 (``"1s2 2s"`` == ``"1s2 2s1"``).

    Returns
    -------
    list of (int, int, int)
        ``(n, l, q)`` per shell, in the order written.

    Raises
    ------
    ValueError
        If a token is not a valid shell.  The message names the offending
        token and the accepted forms, rather than surfacing an opaque
        ``KeyError``/``int()`` failure.
    """
    shells: List[Tuple[int, int, int]] = []
    if config_str is None:
        return shells
    for token in str(config_str).strip().split():
        match = _SHELL_RE.match(token)
        if match is None:
            raise ValueError(
                f"invalid shell {token!r} in configuration "
                f"{config_str!r}: expected <n><s|p|d|f|g|h|i><occupation?>, "
                f"e.g. '2p6' or '3s' (occupation defaults to 1)"
            )
        n = int(match.group(1))
        l = L_MAP[match.group(2)]
        q = int(match.group(3)) if match.group(3) else 1
        if n < 1:
            raise ValueError(f"invalid shell {token!r}: n must be >= 1")
        if q < 0 or q > 2 * (2 * l + 1):
            raise ValueError(
                f"invalid shell {token!r}: occupation {q} exceeds the "
                f"{2 * (2 * l + 1)}-electron capacity of l={l}"
            )
        shells.append((n, l, q))
    return shells


__all__ = ["L_MAP", "parse_nr_shells"]
