"""Non-relativistic configuration-string parsing.

One parser, used by both the MCDHF CSF enumerator
(:mod:`atomx.config.csf`) and the jj-averaging layer
(:mod:`atomx.config.jj_coupling`).  Both previously did

    ``n = int(token[0]); l = l_map[token[1]]; q = int(token[2:])``

which fails for a two-digit principal quantum number (``"10s1"`` parses
``token[1] == "0"`` and raises ``KeyError: '0'``) and for the implicit
unit occupation (``"1s2 2s"`` reads ``token[2:] == ""`` and raises
``ValueError: invalid literal for int()``).  See
``docs/archive/2026-09/PHYSICS_NUMERICS_REVIEW_2026-09-19.md`` finding M2.
"""

from __future__ import annotations

import re
from typing import List, Tuple

#: Spectroscopic letter -> orbital angular momentum.
L_MAP: dict[str, int] = {letter: ell for ell, letter in enumerate("spdfghiklmnoqrtuvwxyz")}

#: ``<n><letter><occupation?>`` — occupation defaults to 1.
_SHELL_RE = re.compile(r"^(\d+)([spdfghiklmnoqrtuvwxyz])(\d*)$")


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
                f"{config_str!r}: expected <n><s|p|d|f|g|h|i|k|l|m|n|o|q|r|t|u|v|w|x|y|z><occupation?>, "
                f"e.g. '2p6' or '3s' (occupation defaults to 1)"
            )
        n = int(match.group(1))
        l = L_MAP[match.group(2)]
        q = int(match.group(3)) if match.group(3) else 1
        if n < 1:
            raise ValueError(f"invalid shell {token!r}: n must be >= 1")
        if l >= n:
            raise ValueError(f"invalid shell {token!r}: require 0 <= l < n")
        if q < 0 or q > 2 * (2 * l + 1):
            raise ValueError(
                f"invalid shell {token!r}: occupation {q} exceeds the "
                f"{2 * (2 * l + 1)}-electron capacity of l={l}"
            )
        shells.append((n, l, q))
    return shells


def validate_config_string(config_str: str) -> None:
    """Validate a configuration string, raising :class:`ValueError` if bad.

    Thin wrapper over :func:`parse_nr_shells` for entry points that only
    want the rejection side (Multica HYP-1778): malformed shell tokens
    (``"2x2"``, trailing junk), ``n = 0`` and over-occupied shells
    (``q > 2(2l+1)``) raise a clear ``ValueError`` naming the token here,
    instead of crashing deep in the SCF/QED kernels.  Valid strings
    (including implicit unit occupations, ``"11s"`` == ``"11s1"``) pass
    silently.
    """
    parse_nr_shells(config_str)


__all__ = ["L_MAP", "parse_nr_shells", "validate_config_string"]
