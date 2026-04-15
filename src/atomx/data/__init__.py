"""Cited physics data tables for atomx.

This subpackage hosts numerical tables and fit coefficients that would
otherwise be hard-coded inside computation modules.  Each data module
ships with a full citation, the original units, and the conversion
applied (if any) to atomx's working units (atomic Hartree).

Design
------
Data live as plain Python constants inside small modules rather than as
external YAML/JSON files.  This gives:

- zero additional runtime dependency,
- IDE introspection and type hints for the tables,
- per-module docstrings that live next to the numbers,
- package-data-free wheel builds.

Each module exposes a single ``TABLE`` dict (or a small set of
constants) plus a ``CITATION`` string.  Code that consumes the data
imports from the specific submodule, e.g.::

    from atomx.data.mohr_se import TABLE as MOHR_SE_TABLE

or through the convenience :func:`load` dispatcher below.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

__all__ = ["load", "available_tables"]


_REGISTRY = {
    "mohr_se": "atomx.data.mohr_se",
    "hyperfine_presets": "atomx.data.hyperfine_presets",
    "flambaum_ginges": "atomx.data.flambaum_ginges",
    "nuclear_structure": "atomx.data.nuclear_structure",
}


@lru_cache(maxsize=None)
def load(name: str) -> dict[str, Any]:
    """Return the data payload for a registered table.

    Parameters
    ----------
    name
        Registered key (see :func:`available_tables`).

    Returns
    -------
    dict with keys ``"TABLE"`` and ``"CITATION"`` (and any other
    module-level constants the data module exposes).
    """
    try:
        modname = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown data table {name!r}. Available: {available_tables()}"
        ) from None

    import importlib

    mod = importlib.import_module(modname)
    payload: dict[str, Any] = {}
    for attr in ("TABLE", "CITATION"):
        if hasattr(mod, attr):
            payload[attr] = getattr(mod, attr)
    for attr in dir(mod):
        if attr.isupper() and attr not in payload:
            payload[attr] = getattr(mod, attr)
    return payload


def available_tables() -> list[str]:
    """Return the sorted list of registered data-table names."""
    return sorted(_REGISTRY)
