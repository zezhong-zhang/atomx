"""Shared structural contracts and coefficient access for native atomic results.

These describe the legacy coefficient-facing interface. Consumers needing
sampled orbitals and native integration weights use ``atomx.radial_state``.
Result conversion does not launch a solver.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)


@runtime_checkable
class SubshellLike(Protocol):
    """A single subshell (n, κ) carrying an occupation."""

    label: str
    kappa: int
    occupation: float


@runtime_checkable
class AtomicResultLike(Protocol):
    """Coefficient-facing interface used by native provider consumers.

    B-spline SCF and correlated results expose this surface structurally.
    Sampled shooting payloads instead use ``RadialState``; their having a
    radial mesh does not imply an ``orbitals_raw`` coefficient interface.
    """

    Z: int
    subshells: Iterable[SubshellLike]
    orbital_energies: Dict[str, float]
    orbitals_raw: Dict[int, List[Tuple[Any, Any, float]]]


def get_orbital(
    result: AtomicResultLike,
    label: str,
) -> Tuple[Any, Any, int, float, float]:
    """Convenience accessor: return ``(P, Q, kappa, energy, occupation)``
    for a named subshell.

    Looks up ``label`` in ``result.subshells``, then pulls the matching
    coefficients from ``result.orbitals_raw[kappa][...]``.

    Parameters
    ----------
    result
    label : str

    Returns
    -------
    (P, Q, kappa, energy_au, occupation)

    Raises
    ------
    KeyError
        If the subshell label is not found.
    """
    sh = next((s for s in result.subshells if s.label == label), None)
    if sh is None:
        raise KeyError(
            f"Subshell {label!r} not found; available: "
            f"{[s.label for s in result.subshells]}"
        )
    kappa = sh.kappa
    eps = result.orbital_energies.get(label, 0.0)
    occ = float(sh.occupation)
    same_kappa_subs = [s for s in result.subshells if s.kappa == kappa]
    idx = same_kappa_subs.index(sh)
    P, Q, _ = result.orbitals_raw[kappa][idx]
    return P, Q, kappa, eps, occ


def list_occupied_orbitals(
    result: AtomicResultLike,
) -> List[Tuple[str, int, float, float]]:
    """Return ``[(label, kappa, energy_au, occupation), …]`` for every
    occupied subshell."""
    return [
        (s.label, s.kappa, result.orbital_energies.get(s.label, 0.0),
         float(s.occupation))
        for s in result.subshells
        if s.occupation > 0
    ]


def grid_data(result: AtomicResultLike) -> Optional[Dict[str, Any]]:
    """Return the native radial workspace, or None when none is exposed.

    B-spline workspaces contain basis data; shooting workspaces contain
    sampled radii and native quadrature weights. A malformed advertised
    workspace raises rather than substituting another integration rule.
    """
    from atomx.radial_state import radial_workspace_of

    if (hasattr(type(result), 'radial_state')
            or getattr(result, '_bsp_data', None) is not None
            or getattr(result, 'bsp_data', None) is not None
            or getattr(result, '_grid', None) is not None):
        return radial_workspace_of(result)[0]
    return None


__all__ = [
    "SubshellLike",
    "AtomicResultLike",
    "get_orbital",
    "list_occupied_orbitals",
    "grid_data",
]
