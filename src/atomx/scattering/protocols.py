"""Solver-agnostic protocols for scattering and property calculations.

The scattering layer (PI, RR, EII, CE, AI, DR, R-matrix, …) and the
property layer (transitions, hyperfine, polarisability, EDM, …)
should not hard-depend on a specific physics back-end (Schrödinger
or Dirac) or numerical kernel (B-spline, shooting, MCDHF).  This
module defines the *minimum surface* that any back-end has to expose
so that downstream code can consume it uniformly.

Two layers of protocol:

* :class:`OrbitalLike` — a single bound or continuum spinor.  Carries
  ``P``, ``Q``, ``kappa``, ``occupation``, and ``energy``.  A
  non-relativistic Schrödinger orbital exposes ``Q = None`` (or
  ``np.zeros_like(P)``); the consumer treats it as the relativistic
  small component so the same dipole / Slater integral kernels work.

* :class:`AtomicResultLike` — the bundle returned by any solver.
  Carries:
    - ``Z``: nuclear charge,
    - ``subshells``: iterable of :class:`SubshellLike` (label, kappa, occupation),
    - ``orbital_energies``: dict ``{label: ε}``,
    - ``orbitals_raw``: dict ``{kappa: [(P, Q, occ), …]}``,
    - ``_bsp_data`` (or equivalent grid data) — kernel-specific blob
      passed through opaquely to the numerical helpers that need it.

Every consumer in ``atomx.scattering`` and ``atomx.properties`` should
type-hint its argument as ``AtomicResultLike`` and access only those
fields.  This way:

  - Adding a new SCF backend only requires producing a new
    ``AtomicResultLike``-conformant object.
  - Adding a new scattering process only requires consuming the
    existing protocol.
  - Tests can build mock ``AtomicResultLike`` instances without
    spinning up a real SCF.

This module ships only Protocol / TypedDict / dataclass declarations
— no behaviour.  Existing concrete classes (``SCFResult``,
``MCDHFResult``, ``ShootingResult``) already satisfy the protocol
structurally.

Reference
---------
* PEP 544 (Structural Subtyping / Protocols).
* M. F. Gu, *Astrophys. J.* **657**, 1172 (2007) — FAC's
  data-flow architecture, where the structure / radial / continuum
  layers are similarly decoupled via opaque pointer types.
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
    """Minimum surface for any SCF / MCDHF / shooting result.

    Concrete classes in atomx already satisfy this protocol via
    duck typing — no inheritance required:
        - ``SCFResult`` (B-spline DiracFock)
        - ``MCDHFResult`` (multi-configuration)
        - ``ShootingResult`` (radial shooting)

    Custom backends just need to expose the same attributes.
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
    """Return the kernel-specific grid blob (e.g. B-spline ``bsp_data``)
    if exposed.  Returns ``None`` when the backend doesn't carry one.

    Downstream consumers that need raw grid information should still
    accept results with ``None`` and degrade gracefully — for example,
    by computing on whatever radial mesh is available.
    """
    return getattr(result, "_bsp_data", None)


__all__ = [
    "SubshellLike",
    "AtomicResultLike",
    "get_orbital",
    "list_occupied_orbitals",
    "grid_data",
]
