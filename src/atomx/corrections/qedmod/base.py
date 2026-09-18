"""Base class and registry for self-energy backends.

Each backend subclasses :class:`SEBackend` and is registered into
:data:`SE_BACKENDS` at import time via :func:`register_backend`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

OrbitalTuple = Tuple  # (P, Q, occ, kappa, label)


@dataclass
class SEResult:
    """Result of a self-energy backend evaluation.

    Attributes
    ----------
    E_SE : float
        Total self-energy correction in Hartree (sum over occupied orbitals,
        weighted by occupation number).
    per_orbital : dict
        ``{label: E_SE_a}`` — per-orbital (per-electron, i.e. not yet
        multiplied by the occupation number) self-energy contributions.
    method : str
        Name of the backend that produced this result.
    """

    E_SE: float
    per_orbital: Dict[str, float] = field(default_factory=dict)
    method: str = ""


class SEBackend(ABC):
    """Abstract interface for a self-energy backend.

    A backend is a strategy for computing the one-electron QED self-energy
    contribution ⟨a|Σ_SE|a⟩ for every occupied orbital ``a`` of a Dirac–Fock
    calculation.  Implementations range from tabulated hydrogenic fits
    (Mohr) to full model-operator expansions (Shabaev).
    """

    #: Short registration name used by :func:`get_backend`.
    name: str = ""

    #: One-line description, surfaced in help / compare output.
    description: str = ""

    @abstractmethod
    def compute(
        self,
        orbitals: List[OrbitalTuple],
        bsp_data: Dict,
        Z: int,
    ) -> SEResult:
        """Compute the self-energy correction.

        Parameters
        ----------
        orbitals
            List of ``(P, Q, occ, kappa, label)`` tuples as produced by
            :meth:`DFSolver._flatten_orbitals`.
        bsp_data
            B-spline workspace dict (grid, weights, basis).
        Z
            Nuclear charge.

        Returns
        -------
        :class:`SEResult`
        """


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SE_BACKENDS: Dict[str, SEBackend] = {}
SE_ALIASES = {
    "mss": "mohr",
}


def register_backend(cls):
    """Class decorator that instantiates ``cls`` and registers it.

    The class is expected to set ``name`` (and optionally ``description``)
    as a class attribute.
    """
    if not cls.name:
        raise ValueError(f"SE backend {cls.__name__} must define a non-empty .name")
    if cls.name in SE_BACKENDS:
        raise ValueError(f"SE backend name {cls.name!r} already registered")
    SE_BACKENDS[cls.name] = cls()
    return cls


def available_backends() -> List[str]:
    """Return the list of registered backend names (in registration order)."""
    return list(SE_BACKENDS.keys())


def get_backend(name: str) -> SEBackend:
    """Return the backend instance registered under ``name``."""
    name = SE_ALIASES.get(name, name)
    try:
        return SE_BACKENDS[name]
    except KeyError:
        raise KeyError(
            f"Unknown self-energy backend {name!r}. "
            f"Available: {available_backends()}"
        ) from None
