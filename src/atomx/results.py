"""Unified result types for AtomX calculations.

All calculation backends (B-spline DHF, shooting, MCDHF) produce results
that satisfy the ``AtomicResult`` protocol, enabling consistent comparison
across theory levels.

Wavefunction data is always available via the ``wavefunctions`` property,
regardless of the underlying numerical representation (B-spline coefficients
or tabulated grid values).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, runtime_checkable

import numpy as np

from atomx.types import CouplingScheme, TheoryLevel


# ---------------------------------------------------------------------------
# Wavefunction containers
# ---------------------------------------------------------------------------

@dataclass
class OrbitalWavefunction:
    """Radial wavefunction for a single atomic orbital.

    For non-relativistic (Schrodinger) calculations, only ``P`` is populated
    and ``Q`` is None.  For relativistic (Dirac) calculations, both the large
    component ``P(r)`` and small component ``Q(r)`` are stored.

    Attributes
    ----------
    P : np.ndarray
        Large component (or full radial wavefunction for Schrodinger).
    Q : np.ndarray or None
        Small component (Dirac only).
    n : int
        Principal quantum number.
    l : int
        Orbital angular momentum quantum number.
    kappa : int or None
        Relativistic quantum number (None for non-relativistic).
    occupation : float
        Occupation number.
    energy : float
        Orbital energy in Hartree.
    label : str
        Human-readable label, e.g. ``"1s"`` (NR) or ``"1s1/2"`` (relativistic).
    """
    P: np.ndarray
    Q: Optional[np.ndarray]
    n: int
    l: int
    kappa: Optional[int]
    occupation: float
    energy: float
    label: str = ""

    def expectation_r(self, grid: np.ndarray) -> float:
        """Compute <r> = integral |P|^2 r dr (+ |Q|^2 r dr for Dirac)."""
        integrand = self.P ** 2 * grid
        if self.Q is not None:
            integrand = integrand + self.Q ** 2 * grid
        return np.trapezoid(integrand, grid)

    @property
    def is_relativistic(self) -> bool:
        return self.Q is not None


@dataclass
class Wavefunctions:
    """Container for all orbital wavefunctions from a calculation.

    Attributes
    ----------
    grid : np.ndarray
        Radial grid points r (in Bohr).
    orbitals : dict
        Mapping from orbital label (e.g. ``"1s"`` or ``"1s1/2"``) to
        ``OrbitalWavefunction``.
    basis_type : str
        ``"bspline"`` or ``"grid"`` (shooting solver).
    """
    grid: np.ndarray
    orbitals: Dict[str, OrbitalWavefunction] = field(default_factory=dict)
    basis_type: str = "bspline"

    def __repr__(self) -> str:
        labels = list(self.orbitals.keys())
        n_pts = len(self.grid) if self.grid is not None else 0
        return (f"Wavefunctions({len(labels)} orbitals on {n_pts}-point grid, "
                f"basis={self.basis_type})")

    def plot(self, label: str):
        """Plot the radial wavefunction for a given orbital label.

        Requires matplotlib.  Shows P(r) and Q(r) (if relativistic).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for plotting. Install with: pip install matplotlib")
            return

        orb = self.orbitals.get(label)
        if orb is None:
            available = ", ".join(sorted(self.orbitals.keys()))
            raise KeyError(f"Orbital '{label}' not found. Available: {available}")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.grid, orb.P, label=f"P(r) [{label}]", linewidth=1.5)
        if orb.Q is not None:
            ax.plot(self.grid, orb.Q, label=f"Q(r) [{label}]",
                    linewidth=1.5, linestyle="--")
        ax.set_xlabel("r (Bohr)")
        ax.set_ylabel("Radial wavefunction")
        ax.set_title(f"Orbital {label}  (E = {orb.energy:.6f} Ha)")
        ax.legend()
        ax.axhline(0, color="grey", linewidth=0.5)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Result protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class AtomicResult(Protocol):
    """Common interface for all atomic calculation results.

    Any result object that satisfies this protocol can be used with
    ``atomx.compare()`` and ``atomx.theory_ladder()``.
    """
    converged: bool
    iterations: int
    total_energy: float
    orbital_energies: Dict[str, float]
    energy_breakdown: Dict[str, float]
    theory_level: TheoryLevel
    method_name: str
    wavefunctions: Wavefunctions
    coupling_scheme: CouplingScheme
