"""Unified result types for AtomX calculations.

All calculation backends (B-spline DHF, shooting, MCDHF) produce results
that satisfy the ``AtomicResult`` protocol, enabling consistent comparison
across theory levels.

Key types:

    ``RadialOrbital``: One orbital's radial wavefunction P(r) [and Q(r)].
    ``RadialWavefunctions``: Collection of all orbital wavefunctions on a grid.
    ``AtomicResult``: Protocol satisfied by all result types.
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
class RadialOrbital:
    """Radial wavefunction for a single atomic orbital.

    In atomic physics, the full wavefunction is factored as:

        psi(r, theta, phi) = (1/r) * P(r) * Omega(theta, phi)

    where P(r) is the large (upper) radial component.  For the Dirac
    equation, a second small (lower) component Q(r) appears:

        psi = (1/r) * [ P(r) * Omega_kappa^mu ]
                      [ Q(r) * Omega_{-kappa}^mu ]

    Attributes
    ----------
    large_component : np.ndarray
        P(r), the large radial component.  This is the main wavefunction
        for non-relativistic calculations, and the upper component for Dirac.
    small_component : np.ndarray or None
        Q(r), the small (lower) radial component.  Only present for
        relativistic (Dirac) calculations.  Typically |Q| ~ (v/c)|P|.
    n : int
        Principal quantum number.
    l : int
        Orbital angular momentum quantum number.
    kappa : int or None
        Dirac quantum number.  kappa = -(l+1) for j = l+1/2,
        kappa = +l for j = l-1/2.  None for non-relativistic.
    occupation : float
        Number of electrons in this orbital.
    orbital_energy : float
        One-electron orbital energy in Hartree.
    label : str
        Spectroscopic label, e.g. ``"1s"`` (NR) or ``"2p3/2"`` (Dirac).
    """
    large_component: np.ndarray
    small_component: Optional[np.ndarray]
    n: int
    l: int
    kappa: Optional[int]
    occupation: float
    orbital_energy: float
    label: str = ""

    # Standard aliases used in atomic physics textbooks
    @property
    def P(self) -> np.ndarray:
        """Large component P(r) — standard notation (Grant, Johnson)."""
        return self.large_component

    @property
    def Q(self) -> Optional[np.ndarray]:
        """Small component Q(r) — standard notation (Grant, Johnson)."""
        return self.small_component

    def expectation_r(self, grid: np.ndarray) -> float:
        """Compute <r> = integral(P^2 * r) dr [+ integral(Q^2 * r) dr]."""
        integrand = self.large_component ** 2 * grid
        if self.small_component is not None:
            integrand = integrand + self.small_component ** 2 * grid
        return np.trapezoid(integrand, grid)

    @property
    def is_relativistic(self) -> bool:
        """True if this orbital has a small component (Dirac calculation)."""
        return self.small_component is not None

    @property
    def j(self) -> Optional[float]:
        """Total angular momentum j = |kappa| - 1/2.  None if non-relativistic."""
        if self.kappa is None:
            return None
        return abs(self.kappa) - 0.5


# Backward-compatible alias
OrbitalWavefunction = RadialOrbital


@dataclass
class RadialWavefunctions:
    """Collection of radial orbital wavefunctions from a calculation.

    Provides access to all orbital wavefunctions evaluated on a common
    radial grid, with plotting support.

    Attributes
    ----------
    radial_grid : np.ndarray
        Radial grid points r in Bohr (a_0).
    orbitals : dict
        Mapping from spectroscopic label (e.g. ``"1s"`` or ``"2p3/2"``)
        to ``RadialOrbital``.
    representation : str
        ``"bspline"`` (expansion coefficients evaluated on grid)
        or ``"numerical"`` (tabulated from ODE integration).
    """
    radial_grid: np.ndarray
    orbitals: Dict[str, RadialOrbital] = field(default_factory=dict)
    representation: str = "bspline"

    def __repr__(self) -> str:
        labels = list(self.orbitals.keys())
        n_pts = len(self.radial_grid) if self.radial_grid is not None else 0
        return (f"RadialWavefunctions({len(labels)} orbitals on "
                f"{n_pts}-point grid, {self.representation})")

    def plot(self, label: str):
        """Plot the radial wavefunction P(r) [and Q(r)] for an orbital.

        Parameters
        ----------
        label : str
            Spectroscopic label, e.g. ``"1s"`` or ``"2p3/2"``.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for plotting. "
                  "Install with: pip install matplotlib")
            return

        orb = self.orbitals.get(label)
        if orb is None:
            available = ", ".join(sorted(self.orbitals.keys()))
            raise KeyError(
                f"Orbital '{label}' not found. Available: {available}")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.radial_grid, orb.P, label=f"P(r) [{label}]",
                linewidth=1.5)
        if orb.Q is not None:
            ax.plot(self.radial_grid, orb.Q, label=f"Q(r) [{label}]",
                    linewidth=1.5, linestyle="--")
        ax.set_xlabel(r"$r$ (Bohr)")
        ax.set_ylabel("Radial wavefunction")
        ax.set_title(f"Orbital {label}  "
                     f"($\\varepsilon$ = {orb.orbital_energy:.6f} Ha)")
        ax.legend()
        ax.axhline(0, color="grey", linewidth=0.5)
        plt.tight_layout()
        plt.show()


# Backward-compatible aliases
Wavefunctions = RadialWavefunctions


# ---------------------------------------------------------------------------
# Result protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class AtomicResult(Protocol):
    """Common interface for all atomic calculation results.

    Any result object that satisfies this protocol can be used with
    ``atomx.compare()`` and ``atomx.theory_ladder()``.

    Attributes
    ----------
    converged : bool
        Whether the SCF iteration converged.
    iterations : int
        Number of SCF iterations performed.
    total_energy : float
        Total atomic energy in Hartree.
    orbital_energies : dict
        Mapping from spectroscopic label to one-electron energy (Hartree).
    energy_breakdown : dict
        Decomposition into physical contributions
        (e.g. one-electron, Coulomb, exchange, Breit, QED).
    theory_level : TheoryLevel
        Which level of atomic theory was used.
    method_name : str
        Human-readable method name (e.g. "DiracFock", "MCDHF").
    wavefunctions : RadialWavefunctions
        Radial wavefunctions for all occupied orbitals.
    coupling_scheme : CouplingScheme
        Angular momentum coupling scheme used (LS or jj).
    """
    converged: bool
    iterations: int
    total_energy: float
    orbital_energies: Dict[str, float]
    energy_breakdown: Dict[str, float]
    theory_level: TheoryLevel
    method_name: str
    wavefunctions: RadialWavefunctions
    coupling_scheme: CouplingScheme
