"""Unified result types for AtomX calculations.

All calculation backends (B-spline DHF, shooting, MCDHF) produce results
that satisfy the ``AtomicResult`` protocol, enabling consistent comparison
across theory levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, runtime_checkable

import numpy as np

from atomx.types import CouplingScheme, TheoryLevel


@dataclass
class OrbitalWavefunction:
    """Radial wavefunction for a single atomic orbital.

    Stores the radial components of an atomic orbital on a numerical grid.

    In atomic physics the full wavefunction factorizes as::

        psi(r, theta, phi) = (1/r) P(r) Omega(theta, phi)

    For the Dirac equation a second (small) component appears::

        psi = (1/r) [P(r) Omega_kappa^mu,  Q(r) Omega_{-kappa}^mu]^T

    Attributes
    ----------
    P : np.ndarray
        Large (upper) radial component.  For non-relativistic calculations
        this is the full radial wavefunction r*R_{nl}(r).
    Q : np.ndarray or None
        Small (lower) radial component (Dirac only).
        Typically |Q| ~ (v/c)|P| — small for light atoms, significant
        for inner shells of heavy atoms.
    n : int
        Principal quantum number.
    l : int
        Orbital angular momentum quantum number.
    kappa : int or None
        Dirac quantum number: kappa = -(l+1) for j=l+1/2,
        kappa = +l for j=l-1/2.  None for non-relativistic.
    occupation : float
        Number of electrons in this orbital.
    energy : float
        One-electron orbital energy in Hartree.
    label : str
        Spectroscopic label, e.g. ``"1s"`` or ``"2p3/2"``.
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
        """Compute <r> = integral(P^2 * r) dr [+ integral(Q^2 * r) dr]."""
        integrand = self.P ** 2 * grid
        if self.Q is not None:
            integrand = integrand + self.Q ** 2 * grid
        return np.trapezoid(integrand, grid)

    @property
    def is_relativistic(self) -> bool:
        """True if this orbital has a small component (Dirac calculation)."""
        return self.Q is not None

    @property
    def j(self) -> Optional[float]:
        """Total angular momentum j = |kappa| - 1/2.  None if non-relativistic."""
        if self.kappa is None:
            return None
        return abs(self.kappa) - 0.5


@dataclass
class Wavefunctions:
    """Collection of radial orbital wavefunctions from a calculation.

    Attributes
    ----------
    grid : np.ndarray
        Radial grid points r in Bohr.
    orbitals : dict
        ``{label: OrbitalWavefunction}`` for each occupied orbital.
    basis_type : str
        ``"bspline"`` or ``"grid"`` (ODE shooting).
    """
    grid: np.ndarray
    orbitals: Dict[str, OrbitalWavefunction] = field(default_factory=dict)
    basis_type: str = "bspline"

    def __repr__(self) -> str:
        n_pts = len(self.grid) if self.grid is not None else 0
        return (f"Wavefunctions({len(self.orbitals)} orbitals on "
                f"{n_pts}-point grid, {self.basis_type})")

    def plot(self, label: str):
        """Plot P(r) [and Q(r)] for the given orbital.

        Parameters
        ----------
        label : str
            Spectroscopic label, e.g. ``"1s"`` or ``"2p3/2"``.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required. Install: pip install matplotlib")
            return

        orb = self.orbitals.get(label)
        if orb is None:
            available = ", ".join(sorted(self.orbitals.keys()))
            raise KeyError(f"'{label}' not found. Available: {available}")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.grid, orb.P, label=f"P(r) [{label}]", linewidth=1.5)
        if orb.Q is not None:
            ax.plot(self.grid, orb.Q, label=f"Q(r) [{label}]",
                    linewidth=1.5, linestyle="--")
        ax.set_xlabel(r"$r$ (Bohr)")
        ax.set_ylabel("Radial wavefunction")
        ax.set_title(f"Orbital {label}  "
                     f"($\\varepsilon$ = {orb.energy:.6f} Ha)")
        ax.legend()
        ax.axhline(0, color="grey", linewidth=0.5)
        plt.tight_layout()
        plt.show()


@dataclass
class HyperfineResult:
    """Magnetic dipole / electric quadrupole hyperfine constants per orbital.

    Populated by :meth:`SCFResult.hyperfine_constants`.

    The ``A_MHz`` entry is the final *ladder* value — the DHF baseline
    multiplied by each enabled rung:

        A_ladder = A_DHF
                 × (1 + a_e)           # electron anomalous moment
                 × (1 + δ_QEDhf)       # QED + Breit corr. to HF operator
                 × (1 + δ_recoil)      # nuclear recoil + Bohr–Weisskopf
                 × (1 + δ_corr)        # electron correlation / core polar.

    Every rung is *optional* and defaults to zero.  This design reflects
    the physics: applying any one rung in isolation can make a prediction
    *worse* because the individual QED corrections cancel each other
    (the canonical example is H 1s — see the hyperfine module docstring).
    For hydrogenic 1s ground states a published preset is available
    via :meth:`SCFResult.hyperfine_constants`' ``preset="h1s"`` /
    ``"d1s"`` / ``"he3_1s"`` kwargs, which inject the Karshenboim 2005
    values for δ_QEDhf and δ_recoil.  With the preset enabled the
    ladder reproduces experiment to ~0.05 %.  Reaching the 1 ppm
    benchmark quoted in the atomic-clock literature needs rung
    coefficients computed specifically for the DHF baseline, which is
    a project unto itself.

    The ``A_MHz_ladder`` dict breaks down the intermediate values at
    each rung for transparency; the ``ladder`` attribute lists which
    rungs were actually applied, in order.

    Attributes
    ----------
    isotope : str
        Label of the isotope used, e.g. ``"H-1"``, ``"Cs-133"``.
    Z, A : int
        Nuclear charge and mass number.
    I : float
        Nuclear spin.
    mu_I : float
        Nuclear magnetic dipole moment (μ_N).
    Q_nuc : float
        Nuclear electric quadrupole moment (barns).
    A_MHz : dict
        ``{orbital_label: A_MHz}`` — final ladder-corrected magnetic
        dipole hyperfine constant in megahertz.
    A_MHz_ladder : dict
        ``{orbital_label: {rung_name: A_MHz}}`` — A value at each rung,
        e.g. ``{"1s1/2": {"dhf": 1421.27, "ae": 1422.92}}``.  The last
        rung value in each orbital entry equals ``A_MHz[label]``.
    ladder : list of str
        Ordered list of ladder-rung names that were applied,
        e.g. ``["dhf", "ae"]``.
    B_MHz : dict
        ``{orbital_label: B_MHz}`` — electric quadrupole hyperfine
        constant in MHz.  Currently empty (``B`` not yet implemented).
    """

    isotope: str
    Z: int
    A: int
    I: float
    mu_I: float
    Q_nuc: float
    A_MHz: Dict[str, float] = field(default_factory=dict)
    A_MHz_ladder: Dict[str, Dict[str, float]] = field(default_factory=dict)
    ladder: list = field(default_factory=list)
    B_MHz: Dict[str, float] = field(default_factory=dict)


@runtime_checkable
class AtomicResult(Protocol):
    """Common interface for all atomic calculation results."""
    converged: bool
    iterations: int
    total_energy: float
    orbital_energies: Dict[str, float]
    energy_breakdown: Dict[str, float]
    theory_level: TheoryLevel
    method_name: str
    wavefunctions: Wavefunctions
    coupling_scheme: CouplingScheme
