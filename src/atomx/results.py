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
class Waves:
    """Collection of radial orbital wavefunctions from a calculation.

    The short name deliberately mirrors the ``result.waves`` attribute
    under which this container is exposed on every :class:`AtomicResult`.

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
        return (f"Waves({len(self.orbitals)} orbitals on "
                f"{n_pts}-point grid, {self.basis_type})")

    # ------------------------------------------------------------------
    # Radial plotting (1-D)
    # ------------------------------------------------------------------

    def plot(self, label: str, *, backend: str = "mpl", **kwargs):
        """Plot P(r) [and Q(r)] for the given orbital.

        Parameters
        ----------
        label : str
            Spectroscopic label, e.g. ``"1s"`` or ``"2p3/2"``.
        backend : {"mpl", "plotly"}
            Rendering backend.  ``"mpl"`` returns a matplotlib Figure;
            ``"plotly"`` returns an interactive plotly Figure.

        Returns
        -------
        Figure object from the selected backend.
        """
        if backend == "mpl":
            from atomx.viz.mpl import plot_radial_orbital
            return plot_radial_orbital(self, label, **kwargs)
        if backend == "plotly":
            from atomx.viz.plotly import plot_radial_orbital
            return plot_radial_orbital(self, label, **kwargs)
        raise ValueError(
            f"Unknown backend {backend!r}; use 'mpl' or 'plotly'."
        )

    def plot_density(self, labels=None, *, backend: str = "mpl", **kwargs):
        """Plot ``P² + Q²`` radial density for one or more orbitals."""
        if backend == "mpl":
            from atomx.viz.mpl import plot_orbital_density
            return plot_orbital_density(self, labels, **kwargs)
        if backend == "plotly":
            from atomx.viz.plotly import plot_orbital_density
            return plot_orbital_density(self, labels, **kwargs)
        raise ValueError(
            f"Unknown backend {backend!r}; use 'mpl' or 'plotly'."
        )

    # ------------------------------------------------------------------
    # 3-D interactive visualisation (angular + radial)
    # ------------------------------------------------------------------

    def plot3d(
        self,
        label: str,
        *,
        mu: Optional[float] = None,
        box_size: Optional[float] = None,
        n_grid: int = 60,
        isosurface: Optional[float] = None,
        backend: str = "plotly",
        **kwargs,
    ):
        """Render a 3-D isosurface of the orbital amplitude or density.

        Angular dependence is dispatched on the orbital flavour:

        - **Non-relativistic** (``kappa is None``): plots the signed
          amplitude ``psi(r,θ,φ) = P(r)/r · Y_{l,m}(θ,φ)`` with
          ``m = 0`` (unless overridden via ``m=`` kwarg); two
          isosurfaces are drawn at ±iso.
        - **Relativistic** (``kappa is not None``): plots the spinor
          density ``|ψ|² = (P² + Q²)/r² · |Ω_κ^μ(θ,φ)|²`` at the
          chosen magnetic projection ``mu`` (default ``mu = j``); a
          single non-negative isosurface is drawn.

        Parameters
        ----------
        label
            Orbital spectroscopic label.
        mu
            Magnetic-projection quantum number.  Only used for Dirac
            orbitals; default is ``j = |kappa| − 1/2``.
        box_size
            Half-width of the cubic sampling box in Bohr.  Defaults to
            ``3 · ⟨r⟩`` of the orbital.
        n_grid
            Points per Cartesian axis of the sampling grid.
        isosurface
            Iso-value.  Default = 15 % of the absolute maximum of the
            sampled field.
        backend
            Currently only ``"plotly"``.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        if backend != "plotly":
            raise ValueError(
                f"plot3d currently only supports backend='plotly', got {backend!r}."
            )
        from atomx.viz.plotly3d import plot_orbital_3d

        return plot_orbital_3d(
            self,
            label,
            mu=mu,
            box_size=box_size,
            n_grid=n_grid,
            isosurface=isosurface,
            **kwargs,
        )


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

    # ------------------------------------------------------------------
    # Unit-aware accessors (pint)
    # ------------------------------------------------------------------

    def A_as(self, label: str, unit: str = "MHz"):
        """Magnetic hyperfine constant A for an orbital in the target unit.

        Requires pint (``pip install atomx[units]``).  The internal
        unit is MHz; spectroscopic conversion to eV / cm⁻¹ / Hz is
        handled automatically via pint's ``spectroscopy`` context.

        >>> hf.A_as("1s1/2", "Hz")
        <Quantity(1.42e9, 'hertz')>
        >>> hf.A_as("1s1/2", "cm^-1")
        <Quantity(0.0474, '1 / centimeter')>
        """
        if label not in self.A_MHz:
            raise KeyError(
                f"{label!r} not in A_MHz; available: {list(self.A_MHz)}")
        return _as_quantity(self.A_MHz[label], "MHz", unit)

    def B_as(self, label: str, unit: str = "MHz"):
        """Electric-quadrupole hyperfine constant B for an orbital.

        Returns 0-valued Quantity when the orbital has no B entry
        (j = 1/2, or isotope with I ≤ 1/2).
        """
        if label not in self.B_MHz:
            return _as_quantity(0.0, "MHz", unit)
        return _as_quantity(self.B_MHz[label], "MHz", unit)


def _as_quantity(value: float, internal_unit: str, target: str):
    """Helper: convert ``value`` from ``internal_unit`` to ``target``.

    Uses the spectroscopy context transparently for energy ↔ wavenumber
    ↔ frequency conversions (e.g. Hartree → cm⁻¹ or MHz).
    """
    from atomx.units import ureg
    q = float(value) * ureg(internal_unit)
    try:
        return q.to(target)
    except Exception:
        import pint as _pint
        try:
            return q.to(target, "spectroscopy")
        except _pint.errors.DimensionalityError:
            raise


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
    waves: Waves
    coupling_scheme: CouplingScheme
