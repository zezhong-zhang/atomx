"""Unified result types for AtomX calculations.

All calculation backends (B-spline DHF, shooting, MCDHF) produce results
that satisfy the ``AtomicResult`` protocol, enabling consistent comparison
across theory levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, runtime_checkable

import numpy as np

from atomx.numerics.radial_metric import RadialMetric
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
    metric: Optional[RadialMetric] = field(default=None, repr=False)

    def expectation_r(self, grid: np.ndarray) -> float:
        """Compute <r> = integral(P^2 * r) dr [+ integral(Q^2 * r) dr]."""
        integrand = self.P ** 2 * grid
        if self.Q is not None:
            integrand = integrand + self.Q ** 2 * grid
        if self.metric is not None:
            if not np.array_equal(grid, self.metric.radius):
                raise ValueError("grid differs from the orbital native metric")
            return float(self.metric.integrate(integrand))
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

    @property
    def s(self) -> float:
        """Electron spin quantum number — always ``½`` for a single orbital."""
        return 0.5

    @property
    def l_symbol(self) -> str:
        """Spectroscopic letter for the orbital angular momentum
        (``s``, ``p``, ``d``, …)."""
        return "spdfghiklmnoqrtuvwxyz"[self.l] if self.l < 21 else f"l{self.l}"

    def to_dict(self) -> dict:
        """Compact ``{quantum_number → value}`` summary for ``Atom.summary()``
        and external bookkeeping.  Excludes the radial arrays — fetch
        those via the ``P`` / ``Q`` attributes directly."""
        return {
            "label": self.label,
            "n": self.n,
            "l": self.l,
            "l_symbol": self.l_symbol,
            "kappa": self.kappa,
            "j": self.j,
            "s": self.s,
            "occupation": self.occupation,
            "energy_Ha": float(self.energy),
            "is_relativistic": self.is_relativistic,
            "n_grid_points": int(self.P.size) if self.P is not None else 0,
        }


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
    metric: Optional[RadialMetric] = field(default=None, repr=False)

    def __post_init__(self):
        if self.metric is not None:
            if not np.array_equal(self.grid, self.metric.radius):
                raise ValueError("waves grid differs from radial metric")
            for orbital in self.orbitals.values():
                orbital.metric = self.metric

    def __repr__(self) -> str:
        n_pts = len(self.grid) if self.grid is not None else 0
        return (f"Waves({len(self.orbitals)} orbitals on "
                f"{n_pts}-point grid, {self.basis_type})")

    # ------------------------------------------------------------------
    # Label resolution (short-name → full jj label)
    # ------------------------------------------------------------------

    def resolve_label(self, short: str) -> str:
        """Resolve a possibly-abbreviated label to a **single** orbital key.

        Rules applied in order:

        1. **Exact match** — ``"2p3/2"`` returns ``"2p3/2"``.
        2. **Prefix match** — ``"2p"`` matches ``"2p1/2"`` and
           ``"2p3/2"``.  If exactly one orbital matches, that label is
           returned.  If multiple match, the one with the *highest* j
           is chosen (i.e. the dominant fine-structure component
           ``j = l + ½``).
        3. **No match** — raises ``KeyError`` listing available labels.

        Use :meth:`resolve_all` when you want *every* matching
        component (e.g. both ``3d3/2`` and ``3d5/2`` for ``"3d"``).
        """
        if short in self.orbitals:
            return short
        candidates = [k for k in self.orbitals if k.startswith(short)]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            return max(candidates, key=lambda c: self.orbitals[c].j or 0)
        available = ", ".join(sorted(self.orbitals))
        raise KeyError(
            f"orbital {short!r} not found.  Available: {available}"
        )

    def resolve_all(self, short: str) -> list:
        """Resolve an abbreviated label to **all** matching orbital keys.

        ``"3d"`` → ``["3d3/2", "3d5/2"]``.

        Exact matches return a one-element list.  Raises ``KeyError``
        when nothing matches.
        """
        if short in self.orbitals:
            return [short]
        candidates = sorted(
            (k for k in self.orbitals if k.startswith(short)),
            key=lambda c: self.orbitals[c].j or 0,
        )
        if candidates:
            return candidates
        available = ", ".join(sorted(self.orbitals))
        raise KeyError(
            f"orbital {short!r} not found.  Available: {available}"
        )

    def _normalize_labels(self, labels) -> list:
        """Coerce ``labels`` to a flat list of resolved orbital keys.

        Accepts ``None`` (all orbitals), a single string, or a
        sequence of strings.  Each short name is expanded to *all*
        matching jj-components via :meth:`resolve_all`, so
        ``"3d"`` becomes ``["3d3/2", "3d5/2"]``.
        """
        if labels is None:
            return list(self.orbitals.keys())
        if isinstance(labels, str):
            return self.resolve_all(labels)
        out: list = []
        for l in labels:
            out.extend(self.resolve_all(l))
        return out

    # ------------------------------------------------------------------
    # Radial plotting (1-D)
    # ------------------------------------------------------------------

    def plot(self, label: str, *, backend: str = "mpl", **kwargs):
        """Plot P(r) [and Q(r)] for the given orbital(s).

        Parameters
        ----------
        label : str
            Spectroscopic label, e.g. ``"1s"`` or ``"2p3/2"``.
            Short names like ``"2p"`` are expanded to all matching
            fine-structure components (``"2p1/2"`` + ``"2p3/2"``)
            rendered as separate traces / subplots so you can compare
            spin-up and spin-down.
        backend : {"mpl", "plotly"}
            ``"plotly"`` (default for multi-match) adds a dropdown
            menu to switch between components.

        Returns
        -------
        Figure object from the selected backend.
        """
        resolved = self.resolve_all(label)

        if len(resolved) == 1:
            if backend == "mpl":
                from atomx.viz.mpl import plot_radial_orbital
                return plot_radial_orbital(self, resolved[0], **kwargs)
            if backend == "plotly":
                from atomx.viz.plotly import plot_radial_orbital
                return plot_radial_orbital(self, resolved[0], **kwargs)
            raise ValueError(f"Unknown backend {backend!r}.")

        # Multiple components: overlay all on one figure with a toggle.
        if backend == "mpl":
            import matplotlib.pyplot as _plt

            from atomx.viz.mpl import plot_radial_orbital as _mpl_plot
            fig, axes = _plt.subplots(
                1, len(resolved), figsize=(7 * len(resolved), 4.5), sharey=True
            )
            if len(resolved) == 1:
                axes = [axes]
            for ax, lbl in zip(axes, resolved, strict=False):
                _mpl_plot(self, lbl, ax=ax, **kwargs)
            fig.tight_layout()
            return fig

        if backend == "plotly":
            from plotly.subplots import make_subplots

            from atomx.viz.plotly import plot_radial_orbital as _ply_plot

            fig = make_subplots(
                rows=1, cols=len(resolved),
                subplot_titles=resolved,
            )
            for i, lbl in enumerate(resolved, 1):
                sub = _ply_plot(self, lbl, **kwargs)
                for trace in sub.data:
                    trace.name = f"{lbl}  {trace.name}"
                    fig.add_trace(trace, row=1, col=i)
            fig.update_layout(
                title=f"Orbital {label}  (all j components)",
                template="plotly_white",
            )
            return fig

        raise ValueError(f"Unknown backend {backend!r}.")

    def plot_density(self, labels=None, *, backend: str = "mpl", **kwargs):
        """Plot ``P² + Q²`` radial density for one or more orbitals.

        ``labels`` may be ``None`` (all orbitals), a single string
        (``"1s"``), or a list of strings.  Short names are resolved
        via :meth:`resolve_label`.
        """
        labels = self._normalize_labels(labels)
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
        mode: str = "all",
        m: Optional[int] = None,
        mu: Optional[float] = None,
        box_size: Optional[float] = None,
        n_grid: int = 60,
        isosurface: Optional[float] = None,
        backend: str = "plotly",
        **kwargs,
    ):
        """Render 3-D orbital isosurface(s) with an interactive dropdown.

        By default (``mode="all"``) the figure contains a dropdown
        selector with:

        1. **Total (m-averaged)** — radial shell, no angular structure.
        2. **Y_{l,m}** entries for each ``m ∈ {−l, …, +l}``
           (signed amplitude, textbook lobes).
        3. **Ω^κ_μ** entries for each ``μ ∈ {−j, …, +j}``
           (spinor density; Dirac orbitals only).

        Parameters
        ----------
        label
            Orbital spectroscopic label.  Short names (``"2p"``) are
            resolved via :meth:`resolve_label`.
        mode
            ``"all"`` — total + NR + relativistic entries (default).
            ``"nr"`` — total + Y_{l,m} only.
            ``"rel"`` — total + Ω^κ_μ only.
            ``"auto"`` — legacy single surface (m=0 or μ=+j, no dropdown).
        m
            Pin to a single Y_{l,m} surface (no dropdown).
        mu
            Pin to a single Ω^κ_μ surface (no dropdown).
        box_size
            Half-width of the cubic sampling box in Bohr.
        n_grid
            Points per Cartesian axis of the sampling grid.
        isosurface
            Iso-value.  Default = 15 % of the field maximum.
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

        resolved = self.resolve_all(label)
        if len(resolved) == 1:
            return plot_orbital_3d(
                self, resolved[0],
                mode=mode, m=m, mu=mu, box_size=box_size, n_grid=n_grid,
                isosurface=isosurface, **kwargs,
            )

        # Multiple j-components (e.g. "2p" → 2p1/2 + 2p3/2).
        # Build the view catalogue for ALL j-components into one flat
        # dropdown using the same single-trace restyle approach.
        from atomx.viz.plotly3d import (
            _assemble_dropdown_figure,
            _build_cart_grid,
            _collect_views,
            _resolve_box_size,
        )

        # Use the largest ⟨r⟩ among the components for the box.
        all_views = []
        box = None
        for lbl in resolved:
            orb = self.orbitals[lbl]
            b = _resolve_box_size(box_size, orb, self.grid)
            box = max(box or 0.0, b)

        X, Y, Z, r, theta, phi = _build_cart_grid(box, n_grid)
        for lbl in resolved:
            orb = self.orbitals[lbl]
            views = _collect_views(orb, self.grid, r, theta, phi, mode)
            # Prefix each view label with the j-component name.
            for vname, vfield, vsigned in views:
                all_views.append((f"{lbl} : {vname}", vfield, vsigned))

        return _assemble_dropdown_figure(
            X, Y, Z, all_views, label=label, user_iso=isosurface,
            cs_pos=kwargs.get("colorscale_pos", "RdBu_r"),
            cs_neg=kwargs.get("colorscale_neg", "RdBu_r"),
            cs_density=kwargs.get("colorscale_density", "Reds"),
            opacity=kwargs.get("opacity", 0.55),
        )


@dataclass
class ShootingResult:
    """Result of an ODE shooting calculation.

    Satisfies the ``AtomicResult`` protocol from ``atomx.results``.
    """
    converged: bool
    iterations: int
    total_energy: float
    orbital_energies: Dict[str, float]
    method: str = "shooting"
    energy_components: Dict[str, float] = field(default_factory=dict)
    Z: int = 0
    #: Solved subshells (n, kappa, occupation, label) in ``_pa``/``_qa``
    #: row order — the metadata the post-SCF correction adapters need to
    #: pair grid orbitals with occupations and angular quantum numbers.
    subshells: Optional[list] = None
    #: Configuration ensemble defining fractional-occupation pair moments.
    _jj_configs: Optional[list] = field(default=None, repr=False)
    _grid: Optional[np.ndarray] = field(default=None, repr=False)
    _orbital_data: Optional[Dict] = field(default=None, repr=False)
    #: Converged r*V(r) (total local DFS potential), integration weights,
    #: and per-shell (P, Q) waves on ``_grid`` — surfaced so frozen-orbital
    #: solvers (scattering/DR orbital bases) can reuse the FAC-style
    #: optimized potential.  See :func:`atomx.numerics.shooting.local.dfs_frozen_orbital`.
    _rv: Optional[np.ndarray] = field(default=None, repr=False)
    #: Actual r*(V_nucleus + V_environment) used by the orbital solver.
    #: Preserve the sampled field: rebuilding it from a model name can use
    #: different nuclear parameters or lose an external screening potential.
    _rv_nuc: Optional[np.ndarray] = field(default=None, repr=False)
    _difr: Optional[np.ndarray] = field(default=None, repr=False)
    _pa: Optional[np.ndarray] = field(default=None, repr=False)
    _qa: Optional[np.ndarray] = field(default=None, repr=False)
    _equation: str = field(default="dirac", repr=False)
    #: Shooting carries no B-spline workspace; kept as an attribute so
    #: callers that inspect ``result._bsp_data`` uniformly get ``None``
    #: instead of an AttributeError.
    _bsp_data: Optional[Dict] = field(default=None, repr=False)
    #: Exchange MODEL that produced the orbitals: "dfs" for the local
    #: Slater/Gu solvers, "hf" for the non-local
    #: Hartree-Fock solver (:mod:`atomx.numerics.shooting.hartree_fock`).
    _model: str = field(default="dfs", repr=False)
    solver_diagnostics: Dict = field(default_factory=dict)
    radial_convergence: Dict = field(default_factory=dict)
    calculation_spec: object = None

    def save(self, path):
        """Save native orbitals and physical metadata through the I/O layer."""
        from atomx.io.results import save_shooting_result
        return save_shooting_result(self, path)

    @classmethod
    def load(cls, path):
        """Restore a native shooting result, including its correction ledger."""
        from atomx.io.results import load_shooting_result
        return load_shooting_result(path)

    @property
    def energy_breakdown(self) -> Dict[str, float]:
        return self.energy_components

    @property
    def theory_level(self):
        if self.method == "BreitPauli":
            return TheoryLevel.BREIT_PAULI
        if self._equation == "schrodinger":
            return TheoryLevel.HARTREE_FOCK
        if self._model == "hf":
            return TheoryLevel.DIRAC_HARTREE_FOCK
        return TheoryLevel.DIRAC_FOCK_SLATER

    @property
    def method_name(self) -> str:
        if self.method == "BreitPauli":
            return "Breit-Pauli (hydrogenic shifts on HF-Slater orbitals)"
        if self._equation != "schrodinger" and self._model == "hf":
            return "Dirac-Hartree-Fock (shooting)"
        eq = "Schrodinger" if self._equation == "schrodinger" else "Dirac"
        return f"{eq}-Fock-Slater (shooting)"

    @property
    def coupling_scheme(self):
        if self._equation == "schrodinger":
            return CouplingScheme.LS
        return CouplingScheme.JJ

    @property
    def radial_state(self):
        """Owned native radial snapshot for correction and analysis consumers."""
        from atomx.radial_state import radial_state_of
        if '_radial_state' not in self.__dict__:
            self.__dict__['_radial_state'] = radial_state_of(self)
        return self.__dict__['_radial_state']

    @property
    def waves(self):
        if self._grid is None:
            return Waves(grid=np.array([]), basis_type="grid")

        orbitals = {}
        if self._orbital_data is not None:
            for label, data in self._orbital_data.items():
                orbitals[label] = OrbitalWavefunction(
                    P=data["P"], Q=data.get("Q"),
                    n=data["n"], l=data["l"],
                    kappa=data.get("kappa"),
                    occupation=data["occupation"],
                    energy=self.orbital_energies.get(label, 0.0),
                    label=label)
        elif self._pa is not None and self.subshells:
            # The SCF solvers surface the converged grid orbitals as the
            # raw ``_pa``/``_qa`` arrays (row order == ``subshells``
            # order) rather than an ``_orbital_data`` dict — build the
            # protocol Waves from them so ``Atom.waves`` / ``.density()``
            # / ``.r_uniform()`` work uniformly across engines.
            for i, sh in enumerate(self.subshells):
                Q = None
                if self._equation != "schrodinger" and self._qa is not None:
                    Q = np.asarray(self._qa[i])
                orbitals[sh.label] = OrbitalWavefunction(
                    P=np.asarray(self._pa[i]), Q=Q,
                    n=sh.n, l=sh.l, kappa=sh.kappa,
                    occupation=float(sh.occupation),
                    energy=self.orbital_energies.get(sh.label, 0.0),
                    label=sh.label)
        if not orbitals:
            return Waves(grid=np.array([]), basis_type="grid")
        return Waves(
            grid=self._grid, orbitals=orbitals, basis_type="grid",
            metric=(RadialMetric.shooting(self._grid, self._difr)
                    if self._difr is not None else None))

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
