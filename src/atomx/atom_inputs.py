"""Canonical constructor inputs for :class:`atomx.Atom`.

``Atom(Z, ...)`` answers ten physical questions, one per argument.  Five of
them need more than a single scalar, so each has a frozen dataclass here —
and the argument accepts either the plain value (the common case) or the
object::

    from atomx import Atom, Basis, Corrections, Correlation, Environment

    ne = Atom(
        10,
        config="1s2 2s2 2p6",
        charge=0,
        nuclear="fermi",
        exchange="hf",                     # or Exchange(model=..., alpha_x=...)
        corrections="breit+qed",           # or Corrections(...)
        correlation=None,                  # or "s"/"sd"/Correlation(...)
        engine="bspline",                  # or "shooting"
        basis=Basis(ns=200, r_max=90.0),   # or a Sampling tier
        environment=Environment(plasma_ne=1e22, plasma_Te=100.0),
    )

Each object is a *value*: it validates itself in ``__post_init__`` and is
resolved by :class:`~atomx.Atom` into the flat state the solvers consume.
There is exactly one spelling per physical question — the objects do not
alias retired keywords, they replace them (see ``docs/MIGRATION.md``).

The dataclasses here are deliberately distinct from the legacy enums in
:mod:`atomx.types` (``atomx.types.Exchange``, ``atomx.types.Correlation``),
which are the value objects of the *method* API
(``DiracFock(exchange=Exchange.SLATER)``, ``MCDHF(correlation=...)``).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple

__all__ = [
    "Basis",
    "CORRECTION_PRESETS",
    "Corrections",
    "Correlation",
    "ENVIRONMENT_MODELS",
    "EXCHANGE_MODELS",
    "Environment",
    "Exchange",
    "ResolvedCorrections",
    "SE_BACKENDS",
    "resolve_corrections",
]

#: Exchange models the SCF can be solved with.  ``tfda``/``gsz`` are
#: parametric potentials rather than true exchange functionals and require
#: the B-spline engine.
EXCHANGE_MODELS = ("hf", "slater", "gu", "tfda", "gsz")

#: Self-energy backends accepted by :attr:`Corrections.qed_backend`.
#: ``"auto"`` resolves per element through
#: :func:`atomx.corrections.qed.resolve_se_method`.
SE_BACKENDS = ("auto", "mss", "mohr", "indelicato", "welton",
               "flambaum_ginges", "shabaev_local", "shabaev")

#: Plasma screening models.  Both consume the electron density *and* the
#: temperature (the ion-sphere model of :mod:`atomx.plasma.screening` is
#: deliberately not offered: it has no temperature dependence, so
#: ``plasma_Te`` would be accepted and ignored).
ENVIRONMENT_MODELS = ("debye", "stewart_pyatt")


@dataclass(frozen=True)
class Exchange:
    """The exchange model, plus the knobs only that model reads.

    Attributes
    ----------
    model : {"hf", "slater", "gu", "tfda", "gsz"}
        ``"hf"`` — non-local Hartree–Fock exchange (Slater Y^k expansion);
        exact within the single-determinant approximation.
        ``"slater"`` — local density exchange
        ``V_x = -alpha_x (3 rho / pi)^(1/3)``.
        ``"gu"`` — the same local form with Gu's empirical
        ``alpha_xc(Z, N)`` (FAC's default local-exchange model).
        ``"tfda"``/``"gsz"`` — Thomas–Fermi–Dirac–Amaldi / Green–Sellin–
        Zachor *parametric potentials* (no SCF); B-spline engine only.
    alpha_x : float, optional
        Slater coefficient in atomx's convention
        ``V_x = -alpha_x (3 rho / pi)^(1/3)``.  Only ``model="slater"`` has
        a coefficient to scale, so passing it with any other model raises
        rather than being dropped.  Default ``None`` = 3/2 for
        ``"slater"``, the original Slater potential.
    slater_method : {"slag6", "power_series"}, optional
        Numerical backend for the relativistic Slater integrals of the
        local-exchange paths: ``"slag6"`` (default) = forward+backward
        SLAG6 quadrature, accurate for heavy ``Z``; ``"power_series"`` =
        the legacy power-series ODE (kept for regression comparisons).
        Read by the shooting engine only — the B-spline Galerkin path
        evaluates its Slater integrals with its own quadrature, so asking
        for ``"power_series"`` there raises instead of being ignored.
    """

    model: str = "hf"
    alpha_x: Optional[float] = None
    slater_method: Optional[str] = None

    def __post_init__(self):
        if self.model not in EXCHANGE_MODELS:
            raise ValueError(
                f"model must be one of {', '.join(repr(m) for m in EXCHANGE_MODELS)} "
                f"(got {self.model!r})")
        if self.alpha_x is not None and self.model != "slater":
            raise ValueError(
                f"alpha_x={self.alpha_x!r} only applies to model='slater' "
                f"(got model={self.model!r}): the coefficient scales the "
                f"local Slater potential V_x = -alpha_x (3 rho/pi)^(1/3), "
                f"and the non-local 'hf' and empirical 'gu' models have no "
                f"such coefficient.  Use model='slater' or drop alpha_x.")
        if self.slater_method is not None:
            if self.slater_method not in ("slag6", "power_series"):
                raise ValueError(
                    f"slater_method must be 'slag6' or 'power_series' "
                    f"(got {self.slater_method!r})")
            if self.model not in ("slater", "gu"):
                raise ValueError(
                    f"slater_method={self.slater_method!r} only applies to "
                    f"the local-exchange models 'slater'/'gu' (got "
                    f"model={self.model!r}): 'hf' exchange is evaluated as "
                    f"Slater Y^k integrals, not through a local Slater "
                    f"functional.  Drop slater_method or use "
                    f"model='slater'/'gu'.")

    @property
    def is_local(self) -> bool:
        """True for the models with a local ``rho^(1/3)`` functional."""
        return self.model in ("slater", "gu")

    @property
    def is_parametric(self) -> bool:
        """True for the B-spline-only parametric potentials."""
        return self.model in ("tfda", "gsz")


#: ``corrections=`` shorthand → (breit, qed).
CORRECTION_PRESETS: Dict[str, Tuple[bool, bool]] = {
    "none": (False, False),
    "breit": (True, False),
    "qed": (False, True),
    "breit+qed": (True, True),
}


@dataclass(frozen=True)
class Corrections:
    """Post-SCF relativistic corrections: Breit and QED.

    Attributes
    ----------
    breit : bool, optional
        Include transverse exchange after orbital optimization: a DHF
        expectation or a resolved CSF Hamiltonian contribution. Static
        Breit (Gaunt + retardation) is the default. ``None`` = on.
    qed : bool, optional
        Include the QED corrections (vacuum polarization + self energy).
        Default ``None`` = on.
    qed_backend : str, optional
        Self-energy backend (see :data:`SE_BACKENDS`).  Only meaningful
        when the self energy runs, so ``qed=False`` together with an
        explicit backend raises instead of being ignored.  Default
        ``None`` = ``"auto"``: MSS/Mohr for ``Z <= 36``, Indelicato for
        ``Z >= 37``.
    stack : CorrectionStack, optional
        An explicit :class:`atomx.corrections.CorrectionStack`.  When
        given it *replaces* the ``breit``/``qed``/``qed_backend`` sugar,
        so a flag that contradicts the stack raises rather than being
        silently overridden.
    breit_frequency : {"static", "orbital_differences"}, optional
        Static Breit by default. ``"orbital_differences"`` selects the real
        finite-frequency transverse operator, averaging operators evaluated
        at the two vertex frequencies. Applied on fixed orbitals after Dirac-Coulomb solving.
        An explicit stack must agree with this choice.
    breit_orbital_energies : mapping, optional
        ``(n, kappa) -> energy`` in Hartree for the frequency prescription.
        Ordinary DHF uses its converged orbital energies when omitted.
        CSF calculations (Hamiltonian DF and MCDHF) require an explicit
        complete mapping; correlation-orbital energies are not inferred.
    """

    breit: Optional[bool] = None
    qed: Optional[bool] = None
    qed_backend: Optional[str] = None
    stack: Optional[Any] = None
    breit_frequency: Optional[str] = None
    breit_orbital_energies: Optional[dict] = None

    def __post_init__(self):
        if self.breit_frequency not in (None, 'static', 'orbital_differences'):
            raise ValueError("breit_frequency must be 'static' or 'orbital_differences'")
        if self.breit_frequency == 'orbital_differences' and self.breit is False:
            raise ValueError('finite-frequency Breit requires breit=True')
        if self.breit_orbital_energies is not None:
            frequency = self.breit_frequency or getattr(self.stack, 'breit_frequency', 'static')
            if frequency != 'orbital_differences':
                raise ValueError('breit_orbital_energies requires orbital_differences frequency')
            from atomx.physics.csf.transverse import validate_orbital_energies
            object.__setattr__(self, 'breit_orbital_energies', validate_orbital_energies(
                self.breit_orbital_energies, self.breit_orbital_energies))
        if self.qed_backend is not None and self.qed_backend not in SE_BACKENDS:
            raise ValueError(
                f"qed_backend must be one of "
                f"{', '.join(repr(b) for b in SE_BACKENDS)} "
                f"(got {self.qed_backend!r})")
        if self.qed is False and self.qed_backend not in (None, "auto"):
            raise ValueError(
                f"qed_backend={self.qed_backend!r} is only read when the "
                f"self energy runs, but qed=False switches the whole QED "
                f"correction off (vacuum polarization + self energy).  "
                f"Either drop qed=False (and the backend applies) or drop "
                f"qed_backend={self.qed_backend!r}.")


@dataclass(frozen=True)
class ResolvedCorrections:
    """What :func:`resolve_corrections` decided: the physics that will run."""

    breit: bool
    qed: bool
    qed_backend: str
    stack: Optional[Any]
    breit_frequency: str = 'static'
    breit_orbital_energies: Optional[dict] = None

    @property
    def any(self) -> bool:
        """True when at least one post-SCF correction is switched on."""
        return bool(self.breit or self.qed)


def resolve_corrections(spec=None) -> ResolvedCorrections:
    """Resolve ``corrections=`` (preset string, object or ``None``).

    Parameters
    ----------
    spec : None, str, or Corrections
        ``None`` (both corrections on), a preset name from
        :data:`CORRECTION_PRESETS`, or a :class:`Corrections` object.

    Returns
    -------
    ResolvedCorrections
        The booleans and backend actually in force, plus the explicit
        stack when one was given.

    Raises
    ------
    TypeError
        If ``spec`` is neither ``None``, a preset string, nor a
        :class:`Corrections`.
    ValueError
        If an explicit ``breit``/``qed`` flag contradicts the stack, or
        the preset name is unknown.
    """
    if spec is None:
        spec = Corrections()
    elif isinstance(spec, str):
        key = spec.strip().lower()
        if key not in CORRECTION_PRESETS:
            raise ValueError(
                f"corrections preset must be one of "
                f"{', '.join(repr(k) for k in CORRECTION_PRESETS)} "
                f"(got {spec!r}); pass Corrections(...) for anything else")
        breit, qed = CORRECTION_PRESETS[key]
        spec = Corrections(breit=breit, qed=qed)
    elif not isinstance(spec, Corrections):
        names = ", ".join(f.name for f in fields(Corrections))
        raise TypeError(
            f"corrections must be None, a preset string, or a Corrections "
            f"instance (got {type(spec).__name__}); construct it with "
            f"Corrections({names}) — see help(atomx.Corrections)")

    stack = spec.stack
    if stack is None:
        return ResolvedCorrections(
            breit=True if spec.breit is None else bool(spec.breit),
            qed=True if spec.qed is None else bool(spec.qed),
            qed_backend=spec.qed_backend or "auto",
            stack=None,
            breit_frequency=spec.breit_frequency or 'static',
            breit_orbital_energies=spec.breit_orbital_energies,
        )

    # An explicit stack is authoritative: every flag the caller *did* set
    # must agree with it, or the run would silently do something else.
    stack_breit = bool(getattr(stack, "include_breit", False))
    stack_frequency = getattr(stack, 'breit_frequency', 'static')
    stack_energies = getattr(stack, 'breit_orbital_energies', None)
    if spec.breit_frequency is not None and spec.breit_frequency != stack_frequency:
        raise ValueError('breit_frequency contradicts the correction stack')
    if spec.breit_orbital_energies is not None and spec.breit_orbital_energies != stack_energies:
        raise ValueError('breit_orbital_energies contradicts the correction stack')
    stack_qed = (
        getattr(stack, "se_backend", None) is not None
        or bool(getattr(stack, "include_vp", False))
        or bool(getattr(stack, "include_wk_vp", False)))
    if spec.breit is not None and bool(spec.breit) != stack_breit:
        raise ValueError(
            f"breit={bool(spec.breit)} contradicts the correction stack "
            f"({type(stack).__name__}(include_breit={stack_breit})): the "
            f"stack is authoritative and defines what runs.  Drop "
            f"breit={bool(spec.breit)} and let the stack speak, or rebuild "
            f"the stack with include_breit={bool(spec.breit)}.")
    if spec.qed is not None and bool(spec.qed) != stack_qed:
        raise ValueError(
            f"qed={bool(spec.qed)} contradicts the correction stack "
            f"({type(stack).__name__}(se_backend="
            f"{getattr(stack, 'se_backend', None)!r}, include_vp="
            f"{bool(getattr(stack, 'include_vp', False))})): the stack is "
            f"authoritative and defines what runs.  Drop qed="
            f"{bool(spec.qed)} and let the stack speak, or rebuild the "
            f"stack to match.")
    return ResolvedCorrections(
        breit=stack_breit,
        qed=stack_qed,
        qed_backend=spec.qed_backend or "auto",
        stack=stack,
        breit_frequency=stack_frequency,
        breit_orbital_energies=stack_energies,
    )


@dataclass(frozen=True)
class Correlation:
    """Electron correlation: which CSFs enter, and the CI/SCF protocol.

    Attributes
    ----------
    expansion : {"s", "sd"}, optional
        Auto-generate the configuration expansion from the reference
        configuration (``config=``) by promoting electrons into suggested
        virtual orbitals: ``"s"`` = singles, ``"sd"`` = singles + doubles.
        Closed shells are frozen for ``Z > 10`` (``1s 2s 2p``) and
        ``Z > 4`` (``1s``).  Default ``None`` = no auto-expansion
        (single-configuration DHF, or the explicit list in ``config=``).
    averaging : {"hamiltonian", "statistical", "equal", "fractional", "ci"}, optional
        Single-configuration HF defaults to ``"hamiltonian"``: diagonalize
        the Hamiltonian among the intermediate-coupling CSFs of that one
        nonrelativistic configuration and optimize orbitals using the full
        state energy, including CSF interference. Effective occupations are
        ``sum_j |c_j|**2 n_ij`` (Guerra et al., Eq. 10). No excited
        configurations are generated. Degenerate ground states are averaged
        equally. Both engines solve direct radial DF source equations on the
        self-consistent electron branch, without a finite continuum optimizer.
        Radial variation is Dirac–Coulomb; Breit/QED/recoil retain
        their recorded subsequent correction stages.
        ``"statistical"`` is a separate fixed ensemble with integer jj
        configuration degeneracy weights; ``"equal"`` weights those
        configurations equally. ``"fractional"`` requests products of mean
        occupations. Local exchange defaults to ``"statistical"``.
        ``"ci"`` is reserved for explicit configuration lists/expansions.
    radial_optimization : {"mcdhf", "fac_frozen", "fac_frozen_local"}
        How the multi-configuration outer loop treats the radial shapes:
        ``"mcdhf"`` (default) re-optimises orbitals and CI coefficients
        for the ground state. On the B-spline engine this minimizes the
        correlated Dirac–Coulomb functional in a fixed positive-energy
        nuclear Dirac subspace. Correlation orbitals (for example a compact
        ``3p``) need not represent spectroscopic excited orbitals; use a
        suitable state space or the frozen-core protocol for such channels.
        ``"fac_frozen"`` freezes the reference orbitals
        and optimises only the CI coefficients (FAC-style, cheap);
        ``"fac_frozen_local"`` is the same with the local-exchange radial
        flavour.  B-spline engine only.
    j_resolved_csfs : bool
        Enumerate the jj-CSFs configuration by configuration instead of
        the default m-scheme/fast grouping, so each J block carries its
        own CSF list.  Requires a multi-configuration input.
    per_block_orbital_sets : bool
        Build one orbital set per ``(J, parity)`` block — the DODS/EOL
        scaffold for extended-optimised-level MCDHF.  Requires a
        multi-configuration input.
    orbital_preconditioner : {"spectral", "shooting"}
        Numerical update for the B-spline correlated optimizer. Experimental
        ``"shooting"`` uses a two-sided radial kinetic Green inverse; the
        full CI functional and spectral convergence test are unchanged.
    max_determinants : int, optional
        Cap on the raw determinant dimension of each ``(2J, parity)``
        exact-CI block.  ``None`` (default) keeps the conservative
        built-in cap (2000); raise it when a valence-correlation
        expansion (e.g. lanthanide ``4f`` + singles) legitimately
        exceeds it.  Block build cost grows ~quadratically.
    state_average : CIStateAverage or CIChargeStateAverage, optional
        Optimize a fixed average of resolved CI energies in common
        orthonormal orbitals (B-spline MCDHF only). ``None`` retains the
        ground-state target. Result densities remain ground-state densities;
        ensemble weights/occupations are recorded separately in diagnostics.
        CIChargeStateAverage additionally optimizes explicitly selected charge
        sectors in the same orbitals; it requires Dirac-Coulomb corrections.
    """

    expansion: Optional[str] = None
    averaging: Optional[str] = None
    radial_optimization: str = "mcdhf"
    j_resolved_csfs: bool = False
    per_block_orbital_sets: bool = False
    orbital_preconditioner: str = "spectral"
    max_determinants: Optional[int] = None
    state_average: object = None

    def __post_init__(self):
        if self.state_average is not None:
            from atomx.physics.csf.state_average import (
                CIChargeStateAverage,
                CIStateAverage,
            )
            if not isinstance(self.state_average, (CIStateAverage, CIChargeStateAverage)):
                raise TypeError('state_average must be CIStateAverage, CIChargeStateAverage or None')
            if self.radial_optimization != 'mcdhf' or self.per_block_orbital_sets:
                raise ValueError('CI state averaging requires shared MCDHF optimization')
        if self.orbital_preconditioner not in ("spectral", "shooting"):
            raise ValueError("orbital_preconditioner must be 'spectral' or 'shooting'")
        if self.orbital_preconditioner == "shooting" and self.radial_optimization != "mcdhf":
            raise ValueError("shooting preconditioning requires radial_optimization='mcdhf'")
        if self.expansion not in (None, "s", "sd"):
            raise ValueError(
                f"expansion must be None, 's', or 'sd' "
                f"(got {self.expansion!r})")
        if self.averaging not in (None, "hamiltonian", "ci", "statistical", "equal",
                                  "fractional"):
            raise ValueError(
                f"averaging must be None, 'hamiltonian', 'ci', 'statistical', 'equal', or "
                f"'fractional' (got {self.averaging!r})")
        if self.radial_optimization not in ("mcdhf", "fac_frozen",
                                            "fac_frozen_local"):
            raise ValueError(
                f"radial_optimization must be 'mcdhf', 'fac_frozen', or "
                f"'fac_frozen_local' (got {self.radial_optimization!r})")


@dataclass(frozen=True)
class Basis:
    """Radial basis: B-spline size, radial box, knot grid and SCF start.

    Attributes
    ----------
    ns : int, optional
        Number of B-splines (B-spline engine only).  Default ``None``
        auto-selects the DBSR_HF semi-exponential physics grid, whose node
        count replaces any nominal value; passing ``ns`` explicitly fixes
        the basis and makes the ``grid_*`` overrides inert (which raises,
        rather than dropping them silently).
    ks : int
        B-spline polynomial order.  Default 9 (DBSR_HF).  Higher order
        improves the inner-region representation at high ``Z``.
    r_max : float, optional
        Outer radius of the radial box in Bohr — the box the orbitals are
        confined to, and hence the highest Rydberg state that can be
        represented.  Default ``None`` = ``max(75, 5 n_max^2 / Z)``, large
        enough for the most diffuse requested orbital (a hydrogen 6s needs
        ~180 Bohr).  Used by both engines.
    grid_hi, grid_he, grid_hmax : float, optional
        DBSR_HF knot-shape overrides for the automatic physics grid
        (B-spline engine, ``ns=None``), in Bohr: ``grid_hi`` is the
        innermost step (``hs = hi/Z`` for a point nucleus,
        ``hs = 4 ln(3) a_fermi hi`` for Fermi), ``grid_he`` the
        exponential growth factor (the step is multiplied by ``1 + he``
        per interval) and ``grid_hmax`` the outer constant step.
        Defaults ``None`` keep the DBSR_HF values ``0.25 / 0.25 / 1.0``
        (``he`` is 0.0625 for ``Z <= 2``).  These exist because the
        default grid cannot resolve a high-``Z`` K shell: the
        hydrogen-like U^91+ 1s eigenvalue is +97.7 eV off at the default
        and 0.24 eV off at ``grid_hi=0.005, grid_he=0.10``.
    starting_potential : {"tfda", "gsz"}, optional
        Parametric potential used as the SCF initial guess — a convergence
        aid only, with no effect on a converged solution.  ``None``
        (default) lets the B-spline solver auto-select ``"tfda"`` for
        ``Z >= 20``.
    sampling : Sampling, optional
        Named numerical-discretization tier (e.g.
        :attr:`atomx.sampling.Sampling.high`).  A tier is a *preset*, not a
        second spelling: it supplies ``ns``/``ks``/``r_max`` here, the
        shooting grid ``n_grid``, and :meth:`Atom.solve`'s
        ``tol``/``max_iter``.  A field set on this object always wins over
        the tier.  Default ``None``.
    shooting_space : ShootingSpace, optional
        Finite electron-space controls for explicit MCDHF with shooting.
        Direct Hamiltonian DF instead uses ``sampling.n_grid`` and ``r_max``.
    """

    ns: Optional[int] = None
    ks: int = 9
    r_max: Optional[float] = None
    grid_hi: Optional[float] = None
    grid_he: Optional[float] = None
    grid_hmax: Optional[float] = None
    starting_potential: Optional[str] = None
    sampling: Optional[Any] = None
    shooting_space: Optional[Any] = None

    def __post_init__(self):
        if self.shooting_space is not None:
            from atomx.numerics.shooting.electron_space import ShootingSpace
            if not isinstance(self.shooting_space, ShootingSpace):
                raise TypeError('shooting_space must be ShootingSpace')
        if self.starting_potential not in (None, "tfda", "gsz"):
            raise ValueError(
                f"starting_potential must be None, 'tfda', or 'gsz' "
                f"(got {self.starting_potential!r})")
        if not isinstance(self.ks, int) or self.ks < 4:
            raise ValueError(f"ks must be an integer >= 4 (got {self.ks!r})")
        if self.ns is not None and (not isinstance(self.ns, int)
                                    or self.ns < 10):
            raise ValueError(
                f"ns must be an integer >= 10 or None (got {self.ns!r})")
        for name in ("r_max", "grid_hi", "grid_he", "grid_hmax"):
            value = getattr(self, name)
            if value is not None and not float(value) > 0.0:
                raise ValueError(f"{name} must be > 0 (got {value!r})")


@dataclass(frozen=True)
class Environment:
    """The surrounding plasma: which free-electron sea the atom sits in.

    Both fields describe the same electron sea and are only meaningful
    together: the Debye length is ``lambda_D = sqrt(kT / (4 pi n_e))``, so
    neither alone defines a screening length.  Given both, the screening
    shift is added to the nuclear potential inside the SCF, which raises
    the continuum and lowers the ionisation energy (the ionisation
    potential depression).

    Attributes
    ----------
    plasma_ne : float, optional
        Free-electron density in cm^-3.
    plasma_Te : float, optional
        Free-electron temperature in eV.
    screening : {"debye", "stewart_pyatt"}
        Screening model from :mod:`atomx.plasma.screening`.
        ``"debye"`` (default) is the Debye–Hückel Yukawa potential
        ``-Z exp(-r/lambda_D)/r``, valid for weakly coupled plasmas and
        defined for every charge state.  ``"stewart_pyatt"`` interpolates
        continuously to the ion-sphere limit for strongly coupled plasmas
        and needs a positive net charge to define the ion-sphere radius.
    """

    plasma_ne: Optional[float] = None
    plasma_Te: Optional[float] = None
    screening: str = "debye"

    def __post_init__(self):
        if self.screening not in ENVIRONMENT_MODELS:
            raise ValueError(
                f"screening must be one of "
                f"{', '.join(repr(m) for m in ENVIRONMENT_MODELS)} "
                f"(got {self.screening!r})")
        if (self.plasma_ne is None) != (self.plasma_Te is None):
            missing = "plasma_ne" if self.plasma_ne is None else "plasma_Te"
            given = "plasma_Te" if self.plasma_ne is None else "plasma_ne"
            raise ValueError(
                f"{given} requires {missing} as well: the Debye screening "
                f"length needs the electron density (cm^-3) and the "
                f"electron temperature (eV) together.  Pass {missing}=... "
                f"or drop {given}.")
        for name in ("plasma_ne", "plasma_Te"):
            value = getattr(self, name)
            if value is not None and not float(value) > 0.0:
                raise ValueError(f"{name} must be > 0 (got {value!r})")
