"""High-level Atom API for AtomX.

Simple interface for common atomic structure calculations:

    from atomx import Atom

    ne = Atom(10)                       # Neon
    ne.solve()                          # Run DHF SCF (Breit + QED on by default)
    print(ne.total_energy)              # -128.61 Hartree
    print(ne.orbital_energies)          # {1s: -32.6, 2s: -1.6, ...}

    # Correction-free (pure Dirac-Coulomb) comparison runs:
    ne = Atom(10, corrections="none")
    ne.solve()

    # Multi-configuration:
    be = Atom(4, config=["1s2 2s2", "1s2 2p2"])
    be.solve()
    print(be.state_energies)
"""

from dataclasses import fields as _dataclass_fields
from typing import Optional

import numpy as np

from atomx.atom_inputs import (
    Basis,
    Corrections,
    Correlation,
    Environment,
    Exchange,
    resolve_corrections,
)
from atomx.config import ground_state_config
from atomx.constants import HARTREE_CM1, HARTREE_EV
from atomx.methods.dirac_fock import DFSolver
from atomx.sampling import Sampling

#: Radial ENGINES the Atom facade accepts.
_ENGINES = ("bspline", "shooting")

#: Documented schema defaults of the accuracy tiers (:class:`Sampling`).  A
#: tier field still at its schema default means "not asked for", so the
#: engine keeps its own tuned value (auto radial box; n_grid=1000 for HF /
#: 400 for the local models).
_TIER_DEFAULTS = {f.name: f.default for f in _dataclass_fields(Sampling)}

#: Removed legacy spellings → the canonical (engine, exchange model) pair
#: that reproduces them bit-identically.  Kept only to build a helpful
#: ValueError; the strings themselves are no longer accepted.
_REMOVED_EXCHANGE = {
    "fac": ("shooting", "gu"),
    "shooting": ("shooting", "slater"),
}

#: Retired constructor keywords → the replacement spelling.  The value is
#: used only in the message, so it stays a plain string.
_RETIRED_INPUTS = {
    "configs": "config= now takes the list directly: "
               "config=[\"1s2 2s2\", \"1s2 2p2\"]",
    "numerics": "engine= selects the radial engine: engine=\"shooting\"",
    "breit": "corrections= carries the Breit switch: "
             "corrections=Corrections(breit=False) or corrections=\"qed\"",
    "qed": "corrections= carries the QED switch: "
           "corrections=Corrections(qed=False) or corrections=\"breit\"",
    "qed_backend": "corrections=Corrections(qed_backend=...) carries the "
                   "self-energy backend",
    "correction_stack": "corrections=Corrections(stack=...) carries the "
                        "explicit correction bundle",
    "alpha_x": "exchange=Exchange(model=\"slater\", alpha_x=...) carries the "
               "Slater coefficient",
    "slater_method": "exchange=Exchange(model=\"slater\"|\"gu\", "
                     "slater_method=...) carries the Slater-integral backend",
    "ns": "basis=Basis(ns=...) carries the B-spline count",
    "ks": "basis=Basis(ks=...) carries the B-spline order",
    "r_max": "basis=Basis(r_max=...) carries the radial box",
    "grid_hi": "basis=Basis(grid_hi=...) carries the knot-grid override",
    "grid_he": "basis=Basis(grid_he=...) carries the knot-grid override",
    "grid_hmax": "basis=Basis(grid_hmax=...) carries the knot-grid override",
    "starting_potential": "basis=Basis(starting_potential=...) carries the "
                          "SCF initial guess",
    "sampling": "basis=Basis(sampling=...) carries the accuracy tier "
                "(solve(sampling=...) is unchanged)",
    "averaging": "correlation=Correlation(averaging=...) carries the CSF "
                 "weights",
    "radial_optimization": "correlation=Correlation(radial_optimization=...) "
                           "carries the MCDHF protocol",
    "j_resolved_csfs": "correlation=Correlation(j_resolved_csfs=...) carries "
                       "the CSF enumeration mode",
    "per_block_orbital_sets": "correlation=Correlation("
                              "per_block_orbital_sets=...) carries the "
                              "per-block orbital sets",
    "plasma_ne": "environment=Environment(plasma_ne=..., plasma_Te=...) "
                 "carries the plasma",
    "plasma_Te": "environment=Environment(plasma_ne=..., plasma_Te=...) "
                 "carries the plasma",
    "system": "the System group was removed: charge= and nuclear= are "
              "top-level arguments",
    "electronic": "the ElectronicStructure group was removed: config= now "
                  "takes the configuration or the list of configurations",
    "interaction": "the Interaction group was removed: exchange= and "
                   "corrections= replace it",
    "ci": "the CIProtocol group was removed: correlation=Correlation(...) "
          "replaces it",
}

#: Valid keywords, for the "unknown argument" message.
_KNOWN_INPUTS = ("Z, config, charge, nuclear, exchange, corrections, "
                 "correlation, engine, basis, environment")


def _reject_retired(retired):
    """Raise a message naming the replacement for each retired keyword."""
    if not retired:
        return
    lines = []
    for name, value in retired.items():
        replacement = _RETIRED_INPUTS.get(name)
        if replacement is None:
            lines.append(
                f"  {name}={value!r} is not an Atom argument; "
                f"valid arguments are {_KNOWN_INPUTS}")
        else:
            lines.append(f"  {name}= was retired; {replacement}")
    raise TypeError(
        "Atom() got " + ("an unexpected keyword argument"
                         if len(retired) == 1 else "unexpected keyword arguments")
        + ":\n" + "\n".join(lines)
        + "\nSee docs/MIGRATION.md for the full old -> new table.")


def _coerce_exchange(exchange):
    """Normalise the ``exchange=`` argument to an :class:`Exchange`."""
    value = getattr(exchange, "value", exchange)  # atomx.types.Exchange members
    if isinstance(value, Exchange):
        return value
    if isinstance(value, str):
        if value in _REMOVED_EXCHANGE:
            engine, model = _REMOVED_EXCHANGE[value]
            raise ValueError(
                f"exchange={value!r} was removed; use "
                f"engine={engine!r}, exchange={model!r} instead "
                f"(bit-identical)")
        if value not in ("hf", "slater", "gu", "tfda", "gsz"):
            raise ValueError(
                f"exchange must be one of 'hf', 'slater', 'gu' "
                f"(or the B-spline-only 'tfda', 'gsz'); got {value!r}")
        return Exchange(model=value)
    raise TypeError(
        f"exchange must be a model name or an Exchange instance "
        f"(got {type(exchange).__name__}); construct it with "
        f"Exchange(model='slater', alpha_x=1.0) — see help(atomx.Exchange)")


def _coerce_correlation(correlation):
    """Normalise the ``correlation=`` argument to a :class:`Correlation`."""
    if correlation is None:
        return Correlation()
    value = getattr(correlation, "value", correlation)  # types.Correlation
    if isinstance(value, Correlation):
        return value
    if isinstance(value, str):
        return Correlation(expansion=value)
    raise TypeError(
        f"correlation must be None, 's', 'sd', or a Correlation instance "
        f"(got {type(correlation).__name__}); construct it with "
        f"Correlation(expansion='sd', averaging='statistical') — see "
        f"help(atomx.Correlation)")


def _resolve_engine_exchange(engine, exchange: Exchange):
    """Resolve the user-facing ``engine`` × ``exchange`` pair.

    ``engine=None`` defaults to the shooting engine for every exchange
    model it supports (hf/slater/gu — it is both the fastest engine and an
    independent cross-check of the B-spline results); the parametric
    B-spline-only potentials (tfda/gsz) resolve to bspline.
    Multi-configuration input overrides to bspline in ``__init__`` — CI
    needs the B-spline MCDHF machinery.
    """
    if engine is not None and engine not in _ENGINES:
        raise ValueError(
            f"engine must be None, 'bspline', or 'shooting' (got {engine!r})")
    if engine is None:
        engine = "bspline" if exchange.is_parametric else "shooting"
    if engine == "shooting" and exchange.is_parametric:
        raise ValueError(
            f"exchange={exchange.model!r} is a parametric B-spline "
            f"potential; it requires engine='bspline'")
    return engine


class Atom:
    """High-level interface for atomic structure calculations.

    Ten arguments, one per physical question.  Five of them accept either a
    plain value (the common case) or a self-describing object from
    :mod:`atomx.atom_inputs`::

        from atomx import Atom, Basis, Corrections, Correlation, Environment

        ne = Atom(10)                          # neutral Ne, DHF + Breit + QED
        ne.solve()

        # The same atom, everything spelled out:
        ne = Atom(
            10,
            config="1s2 2s2 2p6",
            charge=0,
            nuclear="fermi",
            exchange="hf",
            corrections="breit+qed",
            correlation=None,
            engine="bspline",
            basis=Basis(ns=200, ks=9, r_max=90.0),
            environment=Environment(plasma_ne=1e22, plasma_Te=100.0),
        )

    Use :attr:`resolved_inputs` to see what the constructor actually chose
    (the engine behind ``engine=None``, the knot grid behind ``ns=None``,
    the corrections behind a stack, the SCF/CI protocol).

    Parameters
    ----------
    Z : int
        Nuclear charge (proton number, 1-118).  Required first positional
        argument: it is the atom's identity and sets the nuclear potential
        ``-Z alpha / r``, the hydrogenic energy scale, and — through the
        electron number ``N = Z - charge`` — the default configuration.
    config : str or list of str, optional
        One non-relativistic configuration (``"1s2 2s2 2p6"``) or an
        explicit multi-configuration list for the MCDHF/CI machinery
        (``["1s2 2s2", "1s2 2p2"]``).  Default ``None`` takes the built-in
        ground state of ``Z - charge`` electrons
        (:func:`atomx.config.ground_state_config`).  A *list* selects the
        multi-configuration path: it forces the B-spline MCDHF engine (an
        explicit ``engine="shooting"`` raises) and forces
        ``exchange="hf"``.  The single string is also the reference
        configuration that ``correlation=Correlation(expansion=...)``
        expands.
    charge : int, default 0
        Net ion charge in units of the elementary charge: 0 = neutral,
        1 = singly ionized (one electron removed), -1 = singly charged
        anion.  Fixes ``N = Z - charge``, hence the default configuration
        and every occupation sum downstream.
    nuclear : {"fermi", "point", "uniform"}, default "fermi"
        Nuclear charge distribution used to build the potential.

        - ``"fermi"`` — two-parameter Fermi distribution.  The DBSR_HF
          standard; it also regularises the origin for the B-spline basis
          (a point nucleus needs the ``r^gamma`` Dirac cusp, which
          B-splines cannot represent).
        - ``"point"`` — point nucleus (exact Dirac-Coulomb).  Use it for
          hydrogen-like benchmarks; otherwise the finite-size shift is
          missing.
        - ``"uniform"`` — uniformly charged sphere.  B-spline engine only:
          the shooting engine raises rather than silently solving a
          different nuclear model.
    exchange : {"hf", "slater", "gu", "tfda", "gsz"} or Exchange, default "hf"
        Exchange model (see :class:`atomx.atom_inputs.Exchange`):

        - ``"hf"``     — non-local Hartree-Fock exchange (default).
        - ``"slater"`` — local Slater rho^(1/3) exchange; its coefficient
          ``alpha_x`` lives on the :class:`~atomx.atom_inputs.Exchange`
          object, since only this model has one.
        - ``"gu"``     — Gu's empirical Z- and N-dependent alpha_xc(Z, N)
          (FAC's local-exchange model; Gu 2008).
        - ``"tfda"``/``"gsz"`` — parametric potentials; B-spline only.

        Multi-configuration runs support ``"hf"`` only.  The numerical
        backend of the local-exchange Slater integrals
        (``slater_method``) is a field of the same object because it is
        meaningful only for those models.
    corrections : None, str, or Corrections, default None
        Post-SCF relativistic corrections (see
        :class:`atomx.atom_inputs.Corrections`).  ``None`` applies both
        (Breit + QED).  The presets ``"none"``, ``"breit"``, ``"qed"`` and
        ``"breit+qed"`` cover the common cases; ``Corrections(...)``
        carries the self-energy backend and an explicit
        ``CorrectionStack``.  ``Corrections(qed_backend=...)`` reaches
        every engine, including the multi-configuration MCDHF path.
    correlation : None, {"s", "sd"}, or Correlation, default None
        Electron correlation: which CSFs enter, and the CI/SCF protocol
        (see :class:`atomx.atom_inputs.Correlation`).  ``"s"``/``"sd"``
        auto-generate singles / singles+doubles excitations from the
        reference configuration; the object also carries ``averaging``
        (how the jj-CSFs of a partially filled shell are weighted in the
        Hamiltonian state or fixed configuration ensemble), ``radial_optimization``,
        ``j_resolved_csfs`` and ``per_block_orbital_sets`` — the
        multi-configuration protocol, which is only meaningful when a
        multi-configuration expansion exists.

        Note this is a different axis from ``config=[...]``: the list says
        *which* configurations, this says *how* the expansion is solved.
    engine : {"bspline", "shooting"}, optional
        Radial ENGINE, decoupled from the exchange MODEL:

        - ``"bspline"``  — Galerkin B-spline :class:`DFSolver`.
        - ``"shooting"`` — ODE shooting :class:`ShootingSolver`
          (Salvat/Radial method).
        - ``None`` (default) — resolves to ``"shooting"`` for every model
          it supports (``"hf"``/``"slater"``/``"gu"``); the B-spline-only
          parametric potentials (``"tfda"``/``"gsz"``) resolve to
          ``"bspline"``.  Multi-configuration input (``config=[...]`` or
          ``correlation="s"/"sd"``) yields to ``"bspline"`` (MCDHF);
          combining it with an EXPLICIT ``engine="shooting"`` raises.

        Support matrix (engine x model)::

                        hf      slater   gu
            bspline     yes     yes      yes
            shooting    yes*    yes      yes

        (*) ``shooting`` + ``"hf"`` solves the non-local Dirac-HF
        equations with the Froese-Fischer inhomogeneous-equation
        (variation-of-parameters) method
        (:class:`atomx.numerics.shooting.hartree_fock.ShootingHFSolver`);
        validated against the B-spline DHF to <= 1e-4 relative on total
        energies (He, Ne, Fe XVII).
    basis : Basis, optional
        Radial basis and discretisation
        (:class:`atomx.atom_inputs.Basis`): ``ns``/``ks`` (B-spline count
        and order), ``r_max`` (radial box), ``grid_hi``/``grid_he``/
        ``grid_hmax`` (DBSR_HF knot-shape overrides), ``starting_potential``
        (SCF initial guess — a convergence aid with no effect on a
        converged solution) and ``sampling`` (a named accuracy tier that
        presets those fields plus the shooting grid and
        :meth:`solve`'s ``tol``/``max_iter``).  ``None`` (default) uses
        the DBSR_HF defaults for every one of them.
    environment : Environment, optional
        The surrounding plasma
        (:class:`atomx.atom_inputs.Environment`): ``plasma_ne`` (cm^-3)
        and ``plasma_Te`` (eV) define the screening length
        ``lambda_D = sqrt(kT / (4 pi n_e))``, and the screening potential
        is added to the nuclear potential inside the SCF — it raises the
        continuum and lowers the ionisation energy.  Both are required
        together, and the model (``"debye"`` or ``"stewart_pyatt"``) is a
        field of the same object because neither alone defines a screening
        length.

    Examples
    --------
    ::

        ne = Atom(10)                          # Ne, shooting engine, Breit+QED
        ne.solve()
        print(ne.total_energy)                 # -128.61 Hartree
        print(ne.resolved_inputs)              # engine/grid/Breit actually used

        # Correction-free B-spline run with a refined K-shell grid
        ne = Atom(10, config="1s2 2s2 2p6",
                  engine="bspline",
                  basis=Basis(grid_hi=0.02, grid_he=0.10),
                  corrections="none")
        ne.solve()

        # FAC-style local exchange; alpha_x only exists for slater
        fe17 = Atom(26, charge=16, config="1s2 2s2 2p6",
                    exchange=Exchange(model="slater", alpha_x=2/3))
        fe17.solve()

        # Strongly coupled plasma: Stewart-Pyatt screening of the nucleus
        dense = Atom(26, charge=16, config="1s2 2s2 2p6",
                     environment=Environment(plasma_ne=1e23, plasma_Te=50.0,
                                             screening="stewart_pyatt"))
        dense.solve()
    """

    def __init__(self, Z, config=None, *, charge=0, nuclear="fermi",
                 exchange="hf", corrections=None, correlation=None,
                 engine=None, basis=None, environment=None, **retired):
        # Every parameter is documented in the class docstring; the retired
        # spellings are rejected with a message naming their replacement
        # (this is a clean break — there are no aliases).
        _reject_retired(retired)

        # --- Input objects -------------------------------------------
        # Each grouped argument is a plain value object: it validates
        # itself, and Atom resolves it into the flat state the solvers
        # consume.  Nothing here is merged from a second spelling.
        if basis is None:
            basis = Basis()
        elif not isinstance(basis, Basis):
            raise TypeError(
                f"basis must be a Basis instance (got {type(basis).__name__}); "
                f"construct it with Basis(ns=200, r_max=90.0) — see "
                f"help(atomx.Basis)")
        if environment is None:
            environment = Environment()
        elif not isinstance(environment, Environment):
            raise TypeError(
                f"environment must be an Environment instance (got "
                f"{type(environment).__name__}); construct it with "
                f"Environment(plasma_ne=1e22, plasma_Te=100.0) — see "
                f"help(atomx.Environment)")
        if corrections is not None and not isinstance(corrections,
                                                      (str, Corrections)):
            raise TypeError(
                f"corrections must be None, a preset string, or a "
                f"Corrections instance (got {type(corrections).__name__}); "
                f"see help(atomx.Corrections)")
        exchange_spec = _coerce_exchange(exchange)
        correlation_spec = _coerce_correlation(correlation)

        # ``sampling`` is a preset tier, not a second spelling: it fills in
        # the basis fields the caller left open (ns/ks/r_max) and the
        # shooting grid, and Atom.solve() takes tol/max_iter from it too.
        # An explicit Basis field always wins.
        sampling = basis.sampling
        self._sampling = sampling
        ns = basis.ns
        ks = basis.ks
        r_max = basis.r_max
        self.shooting_space = basis.shooting_space
        starting_potential = basis.starting_potential
        if sampling is not None:
            if ns is None:
                ns = sampling.ns
            if ks == 9:  # Basis default — only override if caller didn't ask
                ks = sampling.ks
            if starting_potential is None:
                starting_potential = sampling.starting_potential
            # ``Sampling`` carries the radial box and the shooting grid
            # size as well; both reach the solvers now.  A tier field left
            # at its schema default counts as "not asked for", so
            # ``Sampling.standard`` keeps the engine's own tuned defaults
            # (auto box; n_grid=1000 for HF / 400 for local exchange).
            if r_max is None and sampling.r_max != _TIER_DEFAULTS["r_max"]:
                r_max = float(sampling.r_max)
        self._n_grid = None
        if sampling is not None and sampling.n_grid != _TIER_DEFAULTS["n_grid"]:
            self._n_grid = int(sampling.n_grid)

        # --- Validation ----------------------------------------------
        try:
            Z_int = int(Z)
        except (TypeError, ValueError):
            raise ValueError(f"Z must be an integer (got {Z!r})") from None
        if Z_int != Z:
            raise ValueError(f"Z must be an integer (got {Z!r})")
        if Z_int < 1:
            raise ValueError(f"Z must be >= 1 (got {Z!r})")
        Z = Z_int
        if nuclear not in ("fermi", "point", "uniform"):
            raise ValueError(
                f"nuclear must be 'fermi', 'point', or 'uniform' "
                f"(got {nuclear!r})")
        if config is not None and not isinstance(config, (str, list, tuple)):
            raise TypeError(
                f"config must be a configuration string or a list of "
                f"strings (got {type(config).__name__})")
        if isinstance(config, (list, tuple)):
            config = list(config)
            if not config or not all(isinstance(c, str) for c in config):
                raise TypeError(
                    f"config must be a non-empty list of configuration "
                    f"strings (got {config!r})")
        if (isinstance(config, list)
                and correlation_spec.expansion is not None):
            raise ValueError(
                f"config={config!r} already lists the configurations and "
                f"correlation=Correlation(expansion="
                f"{correlation_spec.expansion!r}) would generate another "
                f"expansion from the reference configuration.  Pass either "
                f"the explicit list (config=[...], correlation=Correlation("
                f"averaging=...)) or the expansion "
                f"(config=\"<reference>\", correlation="
                f"{correlation_spec.expansion!r}).")

        # Hamiltonian DF mixes only the CSFs of one NR configuration.
        # An explicit configuration expansion instead selects MCDHF.
        self._averaging_explicit = correlation_spec.averaging is not None
        correlated = isinstance(config, list) or correlation_spec.expansion is not None
        default_average = ("ci" if correlated else
                           "hamiltonian" if exchange_spec.model == "hf" else "statistical")
        averaging = correlation_spec.averaging or default_average

        # Resolved value (what the shooting engine would use); the
        # explicit/None distinction is what the B-spline rejection
        # below keys on.
        self.slater_method = exchange_spec.slater_method or "slag6"
        self.Z = Z
        self.charge = int(charge)
        self.nuclear = nuclear
        # Unified physics-based grid for all Z
        self._use_physics_grid = (ns is None)
        if ns is None:
            ns = 113  # will be overridden by physics grid
        self.ns = ns
        self.ks = ks
        #: Radial box (Bohr) for both engines.  ``None`` picks a value that
        #: contains every requested orbital: the shooting default of 75
        #: only resolves states up to ``n <~ sqrt(32.6 Z)``, so an
        #: explicitly requested Rydberg state used to be silently clamped
        #: to the bracketing ceiling (``Atom(Z=1, config="6s1")`` returned
        #: ``-Z/r_max`` with a 4 % error).  See
        #: docs/archive/2026-09/PHYSICS_NUMERICS_REVIEW_2026-09-19.md finding C4.
        self.r_max = r_max
        #: DBSR_HF grid-shape overrides for the automatic B-spline knot
        #: sequence (``numerics/bspline/knots.py::make_physics_knots``):
        #: ``hi`` sets the innermost step (``hs = hi/Z`` for a point
        #: nucleus), ``he`` the exponential growth factor and ``hmax`` the
        #: outer constant step.  ``None`` keeps the DBSR_HF defaults
        #: (0.25 / 0.25 / 1.0).  These exist because the default cannot
        #: resolve a high-Z K shell: the hydrogen-like U^91+ 1s eigenvalue
        #: is +97.7 eV off at the default grid, +0.8 eV at ns=400, and is
        #: only brought under 0.3 eV by raising the B-spline order or
        #: refining ``hi``.  See
        #: docs/archive/2026-09/PHYSICS_NUMERICS_REVIEW_2026-09-19.md finding C3.
        self.grid_hi = basis.grid_hi
        self.grid_he = basis.grid_he
        self.grid_hmax = basis.grid_hmax
        resolved = resolve_corrections(corrections)
        self.breit = resolved.breit
        self.qed = resolved.qed
        self.qed_backend = resolved.qed_backend
        self.correction_stack = resolved.stack
        if self.correction_stack is None and resolved.breit_frequency != 'static':
            from atomx.corrections.qed import resolve_se_method
            from atomx.corrections.stack import CorrectionStack
            self.correction_stack = CorrectionStack(
                se_backend=resolve_se_method(self.qed_backend, self.Z) if self.qed else None,
                include_vp=self.qed, include_breit=self.breit,
                breit_frequency=resolved.breit_frequency,
                breit_orbital_energies=resolved.breit_orbital_energies)
        self.exchange = exchange_spec.model
        # Decoupled ENGINE x MODEL pair.
        self._engine_explicit = engine is not None
        self.engine = _resolve_engine_exchange(engine, exchange_spec)
        self.exchange_model = exchange_spec.model
        # ``alpha_x`` selects exchange PHYSICS, so its default must not
        # depend on the numerical engine.  The low-level DFSolver retains
        # its historical 2/3 default for direct callers, but Atom always
        # passes the canonical Slater value explicitly.
        self.alpha_x = (1.5 if exchange_spec.alpha_x is None
                        and self.exchange_model == "slater"
                        else exchange_spec.alpha_x)
        # The B-spline Galerkin path evaluates its local-exchange Slater
        # integrals with its own quadrature; ``slater_method`` selects the
        # shooting backend.  Accepting it there would be a knob that does
        # nothing, so it raises.
        if exchange_spec.slater_method is not None and self.engine == "bspline":
            raise ValueError(
                f"slater_method={self.slater_method!r} selects the numerical "
                f"backend of the SHOOTING engine's local-exchange Slater "
                f"integrals; the B-spline Galerkin path evaluates them with "
                f"its own quadrature and would ignore it.  Drop "
                f"slater_method (the default 'slag6' is used where it "
                f"applies) or pass engine='shooting'.")
        self.correlation = correlation_spec.expansion
        self.starting_potential = starting_potential
        self.plasma_ne = environment.plasma_ne
        self.plasma_Te = environment.plasma_Te
        self.screening_model = environment.screening
        # Density and temperature are two halves of one screening length,
        # lambda_D = sqrt(kT / (4 pi n_e)); either one alone is meaningless
        # (the check lives on Environment, which is where the pair is
        # declared).
        # The ion-sphere radius R_IS = (3 Z_ion / 4 pi n_e)^(1/3) needs a
        # positive net charge: with Z_ion = 0 it diverges and the model
        # silently degenerates to an unscreened nucleus.
        if (self.plasma_ne is not None
                and environment.screening == "stewart_pyatt"
                and self.charge <= 0):
            raise ValueError(
                f"environment=Environment(screening='stewart_pyatt') "
                f"needs a positive net charge to define the ion-sphere "
                f"radius R_IS = (3 Z_ion / (4 pi n_e))^(1/3), but "
                f"charge={self.charge}.  Use the default "
                f"screening='debye' (defined for every charge state), "
                f"or pass charge>0.")
        self.averaging = averaging
        if (self._averaging_explicit and self.averaging in ("ci", "hamiltonian")
                and self.exchange_model != "hf"):
            raise ValueError(
                f"averaging={self.averaging!r} requires exchange='hf'; "
                f"exchange={self.exchange_model!r} is a local-potential model.")
        self.j_resolved_csfs = bool(correlation_spec.j_resolved_csfs)
        self.per_block_orbital_sets = bool(
            correlation_spec.per_block_orbital_sets)
        self.radial_optimization = correlation_spec.radial_optimization
        self.orbital_preconditioner = correlation_spec.orbital_preconditioner
        self.max_determinants = correlation_spec.max_determinants
        self.state_average = correlation_spec.state_average
        # Both flags are read only by the multi-configuration branch of
        # solve(); with a single configuration they have nothing to act on
        # and used to be dropped in silence.
        if config is None and correlation_spec.expansion is None:
            if self.j_resolved_csfs:
                raise ValueError(
                    "j_resolved_csfs=True requires a multi-configuration "
                    "input: it selects how the jj-CSFs of a "
                    "config=[...] (or correlation='sd') expansion are "
                    "enumerated, and a single-configuration run has only "
                    "the CSFs of that one configuration.  Pass "
                    "config=[...] (or correlation='sd'), or drop "
                    "j_resolved_csfs=True.")
            if self.per_block_orbital_sets:
                raise ValueError(
                    "per_block_orbital_sets=True requires a "
                    "multi-configuration input: it builds one orbital set "
                    "per (J, parity) block of the CI expansion, which does "
                    "not exist for a single configuration.  Pass "
                    "config=[...] (or correlation='sd'), or drop "
                    "per_block_orbital_sets=True.")

        # Determine configuration(s)
        if isinstance(config, list):
            self._multi_config = True
            self._config_strings = list(config)
        elif correlation_spec.expansion is not None:
            # Auto-generate correlation CSFs
            self._multi_config = True
            N_electrons = Z - self.charge
            try:
                base_config = config or ground_state_config(N_electrons)
            except ValueError:
                raise ValueError(
                    f"No built-in config for {N_electrons} electrons.") from None
            from atomx.methods.mcdhf.excitations import (
                generate_sd_excitations,
                suggest_virtual_orbitals,
            )
            virtuals = suggest_virtual_orbitals(Z, n_virtual=1)
            max_exc = 2 if correlation_spec.expansion == "sd" else 1
            # Freeze core for Z > 10
            core = None
            if Z > 10:
                core = ["1s", "2s", "2p"]
            elif Z > 4:
                core = ["1s"]
            self._config_strings = generate_sd_excitations(
                base_config, virtuals, core_orbitals=core,
                max_excitations=max_exc)
        else:
            self._multi_config = False
            if config is None:
                N_electrons = Z - self.charge
                config = ground_state_config(N_electrons)
            self._config_strings = [config]

        if not self._multi_config and self.averaging == "ci":
            raise ValueError(
                "averaging='ci' no longer selects MCDHF for a single configuration. "
                "Use Correlation(averaging='hamiltonian') for intermediate-coupling "
                "DHF, or pass config=[...] / a correlation expansion for MCDHF.")
        if self._multi_config and self.averaging != "ci":
            raise ValueError("configuration lists/expansions require averaging='ci'; "
                             "averaging='hamiltonian' is for one configuration")

        # Central config-string validation (HYP-1778): reject malformed
        # shell tokens ("2x2", trailing junk), n = 0 and over-occupied
        # shells (q > 2(2l+1)) here at the entry with a clear ValueError
        # naming the token, instead of letting them surface one solver
        # frame deep.  The shared parser also accepts what the solvers
        # accept (implicit unit occupations, s..i letters), so this
        # cannot reject anything that would have solved.
        from atomx.config.parsing import validate_config_string
        for cfg in self._config_strings:
            validate_config_string(cfg)

        # Preserve the established automatic choice; explicit shooting now
        # runs the full CI-gradient optimizer in its declared electron space.
        if self._multi_config and self.engine == 'shooting' and not self._engine_explicit:
            self.engine = 'bspline'
        if self.shooting_space is not None and not (
                self._multi_config and self.engine == 'shooting'):
            raise ValueError('shooting_space requires explicit MCDHF with the shooting engine; use Sampling.n_grid for DF')

        if self.orbital_preconditioner == "shooting" and (
                not self._multi_config or self.engine != "bspline"):
            raise ValueError("shooting orbital preconditioning requires explicit multi-configuration MCDHF with the B-spline engine")

        # The grid-shape overrides exist only for the automatic B-spline
        # knot sequence (numerics/bspline/knots.py::make_physics_knots).
        # Name the engine that ignores them, or the explicit ``ns`` that
        # replaces the grid they would shape, instead of dropping them in
        # silence.  Resolved AFTER the multi-configuration yield above, so
        # a multi-configuration run that legitimately moves to bspline
        # still accepts them.
        grid_given = [(name, value) for name, value in
                      (("grid_hi", self.grid_hi), ("grid_he", self.grid_he),
                       ("grid_hmax", self.grid_hmax))
                      if value is not None]
        if grid_given and self.engine != "bspline":
            names = ", ".join(name for name, _ in grid_given)
            first_name, first_value = grid_given[0]
            which = f" (with {names})" if len(grid_given) > 1 else ""
            why = ("" if self._engine_explicit else
                   f" (engine=None resolves to the shooting engine for "
                   f"exchange={self.exchange_model!r})")
            raise ValueError(
                f"{first_name}={first_value!r}{which} shapes the "
                f"automatic B-spline knot sequence, but engine resolved "
                f"to {self.engine!r}{why}; the shooting engine "
                f"integrates on its own adaptive radial grid and ignores "
                f"{names}.  Pass engine='bspline' or drop {names}.")
        if grid_given and not self._use_physics_grid:
            names = ", ".join(name for name, _ in grid_given)
            first_name, first_value = grid_given[0]
            source = ("the sampling tier" if (
                self._sampling is not None
                and getattr(self._sampling, "ns", None) == self.ns)
                else "an explicit ns")
            raise ValueError(
                f"{first_name}={first_value!r} shapes the automatic "
                f"DBSR_HF knot sequence, which {source} (ns={self.ns}) "
                f"replaces: the grid then follows from ns/ks/r_max and "
                f"{names} is never read.  Drop ns={self.ns} — recommended, "
                f"it keeps the DBSR_HF grid with your overrides — or drop "
                f"{names}.")

        # Result (populated after solve()).  ``self._result`` may be
        # any :class:`atomx.results.AtomicResult` flavor (SCFResult,
        # MCDHFResult, ShootingResult, …).  All accessors below
        # delegate to it uniformly.
        self._result = None


    def _resolve_r_max(self, subshells) -> float:
        """Radial box for the requested subshells (Bohr).

        Explicit ``r_max`` wins.  Otherwise keep the historical 75 Bohr
        unless a requested orbital is too diffuse to fit: the classical
        extent is ``<r> ~ 1.5 n^2 / Z`` and shooting resolves a state
        only while ``|E| >~ 1.15 Z/r_max``, so ``5 n_max^2 / Z`` is a
        safe container (a 6s orbital of hydrogen needs ~180 Bohr).
        """
        if self.r_max is not None:
            return float(self.r_max)
        n_max = max((getattr(sh, "n", 1) for sh in subshells), default=1)
        return max(75.0, 5.0 * n_max ** 2 / max(self.Z, 1))

    def _screening_shift(self):
        """Additive plasma screening ``delta_V(r)`` for the SCF, or ``None``.

        ``delta_V(r) = V_screened(r) - (-Z/r)``: the difference between
        the plasma model's central potential and the bare nuclear one, so
        the finite nuclear size is untouched and the shift vanishes at the
        origin (it tends to ``Z/lambda`` at large r — the continuum shift
        that lowers the ionisation energy).  It is built by calling
        :func:`atomx.plasma.screening.screening_shift` directly, so the
        SCF and a hand evaluation of the same model agree by construction.
        """
        if self.plasma_ne is None:
            return None
        from atomx.plasma.screening import screening_shift
        Z = self.Z
        ne_cm3 = float(self.plasma_ne)
        Te_eV = float(self.plasma_Te)
        model = self.screening_model
        z_ion = int(self.charge)

        def shift(r):
            return screening_shift(Z, r, ne_cm3, Te_eV, model=model,
                                   z_ion=z_ion)

        return shift

    def _shooting_n_grid(self, default: int) -> int:
        """Radial grid size for the shooting engine.

        ``default`` is the engine's own tuned value (1000 for the HF
        Green-function quadratures, 400 for the local-exchange SCF); an
        accuracy tier that *sets* ``n_grid`` — e.g.
        ``Basis(sampling=Sampling(n_grid=2000))`` — wins.
        """
        return int(self._n_grid) if self._n_grid else int(default)

    def solve(self, method=None, tol=1e-5, max_iter=40, log=None,
              sampling=None, *, orbital_tolerance=None):
        """Solve the requested atomic state, optionally checking radial refinement.

        ``Basis(sampling=Sampling(radial_refinement=RadialRefinement(...)))``
        enables grid and automatic-box refinement for single-configuration
        shooting. Explicit boxes stay fixed. See ``result.radial_convergence``
        for measured changes and limits. Refinement rebuilds the full state and
        corrections on each grid; it does not assert physical-model accuracy.

        Parameters
        ----------
        method : Method, optional
            Explicit theory-level method. Returns its AtomicResult; otherwise
            dispatches from the Atom inputs and returns self for chaining.
        tol : float
            SCF tolerance. Its default is replaced by the sampling tolerance.
            Adaptive refinement may tighten it further, as recorded in the
            calculation specification.
        max_iter : int
            SCF iteration cap; the sampling cap replaces the default value.
        log : str, bool or None
            Calculation log path; True selects the default filename.
        sampling : Sampling, optional
            Override solve controls (tolerance, iteration cap and refinement
            policy). Discretization sizes are set through Basis at construction.
        orbital_tolerance : float, optional
            Independent phase-aligned L2 orbital-update threshold for iterative
            DF. Overrides Sampling.orbital_tolerance. It controls SCF updates,
            not radial discretization or physical-model error. Unsupported
            methods reject this request rather than reinterpret it.
        """
        selected = sampling if sampling is not None else self._sampling
        if orbital_tolerance is None and selected is not None:
            orbital_tolerance = selected.orbital_tolerance
        if orbital_tolerance is not None and (
                not np.isfinite(orbital_tolerance) or orbital_tolerance <= 0):
            raise ValueError('orbital_tolerance must be finite and positive')
        policy = selected.radial_refinement if selected is not None else None
        if policy is None:
            return self._solve_fixed(method, tol, max_iter, log, sampling,
                                     orbital_tolerance=orbital_tolerance)
        if method is not None or self.engine != 'shooting' or self._multi_config:
            raise ValueError('radial_refinement requires single-configuration shooting via Atom.solve()')
        from dataclasses import replace

        from atomx.config.csf import _parse_config_string
        from atomx.numerics.shooting.adaptive import RadialSample, refine_radial_grid
        from atomx.radial_state import native_radial_state

        requested_box, requested_points = self.r_max, self._n_grid
        shells = _parse_config_string(self._config_strings[0])
        initial_box = self._resolve_r_max(shells)
        initial_points = self._shooting_n_grid(1000 if self.exchange_model == 'hf' else 400)
        resolved_tol = selected.tol if tol == 1e-5 else tol
        resolved_tol = min(resolved_tol, policy.scf_tol)

        def sample(n_grid, r_max):
            self._n_grid, self.r_max = n_grid, r_max
            self._solve_fixed(None, resolved_tol, max_iter, log, selected,
                              orbital_tolerance=orbital_tolerance)
            result = self._result
            state = native_radial_state(result)
            densities = np.asarray([p*p+q*q for p, q in state.orbitals.values()])
            energies = np.asarray(getattr(result, 'state_energies', []), dtype=float)
            excitation = energies-energies.min() if energies.size else energies
            return RadialSample(result, float(result.total_energy),
                dict(result.orbital_energies), state.metric.radius,
                state.metric.weights, densities, result.converged, excitation)

        try:
            final, audit = refine_radial_grid(sample, n_grid=initial_points,
                r_max=initial_box, policy=policy, vary_box=requested_box is None)
        except Exception:
            self._result = None
            raise
        finally:
            self.r_max, self._n_grid = requested_box, requested_points
        self._result = final.value
        self._result.radial_convergence = audit
        self._result.converged = bool(final.converged and audit['converged'])
        if "_variational_diagnostics" in self._result.solver_diagnostics:
            self._result.solver_diagnostics["_variational_diagnostics"].update(
                radial_space_converged=audit['radial_space_converged'],
                radial_space_convergence_reason=audit['termination_reason'])
        spec = self._result.calculation_spec
        self._result.calculation_spec = replace(spec, numerics={**spec.numerics,
            'radial_refinement': policy, 'refinement_scf_tolerance': resolved_tol})
        return self

    def _solve_fixed(self, method=None, tol=1e-5, max_iter=40, log=None,
                     sampling=None, *, orbital_tolerance=None):
        """Dispatch one complete calculation at the resolved discretization."""
        # ``sampling`` falls back to whatever was attached to the atom at
        # construction time, so ``Atom(Z=10, sampling=Sampling.high).solve()``
        # picks up the high tolerance without re-passing it.
        sampling = sampling if sampling is not None else self._sampling
        if sampling is not None:
            if tol == 1e-5:  # default — let sampling drive it
                tol = sampling.tol
            if max_iter == 40:
                max_iter = sampling.max_iter

        if method is not None:
            return self._solve_method(method, tol, max_iter, log=log,
                                      orbital_tolerance=orbital_tolerance)

        orbital_controls = {} if orbital_tolerance is None else dict(
            orbital_tolerance=orbital_tolerance)
        if orbital_controls:
            if self._multi_config:
                raise ValueError('orbital_tolerance is an L2 DF update threshold; '
                                 'MCDHF uses variational gradient convergence')
            if self.engine == 'shooting' and self.exchange_model != 'hf':
                raise ValueError('orbital_tolerance is not implemented for local-exchange shooting')
            if self.exchange_model in ('tfda', 'gsz'):
                raise ValueError('orbital_tolerance requires an iterative mean-field solve')

        # Hamiltonian intermediate coupling needs the full CSF functional.
        # With one CSF the ordinary DHF equations are already equivalent.
        hamiltonian_csfs = None
        if not self._multi_config and self.averaging == "hamiltonian":
            from atomx.config.csf import enumerate_jj_csfs
            hamiltonian_csfs = enumerate_jj_csfs(self._config_strings[0])
        use_hamiltonian_df = hamiltonian_csfs is not None and len(hamiltonian_csfs) > 1
        if self.averaging == 'hamiltonian':
            if self.radial_optimization != 'mcdhf' or self.per_block_orbital_sets:
                raise ValueError('Hamiltonian DF requires variational ground-state orbitals; '
                                 'frozen/per-block protocols require an explicit MCDHF request')
            if not use_hamiltonian_df and self.shooting_space is not None:
                raise ValueError('a single-CSF DHF calculation uses native shooting; '
                                 'shooting_space is only used by variational CSF calculations')
        if (self._multi_config or self.correlation) and (
                self.exchange_model not in ("hf",)):
            raise ValueError(
                f"exchange={self.exchange_model!r} is not implemented for "
                f"multi-configuration runs (config=[...] or correlation=...); "
                f"the MCDHF/CI solver uses non-local Hartree-Fock exchange "
                f"only.  Use exchange='hf', or run the single-configuration "
                f"path (config=...) which supports the local models.  "
                f"Previously the argument was ignored while `provenance` "
                f"reported it."
            )

        if self.state_average is not None and not self._multi_config:
            raise ValueError('CI state averaging requires config=[...] or a correlation expansion')
        if self._multi_config or use_hamiltonian_df:
            from atomx.config.csf import (
                enumerate_jj_csfs,
                enumerate_jj_csfs_many,
            )
            if use_hamiltonian_df:
                from atomx.methods.hamiltonian_df import HamiltonianDFSolver
                solver_type = HamiltonianDFSolver
                state_kwargs = dict(configuration=self._config_strings[0],
                                    n_grid=self._shooting_n_grid(1000))
            elif self.j_resolved_csfs:
                from atomx.methods.mcdhf.solver import MCDHFSolver
                solver_type = MCDHFSolver
                csfs = []
                for cfg in self._config_strings:
                    csfs.extend(enumerate_jj_csfs(cfg))
                state_kwargs = dict(csfs=csfs)
            else:
                from atomx.methods.mcdhf.solver import MCDHFSolver
                solver_type = MCDHFSolver
                csfs = enumerate_jj_csfs_many(self._config_strings)
                state_kwargs = dict(csfs=csfs)
            solver = solver_type(
                Z=self.Z, **state_kwargs, ns=self.ns, ks=self.ks,
                r_max=self.r_max, grid_hi=self.grid_hi,
                grid_he=self.grid_he, grid_hmax=self.grid_hmax,
                include_breit=self.breit, include_qed=self.qed,
                qed_se_method=self.qed_backend,
                correction_stack=self.correction_stack,
                nuclear=self.nuclear,
                use_physics_grid=self._use_physics_grid,
                starting_potential=self.starting_potential,
                per_block_orbital_sets=self.per_block_orbital_sets,
                radial_optimization=self.radial_optimization,
                orbital_preconditioner=self.orbital_preconditioner,
                max_determinants=self.max_determinants,
                state_average=self.state_average,
                engine=self.engine,
                shooting_space=self.shooting_space,
                screening_potential=self._screening_shift(),
            )
            if self._multi_config and self.engine == 'shooting' and self._n_grid is not None:
                if self.shooting_space is not None and self._n_grid != self.shooting_space.n_grid:
                    raise ValueError('Sampling.n_grid and ShootingSpace.n_grid disagree')
                from dataclasses import replace
                solver.shooting_space = replace(solver.shooting_space,
                    n_grid=self._n_grid,
                    outer_intervals=min(solver.shooting_space.outer_intervals,
                                        self._n_grid * 2 // 3))
            self._result = solver.solve(max_iter=max_iter, scf_tol=tol, **orbital_controls)
        elif self.engine == "shooting":
            # ODE shooting engine — three exchange models:
            #   ``exchange='hf'`` → non-local Dirac-Hartree-Fock via the
            #     Froese-Fischer inhomogeneous-equation method
            #     (ShootingHFSolver; exchange enters as a Y^k source
            #     term, eigenvalues from a Green-function normalisation
            #     Newton).  Same jj-average angular coefficients as the
            #     B-spline DFSolver — agreement ≤ 1e-4 relative on He /
            #     Ne / Fe XVII totals (tests/test_shooting_hf.py).
            #   ``exchange='slater'`` → Salvat-style DHFS with Slater
            #     α = 3/2 by default, bit-identical to the Fortran
            #     ``radial/DHFS`` reference used by all atomx shooting
            #     benchmarks.
            #   ``exchange='gu'`` → Gu's FAC-style DFS with Z- and
            #     N-dependent α_xc(Z, N) from
            #     :mod:`atomx.physics.local_exchange`.
            # Both produce a ``ShootingResult`` (AtomicResult-compatible).
            #
            # The B-spline ``DFSolver`` also accepts Gu's model
            # (``engine='bspline', exchange='gu'``) and wires α_xc in
            # the same place the pre-existing ``'slater'`` path uses
            # ``alpha_x`` (see Task 4 in
            # ``docs/superpowers/plans/2026-04-20-fac-calculator.md``).
            # ``exchange='gu'`` with the default ``engine=None``
            # routes via shooting because the shooting path is already
            # validated against sfac to 10⁻⁴ relative across Z=2..92.
            from atomx.numerics.shooting.local import ShootingSolver
            # Open-shell subshells need statistical (spherically-
            # averaged) j-splitting so the density is physically
            # correct — a raw ``_parse_config_string`` fills the
            # lower-j subshell to capacity first (e.g. "2p2" → entire
            # 2p_{1/2}), which is wrong for local-exchange SCF.
            # ``generate_jj_average`` gives the correct statistical
            # split (2p_{1/2}^{2/3}, 2p_{3/2}^{4/3}) that matches
            # FAC's internal ``AVERAGE_CONFIG`` expansion.
            # n_grid=400 gives a ~2–3× speedup over the historical default
            # of 1000 with no loss of end-to-end accuracy on the Task 0
            # calibration panel (Z = 2..92): neutral atoms still match
            # Fortran DHFS.f to ≤ 3 × 10⁻⁷ relative (vs ≤ 2 × 10⁻⁸ at
            # n_grid=1000 — the residual is the adaptive dir_integrate
            # substep error, not the grid), and heavy-Z ions still match
            # sfac to ≤ 10⁻³.  Salvat's adaptive dir_integrate
            # automatically sub-divides each grid interval to hit the
            # EPS tolerance, so reducing the static grid just drops
            # redundant outer-region points.  Exposed via ``n_grid``
            # for power users who need the tighter neutral-atom bound.
            correction_jj_configs = None
            if self.exchange_model == "hf":
                from atomx.config.jj_coupling import JJConfig, generate_jj_average
                from atomx.numerics.shooting.hartree_fock import ShootingHFSolver
                subshells, correction_jj_configs = generate_jj_average(
                    self._config_strings[0],
                    averaging="equal" if self.averaging == "equal" else "statistical",
                )
                kept_indices = [i for i, sh in enumerate(subshells)
                                if sh.occupation > 1e-10]
                subshells = [subshells[i] for i in kept_indices]
                correction_jj_configs = [JJConfig(
                    subshells=[cfg.subshells[i] for i in kept_indices],
                    weight=cfg.weight) for cfg in correction_jj_configs]
                if self.averaging == "fractional":
                    correction_jj_configs = None
                # n_grid=1000 (not the local-exchange 400): the HF
                # Green-function quadratures ride on SLAG6 cumulative
                # integrals whose error IS the orbital error, so the
                # denser grid buys the cross-engine 1e-4 agreement.
                #
                # nuclear= is forwarded so both engines solve the SAME
                # nuclear model (Atom defaults to "fermi").  This
                # matters for the post-solve corrections: with a point
                # nucleus the B-spline basis cannot represent the r^γ
                # Dirac cusp (γ = sqrt(1 − (Zα)²) < 1) and its Uehling
                # VP drifts 3.8e-3 from the analytic value at Z = 26,
                # while the shooting power-series start is exact — the
                # Fermi nucleus regularises the origin for both.
                # "uniform" is B-spline-only; reject it rather than
                # silently solving a different model.
                if self.nuclear not in ("point", "fermi"):
                    raise ValueError(
                        f"engine='shooting' supports nuclear='point' or "
                        f"'fermi' (got {self.nuclear!r})")
                solver = ShootingHFSolver(
                    Z=self.Z, subshells=subshells,
                    n_grid=self._shooting_n_grid(1000),
                    r_max=self._resolve_r_max(subshells),
                    nuclear=self.nuclear,
                    jj_configs=correction_jj_configs,
                    screening_potential=self._screening_shift(),
                )
                # HF sweeps are Gauss-Seidel with 0.5 damping — give
                # them headroom beyond Atom's default max_iter=40.
                self._result = solver.solve(
                    max_iter=max(max_iter, 60), tol=tol, **orbital_controls)
                solver = None  # solved above; skip the local-model call
            elif self.exchange_model == "gu":
                from atomx.config.jj_coupling import JJConfig, generate_jj_average
                subshells, correction_jj_configs = generate_jj_average(
                    self._config_strings[0], averaging="statistical",
                )
                # Drop zero-occupation subshells (they only slow SCF)
                kept_indices = [i for i, sh in enumerate(subshells)
                                if sh.occupation > 1e-10]
                subshells = [subshells[i] for i in kept_indices]
                correction_jj_configs = [JJConfig(
                    subshells=[cfg.subshells[i] for i in kept_indices],
                    weight=cfg.weight) for cfg in correction_jj_configs]
                solver = ShootingSolver(
                    Z=self.Z, subshells=subshells,
                    n_grid=self._shooting_n_grid(400),
                    r_max=self._resolve_r_max(subshells),
                    exchange="gu", fac_ihx=-1.0, fac_hxs=1.0,
                    slater_method=self.slater_method or "slag6",
                    screening_potential=self._screening_shift(),
                )
            else:
                from atomx.config.csf import _parse_config_string
                subshells = _parse_config_string(self._config_strings[0])
                solver = ShootingSolver(
                    Z=self.Z, subshells=subshells,
                    n_grid=self._shooting_n_grid(400),
                    r_max=self._resolve_r_max(subshells),
                    alpha_x=self.alpha_x,
                    slater_method=self.slater_method or "slag6",
                    screening_potential=self._screening_shift(),
                )
            if solver is not None:
                self._result = solver.solve(max_iter=max_iter, tol=tol)

            # Post-solve corrections — same seam as DFSolver (review E1):
            # a user-supplied CorrectionStack wins; otherwise the
            # breit/qed booleans are sugar building the equivalent
            # stack.  The adapters evaluate directly on the shooting
            # grid orbitals (numerics.radial_workspace grid flavour), so
            # Breit/SE/VP here are the same physics as the B-spline
            # engine (cross-validated to <= 2e-3 in
            # tests/test_shooting_corrections.py).
            stack = self.correction_stack
            if stack is None and (self.breit or self.qed):
                from atomx.corrections.qed import resolve_se_method
                from atomx.corrections.stack import CorrectionStack
                stack = CorrectionStack(
                    se_backend=(resolve_se_method(self.qed_backend, self.Z)
                                if self.qed else None),
                    include_vp=bool(self.qed),
                    include_breit=bool(self.breit),
                )
            if stack is not None:
                from atomx.corrections.stack import finalize_with_corrections
                self._result._jj_configs = correction_jj_configs
                if (stack.include_breit and stack.breit_mode == "operator"
                        and stack.breit_frequency == 'static'):
                    # The explicit Gaunt mode computes its CI matrix
                    # element here; the DBSR_HF default needs no such
                    # additional contraction.
                    from atomx.physics.csf.breit_ci import (
                        attach_breit_operator_energy,
                    )
                    attach_breit_operator_energy(
                        self._result, jj_configs=correction_jj_configs)
                finalize_with_corrections(self._result, stack)
        else:
            # B-spline DHF/DFS backend with user-selected averaging
            # (Guerra/statistical | equal | fractional).
            # "fractional" skips jj_configs to use
            # the q*(q-1) fallback inside _get_WW.
            from atomx.config.jj_coupling import JJConfig, generate_jj_average
            avg_key = "equal" if self.averaging == "equal" else "statistical"
            subshells, self._jj_configs = generate_jj_average(
                self._config_strings[0], averaging=avg_key)
            # Remove zero-occupation subshells (they add orbitals but slow SCF)
            # Keep configs aligned: filter configs to match
            kept_indices = [i for i, sh in enumerate(subshells) if sh.occupation > 1e-10]
            subshells = [subshells[i] for i in kept_indices]
            filtered_configs = []
            for c in self._jj_configs:
                filtered_subs = [c.subshells[i] for i in kept_indices]
                filtered_configs.append(JJConfig(subshells=filtered_subs, weight=c.weight))
            jj_arg = None if self.averaging == "fractional" else filtered_configs
            df_kwargs = {}
            if self.alpha_x is not None:
                df_kwargs["alpha_x"] = self.alpha_x
            solver = DFSolver(Z=self.Z, subshells=subshells,
                              ns=self.ns, ks=self.ks,
                              exchange=self.exchange_model, **df_kwargs,
                              include_breit=self.breit,
                              include_qed=self.qed,
                              qed_se_method=self.qed_backend,
                              correction_stack=self.correction_stack,
                              nuclear=self.nuclear,
                              starting_potential=self.starting_potential or ('tfda' if self.Z >= 20 else None),
                              screening_potential=self._screening_shift(),
                              jj_configs=jj_arg)

            # DBSR_HF-style semi-exponential grid (matched to nuclear model)
            if self._use_physics_grid:
                solver.install_physics_grid(
                    r_max=self.r_max, hi=self.grid_hi, he=self.grid_he,
                    hmax=self.grid_hmax)

            # f-electron systems (lanthanides/actinides) need more iterations
            has_f = any(sh.l >= 3 for sh in subshells)
            effective_max_iter = max(max_iter, 120) if has_f else max_iter
            self._result = solver.solve(max_iter=effective_max_iter, tol=tol, **orbital_controls)
            # SCFResult retains its own _bsp_data; nothing else to wire.

        # Provenance: record which (engine, model, averaging) triple
        # produced the numbers so results are inspectable after the
        # fact.  NOT mirrored into energy_components: consumers iterate
        # that dict as numbers (compare(), viz), and string entries
        # broke them — provenance lives on result.provenance /
        # Atom.provenance only.
        if self._result is not None:
            averaging_ran = "ci" if self._multi_config else self.averaging
            provenance = {"engine": self.engine,
                          "exchange": self.exchange_model,
                          "averaging": averaging_ran}
            if self.plasma_ne is not None:
                provenance["plasma"] = {
                    "plasma_ne": float(self.plasma_ne),
                    "plasma_Te": float(self.plasma_Te),
                    "screening": self.screening_model,
                }
            self._result.provenance = provenance
            spec = getattr(self._result, 'calculation_spec', None)
            if spec is not None:
                from dataclasses import replace
                method = ('correlated_ci' if self._multi_config else
                          'local_potential' if self.exchange_model != 'hf' else
                          'fractional_occupation' if self.averaging == 'fractional' else
                          'hamiltonian' if self.averaging == 'hamiltonian' else
                          'configuration_average')
                weights = ('variational_ci' if self._multi_config else
                           'hamiltonian_eigenvectors' if self.averaging == 'hamiltonian' else
                           'equal' if self.averaging == 'equal' else
                           'products_of_means' if self.averaging == 'fractional' else
                           'integer_configuration_degeneracy')
                self._result.calculation_spec = replace(spec, state={
                    **spec.state, 'averaging': dict(selection=averaging_ran,
                        method=method, weights=weights)})

        return self

    def _solve_method(self, method, tol, max_iter, log=None, *, orbital_tolerance=None):
        """Dispatch to a theory-level method object."""
        from atomx.methods.breit_pauli import BreitPauli
        from atomx.methods.dirac_fock import DiracFock
        from atomx.methods.hartree_fock_slater import HartreeFockSlater
        from atomx.methods.hydrogenic import HydrogenicDirac, HydrogenicSchrodinger
        from atomx.methods.mcdhf import MCDHF

        if orbital_tolerance is not None and not isinstance(method, DiracFock):
            raise ValueError('orbital_tolerance is an L2 DF update threshold; '
                             f'unsupported by {type(method).__name__}')
        orbital_controls = {} if orbital_tolerance is None else dict(
            orbital_tolerance=orbital_tolerance)

        # Set up logging
        calc_log = None
        if log:
            from atomx.log import CalculationLog
            if isinstance(log, str):
                log_path = log
            else:
                sym = _element_symbol(self.Z)
                log_path = f"{sym}_Z{self.Z}_{method.name}.log"
            calc_log = CalculationLog(log_path, verbose=False)
            calc_log.header(
                Z=self.Z, config=self._config_strings[0],
                method=method.name,
                nuclear=method.nuclear.value,
                theory_level=method.theory_level.name,
            )

        # Parse configuration into subshells
        from atomx.config.csf import _parse_config_string
        config_str = self._config_strings[0]
        subshells_raw = _parse_config_string(config_str)

        if isinstance(method, (HydrogenicSchrodinger, HydrogenicDirac)):
            return method.solve(self.Z, subshells_raw,
                                tol=tol, max_iter=max_iter)

        if isinstance(method, (HartreeFockSlater, BreitPauli)):
            return method.solve(self.Z, subshells_raw,
                                tol=tol, max_iter=max_iter,
                                n_grid=self._shooting_n_grid(1000),
                                r_max=self.r_max if self.r_max is not None else 75.0,
                                screening_potential=self._screening_shift())

        if isinstance(method, DiracFock):
            # Use jj-average with statistical weights
            from atomx.config.jj_coupling import JJConfig, generate_jj_average
            subshells, jj_configs = generate_jj_average(config_str, averaging="statistical")
            # Remove zero-occupation subshells
            kept = [i for i, sh in enumerate(subshells) if sh.occupation > 1e-10]
            subshells = [subshells[i] for i in kept]
            jj_configs = [
                JJConfig(subshells=[c.subshells[i] for i in kept], weight=c.weight)
                for c in jj_configs
            ]
            result = method.solve(
                self.Z, subshells, tol=tol, max_iter=max_iter,
                jj_configs=jj_configs, log=calc_log,
                ns=self.ns, use_physics_grid=self._use_physics_grid,
                screening_potential=self._screening_shift(),
                r_max=self.r_max, grid_hi=self.grid_hi,
                grid_he=self.grid_he, grid_hmax=self.grid_hmax, **orbital_controls)
            if calc_log:
                calc_log.converged(result)
                calc_log.close()
            return result

        if isinstance(method, MCDHF):
            return method.solve(
                self.Z, self._config_strings,
                tol=tol, max_iter=max_iter, ns=self.ns,
                use_physics_grid=self._use_physics_grid,
                r_max=self.r_max, grid_hi=self.grid_hi,
                grid_he=self.grid_he, grid_hmax=self.grid_hmax,
                screening_potential=self._screening_shift())

        raise TypeError(f"Unknown method type: {type(method).__name__}")

    # ------------------------------------------------------------------
    # Result-flavor introspection (back-compat aliases)
    # ------------------------------------------------------------------

    @property
    def result(self):
        """The underlying :class:`atomx.results.AtomicResult`, whatever
        flavor was produced by :meth:`solve` (SCFResult, MCDHFResult,
        ShootingResult, …)."""
        return self._result

    @property
    def subshells(self):
        """Subshells of the solved result (relativistic occupations)."""
        if self._result is None:
            raise RuntimeError("call solve() first")
        return self._result.subshells

    @property
    def calculation_spec(self):
        """Immutable model, state, discretization and correction specification."""
        if self._result is None:
            raise RuntimeError('call solve() first to resolve the actual calculation')
        return self._result.calculation_spec

    @property
    def provenance(self) -> dict:
        """What actually ran: ``{'engine': ..., 'exchange': ...,
        'averaging': ...}``, plus ``'plasma'`` when the atom sits in one
        (empty before :meth:`solve`)."""
        return dict(getattr(self._result, "provenance", None) or {})

    @property
    def resolved_inputs(self) -> str:
        """One-line summary of the constructor inputs ACTUALLY in force.

        Resolves what the constructor left open: the radial engine behind
        ``engine=None``, the knot grid behind ``ns=None``/``grid_*``, the
        corrections behind a stack, and the SCF/CI protocol choices.
        Useful for logging a run and for checking that a grouped object
        took effect.

        Examples
        --------
        >>> Atom(10, config="1s2 2s2 2p6").resolved_inputs
        'Z=10 charge=0 fermi | 1s2 2s2 2p6 | engine=shooting exchange=hf
        averaging=hamiltonian | breit=on qed=on(auto) | radial=shooting(n_grid=1000)
        r_max=auto'
        """
        if len(self._config_strings) == 1:
            config = self._config_strings[0]
        else:
            config = f"{len(self._config_strings)} configs"
        if self.correction_stack is not None:
            corrections = f"stack({type(self.correction_stack).__name__})"
        else:
            corrections = (f"breit={'on' if self.breit else 'off'} "
                           f"qed={'on' if self.qed else 'off'}"
                           f"({self.qed_backend})")
        if self.engine == "shooting":
            # Shooting integrates on its own adaptive grid; the B-spline
            # node count/order below would be noise.
            if self._multi_config:
                from atomx.numerics.shooting.electron_space import ShootingSpace
                space = self.shooting_space or ShootingSpace()
                n_grid = self._n_grid if self._n_grid is not None else space.n_grid
                radial = (f"radial=shooting(n_grid={n_grid}, "
                          f"continuum_roots={space.continuum_roots})")
            else:
                n_grid = self._shooting_n_grid(
                    1000 if self.exchange_model == "hf" else 400)
                radial = f"radial=shooting(n_grid={n_grid})"
        else:
            # ``self.ns`` is only the nominal 113 when the DBSR_HF physics
            # grid is in charge; the real node count comes out of
            # make_physics_knots at solve time.
            ns_str = ("auto(DBSR_HF)" if self._use_physics_grid
                      else str(self.ns))
            if not self._use_physics_grid:
                grid = "fixed"
            elif (self.grid_hi is None and self.grid_he is None
                    and self.grid_hmax is None):
                grid = "DBSR_HF-default"
            else:
                grid = (f"DBSR_HF(hi={self.grid_hi}, he={self.grid_he}, "
                        f"hmax={self.grid_hmax})")
            radial = f"radial=bspline(ns={ns_str}, ks={self.ks}, grid={grid})"
        parts = [f"Z={self.Z} charge={self.charge} {self.nuclear}",
                 config,
                 f"engine={self.engine} exchange={self.exchange_model} "
                 f"averaging={self.averaging}",
                 corrections,
                 f"{radial} "
                 f"r_max={'auto' if self.r_max is None else self.r_max}"]
        if self.starting_potential is not None:
            parts.append(f"start={self.starting_potential}")
        if getattr(self._sampling, "label", None):
            parts.append(f"sampling={self._sampling.label}")
        if self.radial_optimization != "mcdhf":
            parts.append(f"radial={self.radial_optimization}")
        if self.state_average is not None:
            parts.append(f"ci_state_average={self.state_average!r}")
        if self.j_resolved_csfs:
            parts.append("j_resolved_csfs")
        if self.per_block_orbital_sets:
            parts.append("per_block_orbital_sets")
        if self.plasma_ne is not None:
            parts.append(f"plasma(ne={self.plasma_ne}, Te={self.plasma_Te}, "
                         f"{self.screening_model})")
        return " | ".join(parts)

    @property
    def _csf_result(self):
        """State-resolved DF/MCDHF result, or None for a configuration average."""
        from atomx.methods.csf.solver import VariationalCSFResult
        return self._result if isinstance(self._result, VariationalCSFResult) else None

    @property
    def _bsp_data(self):
        """Back-compat alias: B-spline workspace of the underlying
        result (or ``None`` for shooting)."""
        return getattr(self._result, "_bsp_data", None) if self._result else None

    # ------------------------------------------------------------------
    # AtomicResult-protocol delegators
    # ------------------------------------------------------------------

    @property
    def converged(self) -> bool:
        return bool(self._result.converged) if self._result else False

    @property
    def iterations(self) -> int:
        return int(getattr(self._result, "iterations", 0)) if self._result else 0

    @property
    def total_energy(self):
        """Total energy in Hartree."""
        return self._result.total_energy if self._result else None

    @property
    def total_energy_eV(self):
        E = self.total_energy
        return E * HARTREE_EV if E is not None else None

    def total_energy_as(self, unit: str = "hartree"):
        """Total energy in the requested unit (pint Quantity)."""
        from atomx.results import _as_quantity
        E = self.total_energy
        if E is None:
            return None
        return _as_quantity(E, "hartree", unit)

    @property
    def orbital_energies(self) -> dict:
        return dict(self._result.orbital_energies) if self._result else {}

    def orbital_energy_as(self, label: str, unit: str = "hartree"):
        """Orbital energy ε_a in the requested unit (pint Quantity)."""
        from atomx.results import _as_quantity
        if not self._result or label not in self._result.orbital_energies:
            raise KeyError(
                f"{label!r} not in orbital_energies; "
                f"available: {list(self.orbital_energies)}")
        return _as_quantity(self._result.orbital_energies[label], "hartree", unit)

    @property
    def energy_components(self) -> dict:
        """Per-method energy breakdown (alias of ``energy_breakdown``)."""
        if not self._result:
            return {}
        # Both names appear in the wild; prefer the protocol attr.
        return dict(getattr(self._result, "energy_breakdown",
                             getattr(self._result, "energy_components", {})))

    @property
    def energy_breakdown(self) -> dict:
        """AtomicResult-protocol alias for :attr:`energy_components`."""
        return self.energy_components

    @property
    def theory_level(self):
        """``TheoryLevel`` of the underlying result."""
        return self._result.theory_level if self._result else None

    @property
    def method_name(self) -> str:
        """Human-readable method name (DiracFock, MCDHF, BreitPauli, …)."""
        return self._result.method_name if self._result else ""

    @property
    def coupling_scheme(self):
        """``CouplingScheme`` (LS / JJ / INTERMEDIATE) of the result."""
        return self._result.coupling_scheme if self._result else None

    @property
    def waves(self):
        """Radial wavefunctions container (:class:`atomx.results.Waves`)."""
        return self._result.waves if self._result else None

    @property
    def r(self) -> Optional[np.ndarray]:
        """Radial grid in Bohr (shortcut for ``atom.waves.grid``).

        Returns ``None`` if no result has been computed or the grid is empty.
        Same array that indexes the orbital ``P`` / ``Q`` components; using
        it avoids interpolation.
        """
        w = self.waves
        if w is None:
            return None
        g = np.asarray(w.grid, dtype=float)
        return g if g.size else None

    grid = r  # alias — pick the name that reads best at the call site

    # ------------------------------------------------------------------
    # Per-orbital access
    # ------------------------------------------------------------------

    @property
    def orbitals(self) -> dict:
        """``{label: OrbitalWavefunction}`` for every orbital in this atom.

        Each value is an :class:`~atomx.results.OrbitalWavefunction`
        carrying the radial components ``P`` (and ``Q`` for Dirac),
        the quantum numbers ``n / l / kappa / j / s``, ``occupation``,
        and the one-electron ``energy``.

        >>> ne = Atom(Z=10).solve()
        >>> ne.orbitals['2p3/2'].kappa
        -2
        >>> ne.orbitals['2p3/2'].j
        1.5
        """
        w = self.waves
        return dict(w.orbitals) if w is not None else {}

    def orbital(self, label: str):
        """Look up orbital(s) matching ``label``.

        Short labels like ``"1s"`` are resolved via
        :meth:`Waves.resolve_all`:

        - If exactly one orbital matches, returns a single
          :class:`~atomx.results.OrbitalWavefunction`.
        - If multiple match (e.g. ``"2p"`` → ``"2p1/2"`` + ``"2p3/2"``),
          returns a **dict** ``{label: OrbitalWavefunction}``.

        >>> ne = Atom(Z=10).solve()
        >>> ne.orbital('1s')            # single match → OrbitalWavefunction
        >>> ne.orbital('2p')            # two matches → dict
        >>> ne.orbital('2p')['2p3/2']   # drill into the dict

        Returns
        -------
        OrbitalWavefunction or dict[str, OrbitalWavefunction]
        """
        w = self.waves
        if w is None:
            raise RuntimeError("Atom has no result yet — call .solve() first.")
        resolved = w.resolve_all(label)
        if len(resolved) == 1:
            return w.orbitals[resolved[0]]
        return {lbl: w.orbitals[lbl] for lbl in resolved}

    def orbital_table(self) -> list:
        """Tabular per-orbital summary as a list of dicts.

        Each row is ``{label, n, l, l_symbol, kappa, j, s, occupation,
        energy_Ha, is_relativistic, n_grid_points}``.  Suitable for
        feeding into ``pandas.DataFrame(...)`` or pretty-print.
        """
        return [orb.to_dict() for orb in self.orbitals.values()]

    # ------------------------------------------------------------------
    # Multi-state accessors (single-state results return length-1 arrays)
    # ------------------------------------------------------------------

    @property
    def state_energies(self) -> Optional[np.ndarray]:
        """CI state energies; for single-state results, ``[total_energy]``."""
        if not self._result:
            return None
        if hasattr(self._result, "state_energies"):
            return np.asarray(self._result.state_energies)
        return np.array([self.total_energy], dtype=float)

    def excitation_energies(self, unit: str = "cm-1") -> np.ndarray:
        """Excitation energies relative to the ground state.

        For single-state results, returns ``[0.0]`` in the requested unit.
        """
        if self._result is not None and hasattr(self._result, "excitation_energies"):
            return self._result.excitation_energies(unit)
        return np.array([0.0])

    # ------------------------------------------------------------------
    # Derived observables (uniform across flavors)
    # ------------------------------------------------------------------

    def r_uniform(self, n: int = 400, *, kind: str = "log",
                  r_min: Optional[float] = None,
                  r_max: Optional[float] = None) -> Optional[np.ndarray]:
        """A smooth radial grid suitable for plotting.

        ``atom.r`` returns the integration grid (Gauss-Legendre nodes
        for the B-spline backend; Salvat's SGRID for the shooting
        backend) — both are correct for ∫ but cluster at knots, which
        looks zigzag in line plots.  This helper returns a clean
        log-spaced (default) or linear grid that spans the support of
        the stored orbitals.

        Parameters
        ----------
        n : int
            Number of grid points.  Default 400.
        kind : {"log", "linear"}
            Spacing.  Log is the right default for atomic orbitals;
            linear is occasionally useful for outer-region close-ups.
        r_min, r_max : float, optional
            Override the auto-detected support.  ``r_min`` defaults to
            ``max(1e-6, native[0])`` and ``r_max`` to where the orbital
            density first drops below ``1e-12`` of its peak.

        Returns ``None`` if the underlying result has no stored
        wavefunctions.
        """
        if kind not in ("log", "linear"):
            raise ValueError(f"kind must be 'log' or 'linear' (got {kind!r})")
        w = self.waves
        if w is None or not w.orbitals:
            return None
        grid = np.asarray(w.grid, dtype=float)
        if grid.size == 0:
            return None

        if r_min is None:
            r_min = max(1e-6, float(grid[grid > 0].min()) if (grid > 0).any() else 1e-6)
        if r_max is None:
            # Find where the largest orbital density tail drops to 1e-12 of peak
            tail_cut = grid[-1]
            for orb in w.orbitals.values():
                P = np.asarray(orb.P, dtype=float)
                if P.shape != grid.shape:
                    continue
                rho = P ** 2
                if orb.Q is not None:
                    Q = np.asarray(orb.Q, dtype=float)
                    rho = rho + Q ** 2
                peak = rho.max()
                if peak <= 0:
                    continue
                threshold = 1e-12 * peak
                below = np.where(rho < threshold)[0]
                # First index past the orbital's classical-turning region
                # where density falls under threshold and stays below.
                for idx in below:
                    if idx > rho.argmax() and (rho[idx:] < threshold).all():
                        tail_cut = max(tail_cut, float(grid[idx]))
                        break
            r_max = min(tail_cut, float(grid[-1]))

        if kind == "log":
            return np.geomspace(r_min, r_max, n)
        return np.linspace(r_min, r_max, n)

    def density(self, r: Optional[np.ndarray] = None, *,
                grid: str = "native", n: int = 400) -> Optional[np.ndarray]:
        """Total radial electron density.

        When ``r`` is **not supplied** and ``grid='native'`` (defaults),
        returns the native radial density directly from the stored
        orbitals on :attr:`r`:

            D(r) = Σ_a n_a [P_a(r)² + Q_a(r)²]        (∫ D dr = N_electrons)

        This is the "original" per-orbital density — no 1/r² factor and no
        interpolation — matching the DBSR_HF / density_gs H5 convention. The
        grid is available via :attr:`r` (or equivalently :attr:`grid`).

        When ``r`` **is supplied**, P and Q are linearly interpolated onto
        ``r`` and the same quantity Σ n_a (P² + Q²) is returned on that
        grid. Points outside the native grid are filled with 0.

        ``grid='log'`` is a convenience for plotting: returns the density
        on a log-spaced grid of ``n`` points spanning the body of the
        wavefunction (clipped at the small-r and large-r tails).  Use
        :meth:`r_uniform` to get the matching radial axis.

        Returns ``None`` if the underlying result has no stored wavefunctions
        (e.g. the shooting backend).
        """
        if grid not in ("native", "log", "linear"):
            raise ValueError(f"grid must be 'native' | 'log' | 'linear' (got {grid!r})")
        w = self.waves
        if w is None or not w.orbitals:
            return None
        native_grid = np.asarray(w.grid, dtype=float)
        if native_grid.size == 0:
            return None

        if r is None and grid != "native":
            r = self.r_uniform(n=n, kind=grid)
        interpolate = r is not None
        r_eval = np.asarray(r, dtype=float) if interpolate else native_grid
        rho = np.zeros_like(r_eval, dtype=float)

        if self._csf_result is not None:
            keys, density = self._csf_result.one_body_density_matrix()
            sampled = {(orb.n, orb.kappa): orb for orb in w.orbitals.values()}
            p, q = [], []
            for key in keys:
                orb = sampled[key]
                P = np.asarray(orb.P, dtype=float)
                Q = np.zeros_like(P) if orb.Q is None else np.asarray(orb.Q, dtype=float)
                if interpolate:
                    P = np.interp(r_eval, native_grid, P, left=0.0, right=0.0)
                    Q = np.interp(r_eval, native_grid, Q, left=0.0, right=0.0)
                p.append(P)
                q.append(Q)
            p, q = np.asarray(p), np.asarray(q)
            return np.einsum("ab,ar,br->r", density, p, p, optimize=True) + np.einsum(
                "ab,ar,br->r", density, q, q, optimize=True)

        for orb in w.orbitals.values():
            P = np.asarray(orb.P, dtype=float)
            if P.shape != native_grid.shape:
                continue
            if interpolate:
                P_eval = np.interp(r_eval, native_grid, P, left=0.0, right=0.0)
            else:
                P_eval = P
            contrib = P_eval ** 2
            if orb.Q is not None:
                Q = np.asarray(orb.Q, dtype=float)
                if interpolate:
                    Q_eval = np.interp(r_eval, native_grid, Q, left=0.0, right=0.0)
                else:
                    Q_eval = Q
                contrib = contrib + Q_eval ** 2
            rho = rho + orb.occupation * contrib
        return rho

    def corrections(self, stack=None) -> dict:
        """Apply a :class:`atomx.corrections.CorrectionStack` to this result.

        Parameters
        ----------
        stack
            A pre-configured :class:`CorrectionStack`.  If ``None``,
            uses the default ``shabaev_local`` SE + Uehling VP + Breit
            stack.  Backends that require workspace the result does not
            carry are silently skipped (their reasons are recorded
            under the ``"_skipped"`` key of the returned dict).

        Returns
        -------
        dict[str, CorrectionResult]
        """
        if self._result is None:
            return {}
        from atomx.corrections import CorrectionStack
        if stack is None:
            stack = CorrectionStack()
        return stack.apply(self._result)

    def climb(self, through=None, *, ladder=None, sampling=None):
        """Run a ladder of theory rungs against this atom.

        Parameters
        ----------
        through : Method, optional
            Stop at this rung (inclusive).  If ``None``, climb the
            entire ladder.
        ladder : Ladder, optional
            Custom ladder.  Defaults to :meth:`atomx.Ladder.standard`.
        sampling : Sampling, optional
            Numerical-discretization tier (e.g. ``Sampling.standard``).
            Falls back to whatever was attached to this atom at
            construction time.

        Returns
        -------
        :class:`atomx.ladder.LadderRun`
        """
        from atomx.ladder import Ladder
        base = ladder or Ladder.standard()
        actual = base.through(through) if through is not None else base
        return actual.run(self, sampling=sampling or self._sampling)

    def show_ladder(self, through=None, *, unit: str = "eV",
                    ladder=None, sampling=None, file=None):
        """Climb the ladder and pretty-print the result table.

        Returns the underlying :class:`LadderRun` for further inspection.
        """
        run = self.climb(through=through, ladder=ladder, sampling=sampling)
        run.show(unit=unit, file=file)
        return run

    def transitions(self, multipoles=None, gauge='length',
                    line_resolution='uta', ci_det_cap=2000, *,
                    fallback='error', ci_reference=None):
        """Compute radiative transitions between atomic states.

        Parameters
        ----------
        multipoles : list of str, optional
            Multipole types: 'E1', 'M1', 'E2', 'M2', etc. Default: ['E1'].
        line_resolution : {'uta', 'ci'}
            ``'ci'`` switches single-configuration-pair atoms to the
            exact J-resolved determinant-CI line engine
            (:mod:`atomx.properties.transitions.ci_lines`, FAC-verified
            per line), using an explicit solved ci_reference. Unsupported
            requests raise unless fallback='uta' is explicitly selected.  Each result row records its engine in
            ``TransitionResult.resolution``.  Default ``'uta'``.
        ci_det_cap : int
            Determinant-per-block cap for the CI engine.
        gauge : {'length', 'velocity', 'both'}
            ``'length'`` (Babushkin) — default; matches the historical
            ``A_rate_per_s`` value.
            ``'velocity'`` (Coulomb) — velocity-gauge rates only.
            ``'both'`` — populates ``A_length`` and ``A_velocity`` on each
            :class:`TransitionResult`; ``A_rate_per_s`` stays length-gauge
            for backward compatibility, so the FAC-style gauge-ratio
            diagnostic (Gu 2008 §3.2) can be inspected via
            ``result.gauge_ratio``.

        Returns
        -------
        list of TransitionResult
            Transitions sorted by wavelength.
        """
        from atomx.properties.transitions.calculator import TransitionCalculator
        calc = TransitionCalculator(self)
        return calc.compute_all(multipoles, gauge,
                                line_resolution=line_resolution,
                                ci_det_cap=ci_det_cap, fallback=fallback,
                                ci_reference=ci_reference)

    def hyperfine_constants(self, *args, **kwargs):
        """Magnetic-dipole and electric-quadrupole hyperfine constants.

        Delegates to :meth:`SCFResult.hyperfine_constants` when the
        underlying result is an :class:`SCFResult`.  Raises
        :class:`NotImplementedError` for other flavors that don't yet
        own a hyperfine evaluator (e.g. the current shooting backend).
        """
        if self._result is None:
            raise RuntimeError("Atom has no result yet — call .solve() first.")
        if not hasattr(self._result, "hyperfine_constants"):
            raise NotImplementedError(
                f"hyperfine_constants is not yet implemented for "
                f"{type(self._result).__name__}; convert to a DHF baseline "
                f"or use atomx.properties.hyperfine directly."
            )
        return self._result.hyperfine_constants(*args, **kwargs)

    def zeeman_factors(self, *args, **kwargs):
        """Bound-electron Landé g_J factors per orbital.

        Delegates to :meth:`SCFResult.zeeman_factors` when the
        underlying result supports it.
        """
        if self._result is None:
            raise RuntimeError("Atom has no result yet — call .solve() first.")
        if not hasattr(self._result, "zeeman_factors"):
            raise NotImplementedError(
                f"zeeman_factors is not yet implemented for "
                f"{type(self._result).__name__}."
            )
        return self._result.zeeman_factors(*args, **kwargs)

    def plot3d(self, label: str, **kwargs):
        """Shortcut for :meth:`Waves.plot3d` on the underlying result."""
        w = self.waves
        if w is None:
            raise RuntimeError("Atom has no result yet — call .solve() first.")
        return w.plot3d(label, **kwargs)

    def to_table(self) -> dict:
        """Compact dict summary suitable for :func:`atomx.compare`."""
        if self._result is None:
            return {"converged": False}
        out = {
            "Z": self.Z,
            "method": self.method_name,
            "theory_level": self.theory_level.name if self.theory_level else None,
            "coupling": self.coupling_scheme.name if self.coupling_scheme else None,
            "converged": self.converged,
            "iterations": self.iterations,
            "total_energy_Ha": self.total_energy,
            "n_orbitals": len(self.orbital_energies),
        }
        return out

    def line_list(self, multipoles=None, gauge='length', unit='angstrom'):
        """Print a formatted line list of transitions.

        Parameters
        ----------
        multipoles : list of str, optional
        gauge : str
        unit : str
            'angstrom', 'nm', 'eV', or 'cm-1'.
        """
        transitions = self.transitions(multipoles, gauge)
        if not transitions:
            print("No allowed transitions found.")  # noqa: T201
            return transitions

        print(f"{'Transition':>25s}  {'Type':>4s}  {'Wavelength':>12s}  "  # noqa: T201
              f"{'A (s^-1)':>12s}  {'gf':>10s}")
        print("-" * 70)  # noqa: T201
        for tr in transitions:
            label = f"{tr.initial_label} -> {tr.final_label}"
            if unit == 'nm':
                wl = f"{tr.wavelength_nm:.3f} nm"
            elif unit == 'eV':
                wl = f"{tr.energy_eV:.4f} eV"
            elif unit == 'cm-1':
                wl = f"{tr.energy_cm1:.1f} cm-1"
            else:
                wl = f"{tr.wavelength_angstrom:.3f} A"
            print(f"{label:>25s}  {tr.multipole:>4s}  {wl:>12s}  "  # noqa: T201
                  f"{tr.A_rate_per_s:>12.4e}  {tr.gf_value:>10.4e}")

        return transitions

    def excitation_cross_section(self, final_atom, energies_eV, **kwargs):
        """Compute electron-impact excitation cross-section (DW approximation).

        Parameters
        ----------
        final_atom : Atom
            Solved excited-state atom.
        energies_eV : array_like
            Incident electron energies in eV.

        Returns
        -------
        sigma : ndarray
            Cross-section in cm^2.
        omega : ndarray
            Collision strength (dimensionless).
        """
        from atomx.scattering.distorted_wave.solver import dw_excitation_cross_section
        return dw_excitation_cross_section(self, final_atom, energies_eV, **kwargs)

    def __repr__(self):
        sym = _element_symbol(self.Z)
        charge_str = ""
        if self.charge > 0:
            charge_str = f"^{self.charge}+"
        elif self.charge < 0:
            charge_str = f"^{abs(self.charge)}-"
        status = "solved" if self.converged else "unsolved"
        return f"Atom({sym}{charge_str}, Z={self.Z}, {status})"

    def summary(self):
        """Print a summary of the calculation."""
        sym = _element_symbol(self.Z)
        print(f"{'='*50}")  # noqa: T201
        print(f"  {sym} (Z={self.Z})  AtomX DHF Calculation")  # noqa: T201
        print(f"{'='*50}")  # noqa: T201
        print(f"  Config: {self._config_strings[0]}")  # noqa: T201
        print(f"  Resolved: {self.resolved_inputs}")  # noqa: T201
        print(f"  Converged: {self.converged} ({self.iterations} iterations)")  # noqa: T201
        print(f"  Total energy: {self.total_energy:.6f} Ha"  # noqa: T201
              f" ({self.total_energy_eV:.2f} eV)")
        print("\n  Orbital energies:")  # noqa: T201
        for label, E in sorted(self.orbital_energies.items(),
                                key=lambda x: x[1]):
            print(f"    {label:8s}: {E:12.6f} Ha ({E*HARTREE_EV:10.2f} eV)")  # noqa: T201

        if self.breit or self.qed:
            print("\n  Corrections:")  # noqa: T201
            ec = self.energy_components
            for key in ["E_Breit", "E_VP", "E_SE"]:
                if key in ec:
                    print(f"    {key:8s}: {ec[key]:12.2e} Ha"  # noqa: T201
                          f" ({ec[key]*HARTREE_CM1:10.1f} cm-1)")

        if self._multi_config and self._csf_result:
            print("\n  CI States:")  # noqa: T201
            for i, E in enumerate(self.state_energies):
                de = (E - self.state_energies[0]) * HARTREE_CM1
                print(f"    State {i}: {E:.6f} Ha  (ΔE = {de:.0f} cm-1)")  # noqa: T201


def _element_symbol(Z):
    """Element symbol from atomic number."""
    from atomx.constants import element_symbol
    return element_symbol(Z)
