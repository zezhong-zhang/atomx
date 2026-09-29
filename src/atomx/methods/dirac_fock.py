"""Dirac-Fock self-consistent field solver with non-local HF exchange.

Two modes:
- DFS (Dirac-Fock-Slater): local exchange V_x ∝ -ρ^{1/3} — fast, approximate
- DHF (Dirac-Hartree-Fock): non-local exchange via Y^k — accurate, slower

Total energy computed as:
    E = Σ_a q_a ε_a^{1e} + (1/2) Σ_{a,b} [q_a*q_b*F^0(a,b) + Σ_k c^k * G^k(a,b)]
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from atomx.angular.coefficients import (
    exchange_k_range,
)
from atomx.methods.base import Method
from atomx.methods.scf_observables import SCFObservablesMixin
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.convol import (
    ConvolCache,
    compute_slater_fk_cell,
    compute_slater_gk_cell,
)
from atomx.numerics.bspline.dirac_assembly import (
    build_dirac_matrices,
    canonicalize_radial_phase,
    solve_dirac,
)
from atomx.numerics.bspline.evaluate import compute_yk, eval_orbital_on_grid
from atomx.numerics.bspline.integrals import (
    _assemble_integral,
    nuclear_matrix,
    overlap_matrix,
)
from atomx.numerics.bspline.knots import make_knots
from atomx.numerics.diis import diis_extrapolate
from atomx.types import Exchange, TheoryLevel


from atomx.config.subshell import Subshell


def _canonicalized_orbitals_raw(orbitals):
    out = {}
    for kappa, orb_list in orbitals.items():
        fixed = []
        for P, Q, occ in orb_list:
            P_fix, Q_fix = canonicalize_radial_phase(P, Q)
            fixed.append((P_fix, Q_fix, occ))
        out[kappa] = fixed
    return out


@dataclass
class SCFResult(SCFObservablesMixin):
    """Result of an SCF calculation.

    Satisfies the ``AtomicResult`` protocol from ``atomx.results``.
    """
    converged: bool
    iterations: int
    residual: float
    total_energy: float
    orbital_energies: Dict[str, float]
    subshells: List[Subshell]
    energy_components: Dict[str, float] = field(default_factory=dict)
    orbitals_raw: Dict = field(default_factory=dict)  # kappa -> [(P, Q, occ)]
    Z: Optional[int] = None
    _theory_level: Optional[int] = field(default=None, repr=False)
    _method_name: Optional[str] = field(default=None, repr=False)
    _bsp_data: Optional[Dict] = field(default=None, repr=False)
    calculation_spec: object = None

    @property
    def energy_breakdown(self) -> Dict[str, float]:
        """Alias for energy_components (AtomicResult protocol)."""
        return self.energy_components

    @property
    def theory_level(self):
        from atomx.types import TheoryLevel
        if self._theory_level is not None:
            return TheoryLevel(self._theory_level)
        return TheoryLevel.DIRAC_HARTREE_FOCK

    @property
    def method_name(self) -> str:
        return self._method_name or "DiracFock"

    @property
    def coupling_scheme(self):
        from atomx.types import CouplingScheme
        return CouplingScheme.JJ

    def orbitals_by_subshell(self) -> Dict:
        """``{(n, kappa): (P, Q, occ)}`` — orbitals keyed by quantum numbers.

        ``orbitals_raw`` stores per-kappa lists whose positions are aligned
        with the per-kappa order of ``subshells`` (the order the solver
        iterated them).  That positional invariant lives HERE, in one
        place; downstream consumers (MCDHF orbital extraction, correction
        adapters) should key by ``(n, kappa)`` through this accessor
        instead of replicating per-kappa counters — with many same-kappa
        shells (e.g. Pb: six ``ns``, five ``np``) a counter mismatch
        silently mislabels orbitals.
        """
        out: Dict = {}
        counters: Dict[int, int] = {}
        for sh in self.subshells:
            idx = counters.get(sh.kappa, 0)
            counters[sh.kappa] = idx + 1
            orb_list = self.orbitals_raw.get(sh.kappa, [])
            if idx < len(orb_list):
                out[(sh.n, sh.kappa)] = orb_list[idx]
        return out

    # ------------------------------------------------------------------
    # Unit-aware accessors (pint)
    # ------------------------------------------------------------------

    def total_energy_as(self, unit: str = "hartree"):
        """Total energy in the requested unit (pint Quantity).

        Internal unit is Hartree; spectroscopic conversions (to eV,
        cm⁻¹, MHz, K via k_B) are handled automatically.

        Requires :mod:`pint` — install via ``pip install atomx[units]``.

        >>> result.total_energy_as("eV")
        <Quantity(-3497.47, 'electron_volt')>
        >>> result.total_energy_as("cm^-1")
        <Quantity(-28215000, '1 / centimeter')>
        """
        from atomx.results import _as_quantity
        return _as_quantity(self.total_energy, "hartree", unit)

    def orbital_energy_as(self, label: str, unit: str = "hartree"):
        """One-electron orbital energy ε_a for orbital ``label`` in target unit."""
        from atomx.results import _as_quantity
        if label not in self.orbital_energies:
            raise KeyError(
                f"{label!r} not in orbital_energies; "
                f"available: {list(self.orbital_energies)}")
        return _as_quantity(self.orbital_energies[label], "hartree", unit)

    @property
    def radial_state(self):
        """Owned native radial snapshot for correction and analysis consumers."""
        from atomx.radial_state import radial_state_of
        if '_radial_state' not in self.__dict__:
            self.__dict__['_radial_state'] = radial_state_of(self)
        return self.__dict__['_radial_state']

    @property
    def waves(self):
        """Build :class:`atomx.results.Waves` container from B-spline coefficients."""
        from atomx.results import OrbitalWavefunction, Waves, RadialMetric
        if not self.orbitals_raw or not self._bsp_data:
            return Waves(grid=np.array([]))

        bsp_data = self._bsp_data
        gr = bsp_data["gr"]  # (nv, n_gauss)
        grid = gr.ravel()

        orbitals = {}
        by_subshell = self.orbitals_by_subshell()
        for sh in self.subshells:
            orbital = by_subshell.get((sh.n, sh.kappa))
            if orbital is not None:
                P_coeffs, Q_coeffs, occ = orbital
                # Evaluate B-spline on grid
                P_grid = _eval_bspline_on_grid(P_coeffs, bsp_data, "p")
                Q_grid = _eval_bspline_on_grid(Q_coeffs, bsp_data, "q")
                orbitals[sh.label] = OrbitalWavefunction(
                    P=P_grid, Q=Q_grid,
                    n=sh.n, l=sh.l, kappa=sh.kappa,
                    occupation=sh.occupation,
                    energy=self.orbital_energies.get(sh.label, 0.0),
                    label=sh.label)
        return Waves(grid=grid, orbitals=orbitals, basis_type="bspline",
                     metric=RadialMetric.bspline(bsp_data))



    def save(self, path: str):
        """Save a versioned native result, including physical provenance."""
        from atomx.io.results import save_scf_result
        save_scf_result(self, path)

    @classmethod
    def load(cls, path: str) -> "SCFResult":
        """Load a native result; old files retain explicit missing metadata."""
        from atomx.io.results import load_scf_result
        return load_scf_result(path, result_type=cls)


#: Orthogonality is a constraint, not an energy-dependent tolerance.
_SAME_KAPPA_OVERLAP_TOL = 1.0e-9


def _max_same_kappa_overlap(orbitals, bsp_data) -> float:
    """Largest |<a|b>| over distinct orbitals sharing a kappa.

    Returns 0.0 when nothing can be compared (no B-spline workspace, or
    no kappa with two orbitals).
    """
    if bsp_data is None:
        return 0.0
    worst = 0.0
    for _kappa, orb_list in orbitals.items():
        for i in range(len(orb_list)):
            for j in range(i + 1, len(orb_list)):
                Pa, Qa = orb_list[i][0], orb_list[i][1]
                Pb, Qb = orb_list[j][0], orb_list[j][1]
                worst = max(worst, abs(_orbital_overlap(Pa, Qa, Pb, Qb, bsp_data)))
    return float(worst)


def _orbital_overlap(Pa, Qa, Pb, Qb, bsp_data) -> float:
    """``<a|b> = int (P_a P_b + Q_a Q_b) dr`` on the B-spline Gauss grid."""
    fa = _eval_bspline_on_grid(Pa, bsp_data, "p")
    fb = _eval_bspline_on_grid(Pb, bsp_data, "p")
    qa = _eval_bspline_on_grid(Qa, bsp_data, "q")
    qb = _eval_bspline_on_grid(Qb, bsp_data, "q")
    grw = np.asarray(bsp_data["grw"]).ravel()
    return float(np.sum(grw * (fa * fb + qa * qb).ravel()))


class UnboundSubshellError(RuntimeError):
    """A requested subshell has no bound state in the converged potential.

    Raised instead of silently returning a zero orbital.  Before this
    guard, requesting an unbound (virtual) orbital — e.g. a ``3d`` in
    Ne, or any Rydberg state outside the box — produced a total energy
    of ``0.0`` Ha (or, when the basis collapsed towards the negative
    energy continuum, ``-2 m_e c²`` times the occupancy) with
    ``converged=True``.  See
    ``docs/archive/2026-09/PHYSICS_NUMERICS_REVIEW_2026-09-19.md`` finding C4.
    """


def _eval_bspline_on_grid(coeffs, bsp_data, component="p"):
    """Evaluate a B-spline expansion on the quadrature grid, flattened.

    Thin wrapper over the vectorised :func:`eval_orbital_on_grid` (einsum) —
    replaces a former O(nv·n_gauss·ks) triple Python loop.  Returns a flat
    ``(nv*n_gauss,)`` array for the hyperfine / MCDHF callers that consume it.

    Parameters
    ----------
    coeffs : np.ndarray
        B-spline expansion coefficients.
    bsp_data : dict
        B-spline data containing grid and basis values.
    component : str
        "p" for large component, "q" for small component.
    """
    key = "bsp_p" if component == "p" else "bsp_q"
    bsp = bsp_data[key]  # (nv, n_gauss, ks)
    nv, _, ks_comp = bsp.shape
    return eval_orbital_on_grid(coeffs, bsp, nv, ks_comp, len(coeffs)).ravel()


class DFSolver:
    """Dirac-Fock self-consistent field solver.

    Parameters
    ----------
    Z : int
        Nuclear charge.
    subshells : list of Subshell
        Occupied subshells.
    ns, ks : int
        B-spline parameters.
    r_min, r_max : float
        Grid boundaries.
    exchange : str
        "hf" for non-local Hartree-Fock exchange,
        "slater" for local Slater exchange with ``alpha_x`` parameter,
        "gu" for Gu's FAC-style local exchange with α_xc(Z, N)
        (ignores ``alpha_x``; fit from
        :func:`atomx.physics.local_exchange.get_hxs`).
    alpha_x : float
        Slater exchange parameter (used when exchange="slater").
    fac_ihx, fac_hxs : float
        Forwarded to ``get_hxs`` when ``exchange='gu'``; defaults
        match FAC's own defaults (``ihx = -1``, ``hxs = 1``).
    mean_field_coeffs : ndarray, optional
        Asymmetric mean-field seam (R7).  ``(n_orb, n_orb)`` matrix
        replacing the internally derived Guerra product ``WW(i, j)``:
        entry ``[i, j]`` is the *unnormalised* coefficient of source
        orbital ``j`` in target ``i``'s direct and exchange potential.
        See :meth:`_mean_field_weight` for the exact convention.
    mean_field_normalization : ndarray, optional
        ``(n_orb,)`` per-target divisor replacing
        ``subshells[i].occupation``; the effective per-electron weight
        of source ``j`` in target ``i``'s potential is
        ``mean_field_coeffs[i, j] / mean_field_normalization[i]``.
        Required alongside ``mean_field_coeffs`` (no implicit default —
        a silent mismatch between numerator and divisor is exactly the
        convention bug this seam exists to remove).
    """

    def __init__(self, Z, subshells, ns=50, ks=9, r_min=None, r_max=None,
                 exchange="hf", alpha_x=2.0/3.0,
                 fac_ihx=-1.0, fac_hxs=1.0,
                 include_breit=False, include_qed=False,
                 qed_se_method="auto",
                 include_wichmann_kroll=False,
                 correction_stack=None,
                 nuclear="point",
                 starting_potential=None,
                 jj_configs=None,
                 mean_field_coeffs=None,
                 mean_field_normalization=None,
                 kinetic_balance=False,
                 screening_potential=None,
                 log=None, nuclear_mass_number=None):
        """
        Parameters
        ----------
        exchange : str
            "hf" for non-local Hartree-Fock exchange,
            "slater" for local Slater exchange (α from ``alpha_x``),
            "gu" for Gu's Z- and N-dependent α_xc(Z, N),
            "tfda" for Thomas-Fermi-Dirac-Amaldi (no SCF iteration),
            "gsz" for Green-Sellin-Zachor parametric potential.
        qed_se_method : str
            Self-energy backend, see
            :func:`atomx.corrections.qedmod.available_backends`.
            Only consulted when ``include_qed=True``.
            Default ``"auto"`` uses Mohr/MSS for Z <= 36 and the
            FAC/Indelicato table for Z >= 37.  Pass ``"mss"`` or
            ``"mohr"`` to reproduce atomx <= 0.2 self-energy output.
        nuclear : str
            Nuclear model: "point", "fermi", "uniform".
        starting_potential : str or None
            Starting potential for SCF: None (hydrogenic), "tfda", "gsz".
        jj_configs : list of JJConfig, optional
            Multi-configuration jj-coupled configurations with weights.
            Used for WW-based energy coefficients (DBSR_HF av_energy_coef).
            If None, uses simple q*(q-1)/2 and q_a*q_b formulas.
        mean_field_coeffs, mean_field_normalization : ndarray, optional
            Asymmetric mean-field seam (R7).  ``mean_field_coeffs[i, j]``
            replaces ``WW(i, j)`` as the raw direct/exchange coefficient
            of source ``j`` acting on target ``i``, and
            ``mean_field_normalization[i]`` replaces ``q_i`` (the target's
            occupation) as the divisor that turns it into the per-electron
            weight of the target's Fock operator.  The symmetric Guerra
            product cannot express a mean field in which a virtual
            correlation orbital is screened by the occupied electrons
            while those occupied orbitals are *not* screened by the
            empty orbital — the asymmetry a multi-configuration layer
            protocol requires.  ``None`` (default) keeps the internally
            derived ``WW(i, j) / subshells[i].occupation`` convention
            used by every non-MCDHF caller.
        kinetic_balance : bool, optional
            Use restricted kinetic balance (RKB): the P (large) component is
            expanded in order ``ks-1`` B-splines and Q (small) in order ``ks``.
            RKB removes the ~ns spurious positive-energy states of an
            unbalanced dual basis, which matters for sum-over-states
            properties (polarizability, QED) and high-Z spectral cleanliness.
            Default ``False`` reproduces the historical unbalanced basis
            exactly; ground-state energies agree with the RKB basis to ~1e-5
            (e.g. Ne DHF is identical to 6 digits).  The ``Atom`` / MCDHF
            production paths already build the basis with ``kinetic_balance=True``.
        """
        self.Z = Z
        self.subshells = subshells
        self.jj_configs = jj_configs
        # Asymmetric mean-field seam (R7).  ``None`` ⇒ every consumer
        # falls back to the internally derived Guerra product and the
        # target's own occupation, i.e. the pre-seam behaviour
        # bit-for-bit.  A supplied pair is validated here so a shape or
        # finiteness mistake fails at construction rather than silently
        # shifting a Fock coefficient.
        #
        # History (2026-09-20): an intermediate version of this seam was
        # briefly hard-disabled here after a He ``1s² + 1s¹2p¹`` regression,
        # on the reading that "the Guerra arithmetic was already right at
        # zero population".  That reading was wrong on both counts:
        #
        # * The He 2p *did* have zero Guerra population (its CSFs are odd
        #   parity and carry exactly zero amplitude in the even ground
        #   state), so it *was* the R7 case — the pre-seam arithmetic gave
        #   the bare-nucleus 2p (ε = −0.5000 Ha, the hydrogenic Z=2 2p).
        #   That passes ``E(¹P₁) − E(³P₁) = 2G¹(1s,2p)/3`` because the
        #   identity is a CI-consistency check on whatever orbitals are
        #   used; it does not make the orbital physical.  The absolute
        #   splitting was 0.0937 Ha against the NIST 0.0514 Ha.
        # * The regression came from the *anchor*, not from having a seam:
        #   anchoring on the raw reference occupations over-screens
        #   whenever the reference holds more electrons in a shell than
        #   any configuration containing the target does (He: reference
        #   1s², every 2p parent 1s¹ → the 1s² field is neutral and binds
        #   no 2p at all).  ``MCDHFSolver._mean_field_matrices`` now caps
        #   the anchor at the parent configurations' maxima, which gives
        #   He 1s¹ (ε(2p) = −0.125 Ha, splitting 0.0469 Ha vs NIST 0.0514)
        #   and leaves Be unchanged (parents 1s²2s2p / 1s2s²2p, maxima
        #   2/2 = the reference occupations → 2p₁/₂ = −0.2286 Ha).
        # ------------------------------------------------------------------
        if mean_field_coeffs is None:
            if mean_field_normalization is not None:
                raise ValueError(
                    "mean_field_normalization was supplied without "
                    "mean_field_coeffs; the divisor has no numerator to "
                    "normalise (R7 seam)")
            self.mean_field_coeffs = None
            self.mean_field_normalization = None
        else:
            coeffs = np.asarray(mean_field_coeffs, dtype=np.float64)
            n_orb = len(subshells)
            if coeffs.shape != (n_orb, n_orb):
                raise ValueError(
                    "mean_field_coeffs must have shape "
                    f"({n_orb}, {n_orb}) to match subshells "
                    f"(got {coeffs.shape})")
            if not np.all(np.isfinite(coeffs)):
                raise ValueError(
                    "mean_field_coeffs contains non-finite entries")
            if mean_field_normalization is None:
                raise ValueError(
                    "mean_field_coeffs requires mean_field_normalization: "
                    "the per-target divisor that converts the raw "
                    "coefficient into the per-electron weight of the "
                    "target's Fock operator (R7 seam)")
            norm = np.asarray(mean_field_normalization, dtype=np.float64)
            if norm.shape != (n_orb,):
                raise ValueError(
                    "mean_field_normalization must have shape "
                    f"({n_orb},) to match subshells (got {norm.shape})")
            if not np.all(np.isfinite(norm)):
                raise ValueError(
                    "mean_field_normalization contains non-finite entries")
            if np.any(norm <= 0.0):
                raise ValueError(
                    "mean_field_normalization must be strictly positive; "
                    "a zero divisor would drop the two-electron term of "
                    "the target's Fock operator silently (R7 seam)")
            self.mean_field_coeffs = coeffs
            self.mean_field_normalization = norm
        self.ns = ns
        self.ks = ks
        # Point-Coulomb spinors have a noninteger r**gamma cusp. A
        # larger first cell leaves a basis error that increasing ns at
        # fixed first knot does not remove (K/Co cross-engine control).
        self.r_min = r_min if r_min is not None else (0.0004 if nuclear == 'point' else 0.05) / Z
        self.r_max = r_max if r_max is not None else max(40.0, 60.0 / Z)
        if exchange not in ("hf", "slater", "gu", "tfda", "gsz"):
            raise ValueError(
                f"exchange must be one of 'hf', 'slater', 'gu', "
                f"'tfda', 'gsz' (got {exchange!r})")
        self.exchange = exchange
        self.alpha_x = alpha_x
        self.fac_ihx = fac_ihx
        self.fac_hxs = fac_hxs
        # When exchange='gu', resolve Gu's α_xc(Z, N) once — it is
        # a function of (Z, total N) only and does not change during
        # the SCF iterations (this mirrors FAC's potential->ahx,
        # which is set once per OptimizeRadial call at radial.c:2348).
        #
        # Unit note: FAC's ``ahx`` lives in the ``V_x = -ahx · (4π·ρ)^(1/3)``
        # convention, while this code path uses Slater α in the
        # ``V_x = -α · (3·ρ/π)^(1/3)`` convention (see
        # ``_build_exchange_matrix_slater``). The two differ by a
        # constant factor ``(4π² / 3)^(1/3) ≈ 2.3609`` — multiply
        # FAC's ahx by this factor to recover the equivalent Slater
        # α.  See ``atomx.numerics.shooting.local.ShootingSolver._effective_alpha``
        # for the derivation.
        if exchange == "gu":
            import numpy as _np

            from atomx.physics.local_exchange import get_hxs
            n_total = float(sum(sh.occupation for sh in subshells))
            ahx = get_hxs(
                Z=float(Z), N=n_total,
                ihx=self.fac_ihx, hxs=self.fac_hxs,
            )
            self.alpha_x = ahx * (4.0 * _np.pi ** 2 / 3.0) ** (1.0 / 3.0)
        self.include_breit = include_breit
        self.include_qed = include_qed
        self.qed_se_method = qed_se_method
        self.include_wichmann_kroll = include_wichmann_kroll
        # Post-SCF corrections seam (review E1): a user-supplied
        # CorrectionStack wins; otherwise the legacy include_breit /
        # include_qed booleans are sugar that builds the equivalent stack.
        if correction_stack is not None:
            self.correction_stack = correction_stack
        elif include_breit or include_qed:
            from atomx.corrections.qed import resolve_se_method
            from atomx.corrections.stack import CorrectionStack
            self.correction_stack = CorrectionStack(
                se_backend=(resolve_se_method(qed_se_method, Z)
                            if include_qed else None),
                include_vp=bool(include_qed),
                include_wk_vp=bool(include_qed and include_wichmann_kroll),
                include_breit=bool(include_breit),
            )
        else:
            self.correction_stack = None
        self.nuclear = nuclear
        self.starting_potential = starting_potential
        self.N_electrons = sum(sh.occupation for sh in subshells)
        self._log = log  # CalculationLog instance or None

        # Nuclear potential function (None = point nucleus -Z/r)
        from atomx.potentials.nuclear import make_nuclear_potential, nuclear_profile
        self.nuclear_profile = nuclear_profile(Z, nuclear, nuclear_mass_number)
        self._nuclear_potential = make_nuclear_potential(Z, nuclear, nuclear_mass_number)
        # Electron-gas (plasma) screening: add delta_V(r) to the nuclear
        # potential everywhere it is sampled — the Galerkin Hamiltonian
        # (build_dirac_matrices) and the local-exchange density path
        # (_local_potential_*).  Keeping the nuclear model intact (the
        # shift is finite at the origin) is what lets the Fermi/uniform
        # charge distribution keep regularising the r -> 0 region.
        self.screening_potential = screening_potential
        if screening_potential is not None:
            base = self._nuclear_potential
            Z_f = float(Z)

            def _screened(r, _base=base, _shift=screening_potential,
                          _Z=Z_f):
                v = -_Z / r if _base is None else _base(r)
                return v + _shift(r)

            self._nuclear_potential = _screened

        self.t = make_knots(ns, ks, self.r_min, self.r_max)
        self.kinetic_balance = kinetic_balance
        self.bsp_data = prepare_bspline_data(
            self.t, ns, ks, kinetic_balance=kinetic_balance)
        self._precompute()

        # Optional warm-start: dict ``{kappa: [(P, Q, occ), ...]}`` matching the
        # current basis. Set via :attr:`seed_orbitals` (or left ``None`` for
        # hydrogenic + TFDA/GSZ cold start). Used by MCDHFSolver to reuse the
        # previous outer-iteration orbitals, cutting inner SCF iterations from
        # ~30 to ~3 once weights settle.
        self.seed_orbitals = None

    def _flatten_orbitals(self, orbitals):
        """Flatten orbitals dict to list of (P, Q, occ, kappa, label)."""
        orb_list = []
        kappa_counter = {}
        for sh in self.subshells:
            idx = kappa_counter.get(sh.kappa, 0)
            kappa_counter[sh.kappa] = idx + 1
            orbs = orbitals.get(sh.kappa, [])
            if idx < len(orbs):
                P, Q, occ = orbs[idx]
                orb_list.append((P, Q, occ, sh.kappa, sh.label))
        return orb_list

    def _precompute(self):
        """Pre-compute overlap and nuclear matrices."""
        nv = self.bsp_data["nv"]
        ks = self.ks
        ns = self.ns
        gr = self.bsp_data["gr"]
        grw = self.bsp_data["grw"]
        # Use Q-component basis for overlap/nuclear (larger basis)
        ksq = self.bsp_data.get("ksq", ks)
        bsp = self.bsp_data["bsp_q"] if "bsp_q" in self.bsp_data else self.bsp_data["bsp_p"]
        self.S_mat = overlap_matrix(nv, ksq, ns, bsp, grw)
        self.V_nuc_mat = nuclear_matrix(nv, ksq, ns, bsp, gr, grw, self.Z)
        # Reset convolution cache when grid changes
        self._convol_cache = ConvolCache(self.bsp_data)

    def _get_WW(self, i, j):
        """Get WW(i,j) — generalized occupation product for energy coefficients.

        With jj_configs: WW is the multi-configuration weighted average.
        Without: uses statistical weight formula for jj-average:
          w(n-,n+) = C(2l,n-)*C(2l+2,n+) / C(4l+2, N)
          WW(i,i) = Σ_c w(c) * q_c(i) * (q_c(i) - 1)
          WW(i,j) = Σ_c w(c) * q_c(i) * q_c(j)
        """
        if self.jj_configs is not None:
            from atomx.config.jj_coupling import compute_WW
            return compute_WW(i, j, self.jj_configs)
        # Single-config with fractional occupations: q*(q-1) is correct
        qi = self.subshells[i].occupation
        if i == j:
            return qi * (qi - 1.0)
        qj = self.subshells[j].occupation
        return qi * qj

    def _mean_field_weight(self, i, j):
        """Raw coefficient of source ``j`` in target ``i``'s potential (R7).

        Returns ``mean_field_coeffs[i, j]`` when the asymmetric seam was
        supplied, else :meth:`_get_WW` — the Guerra product
        ``WW(i, j) = Σ_c w_c ñ_c(i) ñ_c(j)``.

        The returned number is *unnormalised*: the direct and exchange
        builders divide the assembled target potential by
        :meth:`_mean_field_divisor`, so the physical per-electron weight
        of source ``j`` in target ``i``'s Fock operator is
        ``_mean_field_weight(i, j) / _mean_field_divisor(i)``.  For the
        Guerra default that ratio is ``WW(i, j)/q_i`` — the q_i-weighted
        average of ``q_j``, which is the correct mean field only while
        the target has a Guerra population.  A target with none has
        ``WW(i, ·) ≡ 0`` and would otherwise be solved against a bare
        nucleus, which is the R7 defect the seam exists to fix.
        """
        if self.mean_field_coeffs is not None:
            return float(self.mean_field_coeffs[i, j])
        return self._get_WW(i, j)

    def _mean_field_divisor(self, i):
        """Per-target divisor for the raw mean-field coefficient (R7).

        ``mean_field_normalization[i]`` when the seam was supplied, else
        ``subshells[i].occupation`` — the pre-seam convention.
        """
        if self.mean_field_normalization is not None:
            return float(self.mean_field_normalization[i])
        return float(self.subshells[i].occupation)

    def _flat_orbital_index(self, kappa, orb_idx):
        """Flat index of ``(kappa, orb_idx)`` in ``self.subshells``.

        Mirrors the ``kappa_counter`` bookkeeping used by the SCF loops
        (the ``orb_idx``-th entry of ``orbitals[kappa]`` is the
        ``orb_idx``-th subshell with that ``kappa``).  Returns ``-1``
        when the orbital is not one of ``self.subshells``.
        """
        seen = -1
        for idx, sh in enumerate(self.subshells):
            if sh.kappa == kappa:
                seen += 1
                if seen == orb_idx:
                    return idx
        return -1

    def _effective_mean_field_weight(self, i, j):
        """Per-electron weight of source ``j`` in target ``i`` (R7 seam).

        The quantity that multiplies ``(J_j - K_j)`` in target ``i``'s
        Fock operator.  Raises when asked for an orbital that is not in
        ``self.subshells`` — with an explicit seam in play there is no
        honest fallback coefficient to substitute.
        """
        for role, idx in (("target", i), ("source", j)):
            if idx < 0 or idx >= len(self.subshells):
                raise ValueError(
                    f"mean-field seam requested for {role} orbital index "
                    f"{idx}, outside self.subshells "
                    f"(n_orb={len(self.subshells)}); the supplied "
                    "coefficient matrix has no row/column for it (R7 seam)")
        return self._mean_field_weight(i, j) / self._mean_field_divisor(i)

    def _eval_P_on_grid(self, coeffs):
        """Evaluate P-component (large) orbital on grid using ksp-order basis."""
        ksp = self.bsp_data.get("ksp", self.ks)
        nsp = self.bsp_data.get("nsp", self.ns)
        return eval_orbital_on_grid(
            coeffs, self.bsp_data["bsp_p"],
            self.bsp_data["nv"], ksp, nsp)

    def _eval_Q_on_grid(self, coeffs):
        """Evaluate Q-component (small) orbital on grid using ksq-order basis."""
        ksq = self.bsp_data.get("ksq", self.ks)
        nsq = self.bsp_data.get("nsq", self.ns)
        bsp_q = self.bsp_data.get("bsp_q", self.bsp_data["bsp_p"])
        return eval_orbital_on_grid(coeffs, bsp_q, self.bsp_data["nv"], ksq, nsq)

    def _flatten_orbital_vector(self, orbitals, orbital_schedule, ns_full):
        """Flatten all orbital P,Q vectors into a single 1D array for DIIS."""
        n_orb = len(orbital_schedule)
        vec = np.zeros(n_orb * 2 * ns_full)
        for flat_i, (_sh, kappa, orb_idx) in enumerate(orbital_schedule):
            orb_list = orbitals.get(kappa, [])
            if orb_idx < len(orb_list):
                P, Q, _ = orb_list[orb_idx]
                offset = flat_i * 2 * ns_full
                vec[offset:offset + ns_full] = P[:ns_full]
                vec[offset + ns_full:offset + 2 * ns_full] = Q[:ns_full]
        return vec

    def _unflatten_orbital_vector(self, vec, orbitals, orbital_schedule, ns_full):
        """Restore orbitals from a flattened DIIS vector."""
        for flat_i, (_sh, kappa, orb_idx) in enumerate(orbital_schedule):
            orb_list = orbitals.get(kappa, [])
            if orb_idx < len(orb_list):
                _, _, occ = orb_list[orb_idx]
                offset = flat_i * 2 * ns_full
                P = vec[offset:offset + ns_full].copy()
                Q = vec[offset + ns_full:offset + 2 * ns_full].copy()
                orb_list[orb_idx] = (P, Q, occ)
                orbitals[kappa] = orb_list

    def _build_direct_potential(self, orbitals, target_kappa=None, target_idx=None):
        """Build Hartree potential via cell-based convolution (DBSR_HF algorithm).

        Uses rkb convolution for each orbital's F^0 contribution to the direct
        potential matrix. Self-consistent with cell-based energy computation.

        With a target, the weight of source ``j`` is the *effective* mean
        field ``_mean_field_weight(target, j) / _mean_field_divisor(target)``
        — ``WW(target, j)/q_target`` on the pre-seam path, and the R7
        asymmetric coefficient when the seam is supplied.

        Falls back to Y^k for DFS (Slater local exchange) mode, which doesn't
        benefit from cell algorithm precision.
        """
        # DFS mode: use fast Y^k approach (no cell-algorithm precision needed)
        if self.exchange != "hf":
            return self._build_direct_potential_yk(orbitals, target_kappa, target_idx)

        from atomx.numerics.bspline.convol import (
            _banded_to_full_add,
            _convol_case1,
            _convol_case1_mixed,
            _density_sym,
        )

        bd = self.bsp_data
        ksp = bd.get("ksp", bd["ks"])
        ksq = bd.get("ksq", bd["ks"])
        ns_full = bd["ns"]

        if not hasattr(self, '_convol_cache'):
            self._convol_cache = ConvolCache(bd)

        rkb_pp, _, _ = self._convol_cache.get_rkb(0, "pppp")
        rkb_qq, _, _ = self._convol_cache.get_rkb(0, "qqqq")
        rkb_pq, _, _ = self._convol_cache.get_rkb(0, "pqpq")
        rkb_qp, _, _ = self._convol_cache.get_rkb(0, "qpqp")

        # Accumulate weighted densities FIRST, then do ONE convol each
        # This is O(N_orb) density ops + 4 convol ops, instead of O(N_orb) × 4 convols
        dd_P = np.zeros((ns_full, ksp))  # accumulated P-density for all orbitals
        dd_Q = np.zeros((ns_full, ksq))  # accumulated Q-density

        # Build flat index map for the mean-field coefficients
        flat_idx = {}
        kappa_counter = {}
        for idx, sh in enumerate(self.subshells):
            k_idx = kappa_counter.get(sh.kappa, 0)
            kappa_counter[sh.kappa] = k_idx + 1
            flat_idx[(sh.kappa, k_idx)] = idx

        target_flat = flat_idx.get((target_kappa, target_idx), -1) if target_kappa is not None else -1
        target_q = self._mean_field_divisor(target_flat) if target_flat >= 0 else 1.0

        for kappa, orb_list in orbitals.items():
            for orb_i, (P_b, Q_b, occ_b) in enumerate(orb_list):
                j_flat = flat_idx.get((kappa, orb_i), -1)
                if target_flat >= 0 and j_flat >= 0:
                    coeff = self._mean_field_weight(target_flat, j_flat) / target_q
                else:
                    coeff = occ_b
                    if target_kappa == kappa and target_idx == orb_i:
                        coeff = occ_b - 1.0
                if abs(coeff) < 1e-10:
                    continue

                # Accumulate weighted symmetric densities
                d_P = _density_sym(P_b[:ns_full], P_b[:ns_full], ns_full, ksp)
                d_Q = _density_sym(Q_b[:ns_full], Q_b[:ns_full], ns_full, ksq)
                dd_P += coeff * d_P
                dd_Q += coeff * d_Q

        # 4 convolutions with accumulated densities (instead of 4 × N_orb)
        V_J_PP = np.zeros((ns_full, ns_full))
        V_J_QQ = np.zeros((ns_full, ns_full))

        # pppp(P-density) → PP
        v = _convol_case1(rkb_pp, dd_P, ns_full, ksp)
        _banded_to_full_add(V_J_PP, v, ns_full, ksp, 1.0)
        # pqpq(Q-density) → PP
        v = _convol_case1_mixed(rkb_pq, dd_Q, ns_full, ksp, ksq)
        _banded_to_full_add(V_J_PP, v, ns_full, ksp, 1.0)
        # qpqp(P-density) → QQ
        v = _convol_case1_mixed(rkb_qp, dd_P, ns_full, ksq, ksp)
        _banded_to_full_add(V_J_QQ, v, ns_full, ksq, 1.0)
        # qqqq(Q-density) → QQ
        v = _convol_case1(rkb_qq, dd_Q, ns_full, ksq)
        _banded_to_full_add(V_J_QQ, v, ns_full, ksq, 1.0)

        return V_J_PP, V_J_QQ

    def _build_direct_potential_yk(self, orbitals, target_kappa=None, target_idx=None):
        """Y^k-based Hartree potential (fast, for DFS mode)."""
        nv = self.bsp_data["nv"]
        ns = self.ns
        gr = self.bsp_data["gr"]
        grw = self.bsp_data["grw"]
        n_gauss = gr.shape[1]

        rho_total = np.zeros((nv, n_gauss))
        for _kappa, orb_list in orbitals.items():
            for P, Q, occ in orb_list:
                fP = self._eval_P_on_grid(P)
                fQ = self._eval_Q_on_grid(Q)
                rho_total += occ * (fP ** 2 + fQ ** 2)

        if target_kappa is not None and target_idx is not None:
            orb_list_t = orbitals.get(target_kappa, [])
            if target_idx < len(orb_list_t):
                P_self, Q_self, _ = orb_list_t[target_idx]
                fP_self = self._eval_P_on_grid(P_self)
                fQ_self = self._eval_Q_on_grid(Q_self)
                rho_total -= (fP_self ** 2 + fQ_self ** 2)

        yk = compute_yk(rho_total, gr, grw, nv, n_gauss, k=0)
        yk_weights = grw * yk

        ksp = self.bsp_data.get("ksp", self.ks)
        bsp_p = self.bsp_data["bsp_p"]
        V_J_PP = _assemble_integral(nv, ksp, ksp, ns, bsp_p, bsp_p, yk_weights)
        V_J_PP = 0.5 * (V_J_PP + V_J_PP.T)

        ksq = self.bsp_data.get("ksq", self.ks)
        bsp_q = self.bsp_data.get("bsp_q", bsp_p)
        V_J_QQ = _assemble_integral(nv, ksq, ksq, ns, bsp_q, bsp_q, yk_weights)
        V_J_QQ = 0.5 * (V_J_QQ + V_J_QQ.T)

        return V_J_PP, V_J_QQ

    def _build_exchange_matrix_hf(self, target_kappa, target_idx, orbitals,
                                   exchange_cache=None):
        """Build non-local HF exchange matrix via cell-based convolution.

        The convolution K^k_b (the exchange matrix from source orbital b at
        multipole order k) depends ONLY on the source orbital, NOT on the
        target.  So we cache it in exchange_cache and reuse across targets.

        exchange_cache key: (kappa_b, orb_i, k, block_type) -> matrix
        This gives ~N_target speedup since each source convolution is done once.

        Source weights: ``occ_b`` (the source's occupation from the
        orbital list) by default — the pre-seam convention of this path,
        which equals ``WW(target, b)/q_target`` for a closed shell.  When
        the R7 asymmetric seam is supplied both the direct potential and
        this exchange term are built from the same effective weight
        ``_mean_field_weight(target, b) / _mean_field_divisor(target)``,
        so a collapsed target really does see the reference
        configuration's field (and only that field).
        """
        from atomx.angular.coefficients import exchange_lambda
        from atomx.numerics.bspline.convol import _convol_case4, _density_full

        bd = self.bsp_data
        ksp = bd.get("ksp", self.ks)
        nsp = bd.get("nsp", self.ns)
        ksq = bd.get("ksq", self.ks)
        nsq = bd.get("nsq", self.ns)
        ns_full = bd.get("ns", self.ns)

        if not hasattr(self, '_convol_cache'):
            self._convol_cache = ConvolCache(bd)

        if exchange_cache is None:
            exchange_cache = {}

        use_seam = self.mean_field_coeffs is not None
        target_flat = self._flat_orbital_index(target_kappa, target_idx) if use_seam else -1

        tj_a = 2 * abs(target_kappa) - 1
        K_PP = np.zeros((ns_full, ns_full))
        K_QQ = np.zeros((ns_full, ns_full))
        K_PQ = np.zeros((ns_full, ns_full))
        K_QP = np.zeros((ns_full, ns_full))

        for kappa_b, orb_list in orbitals.items():
            tj_b = 2 * abs(kappa_b) - 1
            for orb_i, (P_b, Q_b, occ_b) in enumerate(orb_list):
                if kappa_b == target_kappa and orb_i == target_idx:
                    continue
                if use_seam:
                    weight = self._effective_mean_field_weight(
                        target_flat, self._flat_orbital_index(kappa_b, orb_i))
                    if abs(weight) < 1e-15:
                        continue
                else:
                    weight = occ_b
                k_values = exchange_k_range(target_kappa, kappa_b)
                for k in k_values:
                    lam = exchange_lambda(target_kappa, kappa_b, k)
                    if lam < 1e-15:
                        continue
                    c_k = weight * lam / ((tj_a + 1.0) * (tj_b + 1.0))

                    # Cache key: convolution result depends only on source + k
                    cache_key = (kappa_b, orb_i, k)

                    if cache_key not in exchange_cache:
                        # Compute and cache all 4 block convolutions for this source
                        rkb_pp, _, _ = self._convol_cache.get_rkb(k, "pppp")
                        d_P = _density_full(P_b[:nsp], P_b[:nsp], nsp)
                        Kk_PP = _convol_case4(rkb_pp, d_P, nsp, ksp)

                        rkb_qq, _, _ = self._convol_cache.get_rkb(k, "qqqq")
                        d_Q = _density_full(Q_b[:nsq], Q_b[:nsq], nsq)
                        Kk_QQ = _convol_case4(rkb_qq, d_Q, nsq, ksq)

                        rkb_pq, _, _ = self._convol_cache.get_rkb(k, "pqpq")
                        d_PQ = _density_full(P_b, Q_b, ns_full)
                        Kk_PQ = _convol_case4(rkb_pq, d_PQ, ns_full, ksp, ksq)

                        rkb_qp, _, _ = self._convol_cache.get_rkb(k, "qpqp")
                        d_QP = _density_full(Q_b, P_b, ns_full)
                        Kk_QP = _convol_case4(rkb_qp, d_QP, ns_full, ksq, ksp)

                        exchange_cache[cache_key] = (Kk_PP, Kk_QQ, Kk_PQ, Kk_QP)

                    Kk_PP, Kk_QQ, Kk_PQ, Kk_QP = exchange_cache[cache_key]

                    # Add with target-dependent coefficient (cheap multiply+add)
                    n = min(Kk_PP.shape[0], ns_full)
                    K_PP[:n, :n] += c_k * Kk_PP[:n, :n]
                    n = min(Kk_QQ.shape[0], ns_full)
                    K_QQ[:n, :n] += c_k * Kk_QQ[:n, :n]
                    K_PQ[:ns_full, :ns_full] += c_k * Kk_PQ[:ns_full, :ns_full]
                    K_QP[:ns_full, :ns_full] += c_k * Kk_QP[:ns_full, :ns_full]

        K_PP = 0.5 * (K_PP + K_PP.T)
        K_QQ = 0.5 * (K_QQ + K_QQ.T)
        K_PQ = 0.5 * (K_PQ + K_QP.T)
        K_QP = K_PQ.T.copy()
        return K_PP, K_QQ, K_PQ, K_QP

    def _precompute_angular_coefficients(self, orbital_schedule):
        """Precompute all angular coefficients before SCF loop.

        Builds (as dense ndarrays of shape ``(n_orb, n_orb, kmax+1)`` and
        list lookups) — direct access via ``arr[i, j, k]`` is ~3× faster
        than ``dict.get((i, j, k), 0.0)`` inside the SCF inner loop, which
        runs ~900k times per Pb SCF and previously showed up as the #1
        Python-overhead hotspot.

          _angular_direct[i, j, k]    — direct coefficient a(i,j,k)
          _angular_exchange[i, j, k]  — exchange coefficient b(i,j,k)
          _angular_k_values[i, j]     — tuple of k with nonzero direct OR
                                        exchange coefficient
          _kmax                       — maximum k needed
          _orb_kappas[i]              — kappa for flat orbital index i

        The ``dict`` attributes ``_angular_direct_dict`` / ``..._exchange_dict``
        are kept as back-compat aliases for any caller that still uses the
        (i,j,k) key form; the hot SCF paths read the ndarrays instead.
        """
        n_orb = len(self.subshells)
        kappas = [sh.kappa for sh in self.subshells]
        self._orb_kappas = kappas

        all_kappas = set(kappas)
        kmax = max(abs(ka) + abs(kb) for ka in all_kappas for kb in all_kappas)
        self._kmax = kmax

        direct = np.zeros((n_orb, n_orb, kmax + 1), dtype=np.float64)
        exchange = np.zeros((n_orb, n_orb, kmax + 1), dtype=np.float64)
        k_values: dict[tuple[int, int], tuple[int, ...]] = {}

        for i in range(n_orb):
            for j in range(n_orb):
                ks_nonzero = []
                for k in range(0, kmax + 1):
                    dc = self._get_direct_coef(i, j, k, kappas[i], kappas[j])
                    ec = self._get_exchange_coef(i, j, k, kappas[i], kappas[j])
                    if abs(dc) > 1e-15 or abs(ec) > 1e-15:
                        direct[i, j, k] = dc
                        exchange[i, j, k] = ec
                        ks_nonzero.append(k)
                k_values[(i, j)] = tuple(ks_nonzero)

        self._angular_direct = direct
        self._angular_exchange = exchange
        self._angular_k_values = k_values

        # Per-target: sorted tuple of k values with ANY nonzero coefficient
        # to any source. Lifts the ``all_k_set`` computation out of the SCF
        # inner loop — it's geometry-only and iteration-invariant.
        self._angular_k_for_target: list[tuple[int, ...]] = []
        for i in range(n_orb):
            ks = set()
            for j in range(n_orb):
                ks.update(k_values[(i, j)])
            self._angular_k_for_target.append(tuple(sorted(ks)))

    def _build_hf_matrix_dbsr(self, target_i, orbitals, skip_pq_exchange=False):
        """Build the Fock matrix for orbital i — DBSR_HF hf_matrix algorithm.

        Uses precomputed angular coefficients and in-place density accumulation
        to minimize redundant computation and memory allocation.

        For each (k, block_type):
          1. Accumulate direct density: dd = sum_j a(i,j,k) * density(j,j)
          2. Accumulate exchange density: xx = sum_j b(i,j,k) * density(i_comp,j_comp)
          3. ONE convolution each for direct and exchange accumulated density.

        The angular coefficients carry the occupation weights; the
        assembled result is divided by ``_mean_field_divisor(target_i)``
        below ("hfm = hfm / qsum(i)"), so the per-electron weight of
        source ``j`` in target ``i``'s Fock operator is
        ``_mean_field_weight(i, j) / _mean_field_divisor(i)``.  Without
        the R7 seam that is ``WW(i, j)/q_i`` (the Guerra mean field);
        with it, a target whose Guerra weight has collapsed gets the
        reference configuration's potential instead of the bare nucleus.

        Parameters
        ----------
        target_i : int
            Flat index of target orbital in self.subshells.
        orbitals : dict
            kappa -> [(P, Q, occ), ...]
        skip_pq_exchange : bool            When True, skip the pqpq exchange contraction and return
            ``hfm_PQ = 0``.  The batch SCF loop adds ``hfm_PQ`` only to the
            strictly upper-right (P, Q) block of H, and ``solve_dirac``
            calls ``scipy.linalg.eigh`` with the default ``lower=True`` —
            LAPACK then reads only the lower triangle, so that block never
            influences eigenvalues or eigenvectors.  Skipping it drops one
            of the four ``_convol_case4`` contractions per (k, block) at
            bit-identical SCF results.  Callers that consume ``hfm_PQ``
            (e.g. reference/validation paths) must keep the default False.

        Returns
        -------
        hfm_PP, hfm_QQ, hfm_PQ, hfm_QP : ndarray
            Combined direct + exchange Fock matrix blocks.
        """
        from atomx.numerics.bspline.convol import (
            SEMISEPARABLE_DIRECT_NS,
            SEMISEPARABLE_EXCHANGE_NS,
            SEMISEPARABLE_MIXED_EXCHANGE_NS,
            _accumulate_direct_density_batch,
            _accumulate_exchange_density_batch,
            _banded_to_full_add,
            _convol_case1_semiseparable_t,
            _convol_case1_t,
            _convol_case4_semiseparable_mixed_t,
            _convol_case4_semiseparable_t,
            _convol_case4_t,
        )

        bd = self.bsp_data
        ksp = bd.get("ksp", bd["ks"])
        ksq = bd.get("ksq", bd["ks"])
        ns_full = bd["ns"]

        if not hasattr(self, '_convol_cache'):
            self._convol_cache = ConvolCache(bd)

        # R7 seam: the divisor that turns the raw angular coefficients
        # (``_mean_field_weight``) into per-electron weights.  Identical
        # to ``subshells[target_i].occupation`` for every caller that has
        # not supplied ``mean_field_normalization``.
        norm_i = self._mean_field_divisor(target_i)

        # Stack orbital coefficients into contiguous (n_orb, ns_full) arrays
        # so the batched numba kernels below can iterate sources without any
        # Python-level loop. We also carry the flat ``q_j`` and ``kappa_j``
        # arrays for downstream code that still expects per-source access.
        n_orb = len(self.subshells)
        P_stack = np.zeros((n_orb, ns_full), dtype=np.float64)
        Q_stack = np.zeros((n_orb, ns_full), dtype=np.float64)
        kappa_counter = {}
        for flat_j, sh in enumerate(self.subshells):
            k_idx = kappa_counter.get(sh.kappa, 0)
            kappa_counter[sh.kappa] = k_idx + 1
            orb_list = orbitals.get(sh.kappa, [])
            if k_idx < len(orb_list):
                P_j, Q_j, _ = orb_list[k_idx]
                P_stack[flat_j, :ns_full] = P_j[:ns_full]
                Q_stack[flat_j, :ns_full] = Q_j[:ns_full]

        # Unique k values needed for this target orbital — precomputed in
        # ``_precompute_angular_coefficients`` since this is purely an
        # angular-geometry quantity and never changes across SCF iterations.
        all_k_set = self._angular_k_for_target[target_i]

        hfm_PP = np.zeros((ns_full, ns_full))
        hfm_QQ = np.zeros((ns_full, ns_full))
        hfm_PQ = np.zeros((ns_full, ns_full))
        hfm_QP = np.zeros((ns_full, ns_full))

        angular_direct = self._angular_direct      # (n, n, kmax+1) ndarray
        angular_exchange = self._angular_exchange  # (n, n, kmax+1) ndarray

        # Loop over block types: (ii, jj) = (1,1), (1,2), (2,1), (2,2)
        for ii, jj in [(1, 1), (1, 2), (2, 1), (2, 2)]:
            if ii == 1 and jj == 1:
                block = "pppp"
                ks_bra, ks_ket = ksp, ksp
            elif ii == 1 and jj == 2:
                block = "pqpq"
                ks_bra, ks_ket = ksp, ksq
            elif ii == 2 and jj == 1:
                block = "qpqp"
                ks_bra, ks_ket = ksq, ksp
            else:
                block = "qqqq"
                ks_bra, ks_ket = ksq, ksq

            comp_jj_stack = Q_stack if jj == 2 else P_stack
            comp_ii_stack = Q_stack if ii == 2 else P_stack

            for k in all_k_set:
                a_direct = angular_direct[target_i, :, k]     # length n_orb
                b_exchange = angular_exchange[target_i, :, k] # length n_orb

                # --- DIRECT: one numba call for the whole j-sum ---
                if np.any(a_direct):
                    dd = np.zeros((ns_full, ks_ket))
                    _accumulate_direct_density_batch(
                        dd, comp_jj_stack, a_direct, target_i, ns_full, ks_ket)
                    if ns_full >= SEMISEPARABLE_DIRECT_NS:
                        factors = self._convol_cache.get_coulomb_factors(
                            k, block)
                        v = _convol_case1_semiseparable_t(
                            *factors, dd, ns_full, ks_bra, ks_ket)
                    else:
                        rkbT, _, _ = self._convol_cache.get_rkb_t(k, block)
                        v = _convol_case1_t(
                            rkbT, dd, ns_full, ks_bra, ks_ket)
                    if ii == 1:
                        _banded_to_full_add(hfm_PP, v, ns_full, ks_bra, 1.0)
                    else:
                        _banded_to_full_add(hfm_QQ, v, ns_full, ks_bra, 1.0)

                # --- EXCHANGE: one numba call for the whole j-sum ---
                # The pqpq (ii=1, jj=2) contraction only feeds hfm_PQ; see
                # the ``skip_pq_exchange`` docstring for why the batch SCF
                # path can drop it without changing any solve_dirac output.
                if skip_pq_exchange and ii == 1 and jj == 2:
                    continue
                if np.any(b_exchange):
                    xx = np.zeros((ns_full, ns_full))
                    _accumulate_exchange_density_batch(
                        xx, comp_ii_stack, comp_jj_stack, b_exchange, ns_full)
                    if ii == jj and ns_full >= SEMISEPARABLE_EXCHANGE_NS:
                        near, rising, falling = (
                            self._convol_cache.get_symmetric_exchange_factors(
                                k, block))
                        Kk = _convol_case4_semiseparable_t(
                            near, rising, falling, xx, ns_full, ks_bra)
                    elif ii != jj and ns_full >= SEMISEPARABLE_MIXED_EXCHANGE_NS:
                        factors = self._convol_cache.get_coulomb_factors(
                            k, block)
                        Kk = _convol_case4_semiseparable_mixed_t(
                            *factors, xx, ns_full, ks_bra, ks_ket)
                    else:
                        rkbT, _, _ = self._convol_cache.get_rkb_t(k, block)
                        Kk = _convol_case4_t(
                            rkbT, xx, ns_full, ks_bra, ks_ket,
                            symmetric_output=(ii == jj))
                    if ii == 1 and jj == 1:
                        hfm_PP[:ns_full, :ns_full] += Kk[:ns_full, :ns_full]
                    elif ii == 1 and jj == 2:
                        hfm_PQ[:ns_full, :ns_full] += Kk[:ns_full, :ns_full]
                    elif ii == 2 and jj == 1:
                        hfm_QP[:ns_full, :ns_full] += Kk[:ns_full, :ns_full]
                    else:
                        hfm_QQ[:ns_full, :ns_full] += Kk[:ns_full, :ns_full]

        # Symmetrize
        hfm_PP = 0.5 * (hfm_PP + hfm_PP.T)
        hfm_QQ = 0.5 * (hfm_QQ + hfm_QQ.T)
        if skip_pq_exchange:
            hfm_PQ = hfm_QP.T.copy()
        else:
            hfm_PQ = 0.5 * (hfm_PQ + hfm_QP.T)
            hfm_QP = hfm_PQ.T.copy()

        # DBSR_HF: hfm = hfm / qsum(i)
        if norm_i > 1e-15:
            hfm_PP /= norm_i
            hfm_QQ /= norm_i
            hfm_PQ /= norm_i
            hfm_QP /= norm_i

        return hfm_PP, hfm_QQ, hfm_PQ, hfm_QP

    def _get_direct_coef(self, i, j, k, kappa_i, kappa_j):
        """Direct angular coefficient a(i,j,k) — DBSR_HF av_energy_coef.

        coef(max(i,j), min(i,j), k) for even k, within triangle rule.

        The occupation product is :meth:`_mean_field_weight` — ``WW(i,j)``
        for every caller that has not supplied the R7 asymmetric seam,
        so the closed-shell DBSR_HF agreement is untouched.  The
        assembled potential is divided by :meth:`_mean_field_divisor` in
        :meth:`_build_hf_matrix_dbsr` / :meth:`_build_direct_potential`.
        """
        from atomx.angular.coefficients import exchange_lambda
        tj_i = 2 * abs(kappa_i) - 1
        tj_j = 2 * abs(kappa_j) - 1

        if k % 2 != 0:
            return 0.0
        if k > (tj_i + tj_j) // 2:
            return 0.0

        if i == j:
            # Same shell: coef = C(i,i)/2 (the density accumulator
            # doubles the target's own contribution).
            c = self._mean_field_weight(i, j) / 2.0
            if k == 0:
                return c
            # k > 0: same-shell F^k
            lam = exchange_lambda(kappa_i, kappa_i, k)
            if abs(lam) < 1e-15:
                return 0.0
            return -c * lam / (tj_i * (tj_i + 1.0))
        else:
            # Cross-shell: only k=0 for direct
            if k != 0:
                return 0.0
            return self._mean_field_weight(i, j)

    def _get_exchange_coef(self, i, j, k, kappa_i, kappa_j):
        """Exchange angular coefficient b(i,j,k) — DBSR_HF av_energy_coef.

        coef(min(i,j), max(i,j), k), zero for i==j.  The occupation
        product is :meth:`_mean_field_weight` (see
        :meth:`_get_direct_coef`).
        """
        from atomx.angular.coefficients import exchange_lambda
        if i == j:
            return 0.0

        l_i = kappa_i if kappa_i > 0 else -kappa_i - 1
        l_j = kappa_j if kappa_j > 0 else -kappa_j - 1
        tj_i = 2 * abs(kappa_i) - 1
        tj_j = 2 * abs(kappa_j) - 1

        if (k + l_i + l_j) % 2 != 0:
            return 0.0
        if k < abs(tj_i - tj_j) // 2:
            return 0.0
        if k > (tj_i + tj_j) // 2:
            return 0.0

        ww = self._mean_field_weight(i, j)
        if abs(ww) < 1e-15:
            return 0.0
        lam = exchange_lambda(kappa_i, kappa_j, k)
        if abs(lam) < 1e-15:
            return 0.0
        return -ww * lam / ((tj_i + 1.0) * (tj_j + 1.0))

    def _build_exchange_matrix_slater(self, orbitals):
        """Build local Slater exchange — separate PP and QQ matrices for KB.

        ``V_x(r) = -α (3 ρ_vol(r) / π)^{1/3}`` uses the *volumetric* charge
        density ``ρ_vol = (P² + Q²) / (4π r²)``, not the radial density
        ``ρ_rad = P² + Q²`` which is normalised to ``∫ρ_rad dr = N``.

        Applies Latter's tail correction (PRA **46**, 1481 (1992); DHFS.f:970)
        so the total radial potential satisfies ``r V(r) ≤ -(1 + Z - N)``,
        the asymptote of a neutral atom.  Pure LDA decays too fast, leaving
        valence orbitals unbound — the common failure mode that made the
        B-spline Slater path produce ~2× too-deep totals for Ne.
        """
        nv = self.bsp_data["nv"]
        ns = self.ns
        gr = self.bsp_data["gr"]
        grw = self.bsp_data["grw"]
        n_gauss = gr.shape[1]

        rho_rad = np.zeros((nv, n_gauss))
        for _kappa, orb_list in orbitals.items():
            for P, Q, occ in orb_list:
                fP = self._eval_P_on_grid(P)
                fQ = self._eval_Q_on_grid(Q)
                rho_rad += occ * (fP ** 2 + fQ ** 2)

        yk = compute_yk(rho_rad, gr, grw, nv, n_gauss, 0)
        if self._nuclear_potential is None:
            V_nuc_grid = -self.Z / gr
        else:
            V_nuc_grid = np.empty_like(gr)
            for iv in range(nv):
                for m in range(n_gauss):
                    V_nuc_grid[iv, m] = self._nuclear_potential(gr[iv, m])

        if self.exchange == "gu":
            from atomx.numerics.bspline.local_potential import (
                local_central_potential_from_radial_density,
            )

            # local_potential spells Gu's model "fac" (FAC-parity
            # potential layer with FAC's ihx/hxs/csi parameters).
            local = local_central_potential_from_radial_density(
                Z=self.Z,
                rho_rad=rho_rad,
                gr=gr,
                grw=grw,
                n_total=self.N_electrons,
                exchange="fac",
                fac_ihx=self.fac_ihx,
                fac_hxs=self.fac_hxs,
                nuclear_grid=V_nuc_grid,
                fac_self_interaction_correction=True,
                latter_correction=True,
            )
            vx_corr = np.asarray(local["exchange"], dtype=float)
        else:
            rho_vol = rho_rad / (4.0 * np.pi * gr ** 2)
            vx = np.zeros_like(rho_vol)
            mask = rho_vol > 1e-30
            vx[mask] = -self.alpha_x * (3.0 / np.pi * rho_vol[mask]) ** (1.0 / 3.0)
            V_total = V_nuc_grid + yk + vx
            rvinf = -(1.0 + self.Z - self.N_electrons)
            cap_mask = gr * V_total > rvinf
            V_total_capped = np.where(cap_mask, rvinf / gr, V_total)
            vx_corr = V_total_capped - V_nuc_grid - yk
        vx_w = grw * vx_corr

        ksp = self.bsp_data.get("ksp", self.ks)
        bsp_p = self.bsp_data["bsp_p"]
        Vx_PP = _assemble_integral(nv, ksp, ksp, ns, bsp_p, bsp_p, vx_w)
        Vx_PP = 0.5 * (Vx_PP + Vx_PP.T)

        ksq = self.bsp_data.get("ksq", self.ks)
        bsp_q = self.bsp_data.get("bsp_q", bsp_p)
        Vx_QQ = _assemble_integral(nv, ksq, ksq, ns, bsp_q, bsp_q, vx_w)
        Vx_QQ = 0.5 * (Vx_QQ + Vx_QQ.T)

        return Vx_PP, Vx_QQ

    def _build_intra_shell_potential(self, target_kappa, target_idx, orbitals):
        """Same-shell F^k contribution for k>0 via cell-based convolution.

        For j=1/2 shells (s, p1/2), only k=0 is allowed -> returns zeros.
        For j>=1 (p3/2, d, f, ...), adds the intra-shell potential:
            coeff = -(q-1) * Lambda(k) / (2j * (2j+1))

        ``q - 1`` is the target's self-interaction-removed occupation,
        i.e. ``_mean_field_weight(i, i) / _mean_field_divisor(i)`` on the
        pre-seam path.  With the R7 seam it is taken from the supplied
        coefficient matrix, so a collapsed target whose reference
        occupation is zero carries no intra-shell field at all (a
        spectator virtual in a frozen potential).

        Uses cell-based convolution (rkb + icase=1) for consistency with
        the direct potential and exchange, avoiding Y^k/cell mismatch.
        """
        from atomx.angular.coefficients import exchange_lambda
        from atomx.numerics.bspline.convol import (
            _banded_to_full_add,
            _convol_case1,
            _density_sym,
        )

        bd = self.bsp_data
        ns_full = bd["ns"]
        tj_a = 2 * abs(target_kappa) - 1

        Vi_PP = np.zeros((ns_full, ns_full))
        Vi_QQ = np.zeros((ns_full, ns_full))

        if tj_a <= 1:
            return Vi_PP, Vi_QQ

        orb_list_t = orbitals.get(target_kappa, [])
        if target_idx >= len(orb_list_t):
            return Vi_PP, Vi_QQ

        P_a, Q_a, q_a = orb_list_t[target_idx]
        if self.mean_field_coeffs is None:
            if q_a * (q_a - 1.0) < 1e-15:
                return Vi_PP, Vi_QQ
            c_ii = q_a - 1.0
        else:
            i = self._flat_orbital_index(target_kappa, target_idx)
            c_ii = self._effective_mean_field_weight(i, i)
            if c_ii <= 0.0:
                return Vi_PP, Vi_QQ

        ksp = bd.get("ksp", bd["ks"])
        ksq = bd.get("ksq", bd["ks"])
        l_a = target_kappa if target_kappa > 0 else -target_kappa - 1

        if not hasattr(self, '_convol_cache'):
            self._convol_cache = ConvolCache(bd)

        for k in range(2, tj_a + 1):
            if (k + 2 * l_a) % 2 != 0:
                continue
            lam = exchange_lambda(target_kappa, target_kappa, k)
            if lam < 1e-15:
                continue
            coeff = -c_ii * lam / (tj_a * (tj_a + 1.0))

            rkb_pp, _, _ = self._convol_cache.get_rkb(k, "pppp")
            d_P = _density_sym(P_a[:ns_full], P_a[:ns_full], ns_full, ksp)
            v = _convol_case1(rkb_pp, d_P, ns_full, ksp)
            _banded_to_full_add(Vi_PP, v, ns_full, ksp, coeff)

            rkb_qq, _, _ = self._convol_cache.get_rkb(k, "qqqq")
            d_Q = _density_sym(Q_a[:ns_full], Q_a[:ns_full], ns_full, ksq)
            v = _convol_case1(rkb_qq, d_Q, ns_full, ksq)
            _banded_to_full_add(Vi_QQ, v, ns_full, ksq, coeff)
        return Vi_PP, Vi_QQ

    def _compute_total_energy(self, orbitals, orbital_energies=None, *, active_indices=None):
        """Compute total energy using direct Slater integral formula (DBSR_HF style).

        E = Σ_a q_a <a|h|a> + E_2e

        where E_2e uses DBSR_HF av_energy_coef formulas:
          same-shell:  q(q-1)/2 * [F^0(a,a) - Σ_{k>0} Lambda(k)/(2j*(2j+1)) * F^k(a,a)]
          cross-shell: q_a*q_b * [F^0(a,b) - Σ_k Lambda(k)/((2j_a+1)(2j_b+1)) * G^k(a,b)]

        The coefficients come from :meth:`_get_WW`, *not* from the R7
        asymmetric seam: this is the average-level energy functional
        (DBSR_HF ``av_energy_coef``), a single symmetric functional of
        the occupations.  The seam describes a per-target radial
        optimisation field, which is not the gradient of one functional
        — MCDHF takes its energies from the CI matrix over these
        orbitals, never from this number, so mixing the two conventions
        here would only corrupt the SCF's own energy bookkeeping.
        """
        from atomx.angular.coefficients import exchange_lambda

        orb_list = self._flatten_orbitals(orbitals)
        bsp_data = self.bsp_data

        # DFS (Slater local exchange) used to evaluate energy with an LDA-based
        # formula that double-counts self-interaction; shooting's Salvat-style
        # path instead converges orbitals with local V_x but evaluates the
        # total energy with the configuration-average HF formula (F^0 + G^k)
        # on those orbitals (DHFS.f:984 _expval).  We match that — converged
        # Slater orbitals are close enough to HF that the HF energy functional
        # is the right target and agrees with shooting at 10⁻³.

        # One-electron: Σ q_a <a|h|a>
        ns_e = bsp_data.get("ns", self.ns)  # max(nsp, nsq)
        E_1e = 0.0
        h0_memo = dict(getattr(self, "_energy_h0_matrices", {}))
        for index, (P, Q, occ, kappa, _label) in enumerate(orb_list):
            if active_indices is not None and index not in active_indices:
                continue
            if kappa not in h0_memo:
                h0_memo[kappa] = build_dirac_matrices(
                    ns_e, kappa, self.Z, bsp_data, self._nuclear_potential)
            H0, S0 = h0_memo[kappa]
            vec = np.concatenate([P[:ns_e], Q[:ns_e]])
            norm_sq = vec @ S0 @ vec
            if norm_sq > 1e-20:
                E_1e += occ * (vec @ H0 @ vec) / norm_sq

        # Two-electron (DBSR_HF av_energy_coef formula)
        # Track direct (E_J = F^0) and exchange (E_K = F^{k>0}, G^k) separately
        # Use cell-based integrals for consistency with DBSR_HF.
        # Reuse the solver-level ConvolCache: the rkb tables depend only on
        # the B-spline geometry (bsp_data), never on the orbitals, so the
        # tables built during the SCF loop are bit-identical to fresh ones —
        # rebuilding them here cost ~0.3 s per solve (~20% of an Fe XVII DHF).
        if not hasattr(self, '_convol_cache'):
            self._convol_cache = ConvolCache(bsp_data)
        convol_cache = self._convol_cache
        def _fk(Pa, Qa, Pb, Qb, k):
            return compute_slater_fk_cell(
                Pa, Qa, Pb, Qb, bsp_data, k, convol_cache)

        def _gk(Pa, Qa, Pb, Qb, k):
            return compute_slater_gk_cell(
                Pa, Qa, Pb, Qb, bsp_data, k, convol_cache)

        # Two-electron coefficients use WW function for multi-config average
        E_J = 0.0   # direct Coulomb (F^0 terms)
        E_K = 0.0   # exchange (F^{k>0} and G^k terms)
        for i, (Pa, Qa, _oa, ka, _la) in enumerate(orb_list):
            tj_a = 2 * abs(ka) - 1
            l_a = ka if ka > 0 else -ka - 1
            for j, (Pb, Qb, _ob, kb, _lb) in enumerate(orb_list):
                if j > i:
                    continue
                if active_indices is not None and i not in active_indices and j not in active_indices:
                    continue
                tj_b = 2 * abs(kb) - 1

                if i == j:
                    # Same shell: WW(i,i)/2 * [F^0 - Σ_{k>0} Λ/(2j*(2j+1)) * F^k]
                    c = self._get_WW(i, i) / 2.0
                    if abs(c) < 1e-15:
                        continue
                    E_J += c * _fk(Pa, Qa, Pa, Qa, 0)
                    for k in range(1, tj_a + 1):
                        if (k + 2 * l_a) % 2 != 0:
                            continue
                        lam = exchange_lambda(ka, ka, k)
                        if lam < 1e-15:
                            continue
                        fk = _fk(Pa, Qa, Pa, Qa, k)
                        E_K += c * lam / (tj_a * (tj_a + 1.0)) * fk
                else:
                    # Cross-shell: WW(i,j) * F^0 - WW(i,j) * Σ_k ... * G^k
                    ww = self._get_WW(i, j)
                    if abs(ww) < 1e-15:
                        continue
                    E_J += ww * _fk(Pa, Qa, Pb, Qb, 0)
                    for k in exchange_k_range(ka, kb):
                        lam = exchange_lambda(ka, kb, k)
                        if lam < 1e-15:
                            continue
                        gk = _gk(Pa, Qa, Pb, Qb, k)
                        E_K += ww * lam / ((tj_a + 1.0) * (tj_b + 1.0)) * gk

        E_2e = E_J - E_K
        E_total = E_1e + E_2e
        components = {"E_1e": E_1e, "E_2e": E_2e, "E_J": E_J, "E_K": E_K}
        return E_total, components

    def _final_energy_checked(self, orbitals, orbital_energies,
                              max_retries=2):
        """Evaluate the final total energy with a NaN tripwire (HYP-1590).

        Under extreme CPU oversubscription a single serial energy-kernel
        call has been observed to return NaN from finite inputs while an
        immediate identical re-evaluation recovers (reproducer: concurrent
        he_dhf solves at box load > 150).  Retry the deterministic
        evaluation a few times; if the corruption persists, raise loudly
        rather than returning a ``converged=True`` result whose
        ``total_energy`` is NaN — a silent NaN here previously surfaced
        as flaky failures in the he_dhf / pair_ladder test families.
        """
        last = None
        for _attempt in range(max_retries + 1):
            total_energy, energy_components = self._compute_total_energy(
                orbitals, orbital_energies)
            if np.isfinite(total_energy):
                return total_energy, energy_components
            last = energy_components
        raise FloatingPointError(
            "DFSolver: total energy is non-finite on "
            f"{max_retries + 1} consecutive evaluations "
            f"(components={last}) — transient numerical corruption did "
            "not recover (HYP-1590)."
        )

    @staticmethod
    def _orthonormalize_orbitals(orbitals, metrics):
        """Symmetric metric retraction after seed loading or extrapolation."""
        for kappa, shells in orbitals.items():
            if not shells:
                continue
            vectors = np.column_stack([np.concatenate((p, q)) for p, q, _ in shells])
            gram = vectors.T @ metrics[kappa] @ vectors
            eigenvalues, rotations = np.linalg.eigh(gram)
            if np.min(eigenvalues) < 1e-12:
                raise ValueError("same-kappa orbitals are linearly dependent")
            vectors = vectors @ ((rotations / np.sqrt(eigenvalues)) @ rotations.T)
            ns = len(shells[0][0])
            orbitals[kappa] = [(vectors[:ns, i].copy(), vectors[ns:, i].copy(), occ)
                               for i, (_, _, occ) in enumerate(shells)]

    def _canonicalize_closed_subshells(self, orbitals, orbital_energies, kappa_groups):
        """Fix the occupied-subspace gauge without changing its determinant."""
        ns_full = self.bsp_data["ns"]
        for kappa, shells in orbitals.items():
            closed = [i for i, (_, _, q) in enumerate(shells)
                      if abs(q - 2 * abs(kappa)) < 1e-12]
            if len(closed) < 2:
                continue
            coefficients = np.column_stack([np.concatenate(shells[i][:2]) for i in closed])
            lagrange = np.zeros((len(closed), len(closed)))
            for j, shell_index in enumerate(closed):
                flat_j = self._flat_orbital_index(kappa, shell_index)
                pp, qq, _, qp = self._build_hf_matrix_dbsr(
                    flat_j, orbitals, skip_pq_exchange=True)
                fock = self._energy_h0_matrices[kappa][0].copy()
                fock[:ns_full, :ns_full] += pp
                fock[ns_full:, ns_full:] += qq
                fock[ns_full:, :ns_full] += qp
                fock = np.tril(fock) + np.tril(fock, -1).T
                lagrange[:, j] = coefficients.T @ (fock @ coefficients[:, j])
            eps, rotation = np.linalg.eigh(0.5 * (lagrange + lagrange.T))
            coefficients = coefficients @ rotation
            for j, shell_index in enumerate(closed):
                shells[shell_index] = (coefficients[:ns_full, j], coefficients[ns_full:, j], shells[shell_index][2])
                orbital_energies[kappa_groups[kappa][shell_index].label] = float(eps[j])

    def _optimize_same_kappa_rotations(self, orbitals, *, core_hole_stationary=True):
        """Minimize each occupied-pair rotation of the HF energy exactly.

        A two-orbital rotation makes a quartic trigonometric polynomial.
        Four analytic derivatives determine its nonconstant coefficients;
        the stationary points follow from a quartic in exp(2j*theta).
        Each Fock matrix is linear in the one-particle density and hence
        contains only constant, cos(2*theta), and sin(2*theta) terms. Three
        Fock samples per target determine all four derivatives. Projecting
        onto the original orbital pair before interpolation keeps the work
        in 2x2 matrices. The mixed exchange blocks are mutual transposes.
        """
        maximum = 0.0
        virtual_keys = getattr(self, "virtual_orbital_keys", set())
        angles = np.arange(4) * (np.pi / 8)
        design = np.column_stack((-2*np.sin(2*angles), 2*np.cos(2*angles),
                                  -4*np.sin(4*angles), 4*np.cos(4*angles)))
        for kappa, shells in orbitals.items():
            for i in range(len(shells)):
                for j in range(i):
                    pi, qi, ni = shells[i]
                    pj, qj, nj = shells[j]
                    capacity = 2 * abs(kappa)
                    if abs(ni - capacity) < 1e-12 and abs(nj - capacity) < 1e-12:
                        continue
                    # A vacancy in the lower-n subshell makes this a
                    # constrained excited configuration.  Global pair-energy
                    # minimization can exchange radial identities even when
                    # both subshells have equal occupation (He 1s1 4s1),
                    # leaving labels on the wrong roots or oscillating.
                    # Ground-state pairs with a filled lower subshell retain
                    # their variational rotation.
                    n_i = self.subshells[self._flat_orbital_index(kappa, i)].n
                    n_j = self.subshells[self._flat_orbital_index(kappa, j)].n
                    lower_occupation = ni if n_i < n_j else nj
                    core_hole = n_i != n_j and lower_occupation < capacity - 1e-12
                    if core_hole and not core_hole_stationary:
                        continue
                    if ((self.subshells[self._flat_orbital_index(kappa, i)].n, kappa) in virtual_keys
                            or (self.subshells[self._flat_orbital_index(kappa, j)].n, kappa) in virtual_keys):
                        continue

                    def rotate(angle, shells=shells, i=i, j=j, pi=pi, qi=qi,
                               pj=pj, qj=qj, ni=ni, nj=nj):
                        c, t = np.cos(angle), np.sin(angle)
                        shells[i] = (c*pi + t*pj, c*qi + t*qj, ni)
                        shells[j] = (c*pj - t*pi, c*qj - t*qi, nj)

                    pair_basis = np.column_stack((
                        np.concatenate((pi, qi)), np.concatenate((pj, qj))))
                    n = self.bsp_data["ns"]
                    large, small = pair_basis[:n], pair_basis[n:]
                    h0_projected = pair_basis.T @ (
                        self._energy_h0_matrices[kappa][0] @ pair_basis)
                    samples = np.empty((2, 3, 2, 2))
                    for sample_i, sample_angle in enumerate((0.0, np.pi / 4,
                                                              np.pi / 2)):
                        rotate(sample_angle)
                        for target_i, index in enumerate((i, j)):
                            pp, qq, _, qp = self._build_hf_matrix_dbsr(
                                self._flat_orbital_index(kappa, index), orbitals,
                                skip_pq_exchange=True)
                            cross = small.T @ (qp @ large)
                            samples[target_i, sample_i] = (
                                h0_projected + large.T @ (pp @ large)
                                + small.T @ (qq @ small) + cross + cross.T)

                    constant = 0.5 * (samples[:, 0] + samples[:, 2])
                    cosine = 0.5 * (samples[:, 0] - samples[:, 2])
                    sine = samples[:, 1] - constant

                    def derivative(angle, constant=constant, cosine=cosine,
                                   sine=sine, ni=ni, nj=nj):
                        c, s = np.cos(angle), np.sin(angle)
                        vi = np.array((c, s))
                        vj = np.array((-s, c))
                        fields = (constant + cosine * np.cos(2 * angle)
                                  + sine * np.sin(2 * angle))
                        return 2 * (ni * (vj @ fields[0] @ vi)
                                    - nj * (vi @ fields[1] @ vj))

                    c1, c2, c3, c4 = np.linalg.solve(design, [derivative(t) for t in angles])
                    roots = np.roots([c4 + 1j*c3, (c2 + 1j*c1)/2,
                                      0, (c2 - 1j*c1)/2, c4 - 1j*c3])
                    stationary = np.angle(roots[abs(abs(roots)-1) < 1e-6])/2
                    candidates = np.r_[0., stationary]
                    values = (c1*np.cos(2*candidates) + c2*np.sin(2*candidates)
                              + c3*np.cos(4*candidates) + c4*np.sin(4*candidates))
                    if core_hole:
                        # Experimental local stationarity for excited states:
                        # follow the nearest stationary point, including a
                        # saddle, rather than globally exchanging radial roots.
                        # The trust radius prevents a single large label swap.
                        angle = (float(np.clip(stationary[np.argmin(abs(stationary))],
                                               -0.1, 0.1)) if stationary.size else 0.0)
                    else:
                        angle = float(candidates[np.argmin(values)])
                    rotate(angle)
                    maximum = max(maximum, abs(angle))
        return maximum

    def solve(self, max_iter=100, tol=1e-6, damping=None,
              mode="batch", damping_release_residual=1e-2):
        """Run the SCF iteration.

        Parameters
        ----------
        max_iter : int
        tol : float
            Convergence tolerance on orbital change (relative).
        damping : float or None
            Orbital damping factor (0 = full replacement, 0.5 = heavy).
            None = auto (0.3 for f-electrons, 0.0 otherwise).
        damping_release_residual : float
            Residual below which an explicitly damped update switches back
            to full replacement. Lower values can stabilize difficult SCF
            branches near convergence.
        mode : str
            "batch" (default): DBSR_HF hf_matrix algorithm — for each orbital,
            accumulate density across ALL sources before convolution (one convol
            per k per block_type).  Uses doubled self-coefficient for direct,
            zero self-exchange.  Fastest mode.

            "orbital": Per-orbital mode with per-orbital self-interaction
            subtraction and cached exchange convolutions.
        """
        self._calculation_controls = dict(max_iter=max_iter, tolerance=tol,
            damping=damping, mode=mode, damping_release_residual=damping_release_residual)
        if damping is None:
            has_f = any(sh.l >= 3 for sh in self.subshells)
            # f-electrons need damping to avoid oscillation
            damping = 0.3 if has_f else 0.0

        ns = self.ns

        # Build ordered list of (subshell, kappa, index_within_kappa)
        kappa_counter = {}
        orbital_schedule = []  # [(subshell, kappa, idx)]
        for sh in self.subshells:
            idx = kappa_counter.get(sh.kappa, 0)
            kappa_counter[sh.kappa] = idx + 1
            orbital_schedule.append((sh, sh.kappa, idx))

        # Initialize with nuclear-only potential (hydrogenic start)
        orbitals = {}  # kappa -> [(P, Q, occ), ...]
        orbital_energies = {}

        kappa_groups = {}
        for sh in self.subshells:
            kappa_groups.setdefault(sh.kappa, []).append(sh)

        has_kb = self.bsp_data.get("ksp", self.ks) != self.bsp_data.get("ksq", self.ks)
        nsp_init = self.bsp_data.get("nsp", ns)
        nsq_init = self.bsp_data.get("nsq", ns)

        # Starting potential: TFDA or GSZ provides better initial orbitals
        V_start_PP, V_start_QQ = None, None
        if self.starting_potential == "tfda" or self.exchange == "tfda":
            from atomx.potentials.thomas_fermi import tfda_potential_matrix
            V_start_PP, V_start_QQ = tfda_potential_matrix(
                self.Z, self.N_electrons, self.bsp_data)
        elif self.starting_potential == "gsz" or self.exchange == "gsz":
            from atomx.potentials.parametric import gsz_potential_matrix
            V_start_PP, V_start_QQ = gsz_potential_matrix(
                self.Z, self.N_electrons, self.bsp_data)

        ns_full_init = max(nsp_init, nsq_init)
        # Warm-start path: if caller seeded orbitals (and basis dim matches),
        # skip the hydrogenic/TFDA one-electron solve entirely. This is used by
        # the MCDHF outer loop so each outer iteration starts from the previous
        # iteration's converged orbitals, cutting inner SCF iterations by ~10×.
        seeded = False
        if self.seed_orbitals is not None:
            ok = True
            for kappa in kappa_groups:
                seed_list = self.seed_orbitals.get(kappa, [])
                if not seed_list or len(seed_list) < len(kappa_groups[kappa]):
                    ok = False
                    break
                for P, Q, _occ in seed_list:
                    if (np.asarray(P).shape[0] != ns_full_init
                            or np.asarray(Q).shape[0] != ns_full_init):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                for kappa, shells in kappa_groups.items():
                    seed_list = self.seed_orbitals[kappa]
                    orb_list = []
                    for i, sh in enumerate(shells):
                        P, Q, _occ = seed_list[i]
                        orb_list.append((np.asarray(P).copy(),
                                         np.asarray(Q).copy(),
                                         sh.occupation))
                    orbitals[kappa] = orb_list
                seeded = True

        if not seeded:
            for kappa, shells in kappa_groups.items():
                H, S = build_dirac_matrices(ns_full_init, kappa, self.Z, self.bsp_data, self._nuclear_potential)
                # Add starting potential if available
                if V_start_PP is not None:
                    n_vp = min(V_start_PP.shape[0], ns_full_init)
                    H[:n_vp, :n_vp] += V_start_PP[:n_vp, :n_vp]
                if V_start_QQ is not None:
                    n_vq = min(V_start_QQ.shape[0], ns_full_init)
                    H[ns_full_init:ns_full_init+n_vq, ns_full_init:ns_full_init+n_vq] += V_start_QQ[:n_vq, :n_vq]
                result = solve_dirac(H, S, ns_full_init, kappa,
                                      nsp=nsp_init if has_kb else None,
                                      nsq=nsq_init if has_kb else None)
                orb_list = []
                for sh in shells:
                    idx = sh.n - sh.l - 1
                    if kappa > 0 and not has_kb:
                        idx += 1
                    if idx < result.n_bound:
                        P, Q, E = result.get_orbital(idx)
                        if (np.isnan(E) or np.any(np.isnan(P))
                                or np.any(np.isnan(Q))):
                            # HYP-1590 (review HYP-1608): a poisoned
                            # hydrogenic seed must fail loud, not be
                            # skipped — orbital indexing is positional
                            # within each kappa group, so a bare skip
                            # would shift every same-kappa orbital one
                            # slot, and the parametric (tfda/gsz) path
                            # below has no SCF sweep to rebuild anything.
                            raise FloatingPointError(
                                "DFSolver: non-finite hydrogenic seed "
                                f"orbital for {sh.label} (kappa={kappa}) — "
                                "the initial eigensolve produced NaN and "
                                "the seed cannot be rebuilt positionally "
                                "(HYP-1590)."
                            )
                        orb_list.append((P, Q, sh.occupation))
                        orbital_energies[sh.label] = E
                orbitals[kappa] = orb_list

        # For parametric potentials (TFDA, GSZ), no SCF iteration needed --
        # the initial orbitals ARE the final orbitals
        if self.exchange in ("tfda", "gsz"):
            # HYP-1590 (review HYP-1608): route through the same final-energy
            # tripwire as the SCF path — never deliver converged=True with a
            # non-finite total energy from the parametric path either.
            total_energy, energy_components = self._final_energy_checked(
                orbitals, orbital_energies)
            return self._finalize_with_corrections(SCFResult(
                converged=True, iterations=1, residual=0.0,
                total_energy=total_energy, orbital_energies=orbital_energies,
                subshells=self.subshells, energy_components=energy_components,
                orbitals_raw=_canonicalized_orbitals_raw(orbitals),
                Z=self.Z,
                _bsp_data=self.bsp_data))

        converged = False
        E_old = 0.0
        residual = 1.0
        #: Subshells for which the *current* iteration found no bound state.
        #: Cleared each sweep; a non-empty set after the last sweep means the
        #: requested configuration is not bound in this potential and the
        #: result must not be reported as converged (finding C4).
        missing_shells: set[str] = set()
        has_kb = self.bsp_data.get("ksp", self.ks) != self.bsp_data.get("ksq", self.ks)
        nsp = self.bsp_data.get("nsp", ns)
        nsq = self.bsp_data.get("nsq", ns)
        ns_full = max(nsp, nsq)

        # Cache one-electron matrices per kappa (they never change)
        H0_cache = {}
        S0_cache = {}
        for kappa in kappa_groups:
            H0, S0 = build_dirac_matrices(ns_full_init, kappa, self.Z, self.bsp_data, self._nuclear_potential)
            H0_cache[kappa] = H0
            S0_cache[kappa] = S0

        self._energy_h0_matrices = {k: (H0_cache[k], S0_cache[k]) for k in H0_cache}
        self._orthonormalize_orbitals(orbitals, S0_cache)

        # Precompute angular coefficients — these depend only on kappa values
        # and never change during SCF iteration.  Build a lookup table once.
        if self.exchange == "hf":
            self._precompute_angular_coefficients(orbital_schedule)

        if self._log:
            self._log.scf_start()

        # Potential mixing state: store previous direct potential for blending
        V_J_PP_prev = None
        V_J_QQ_prev = None

        # DIIS state: orbital vector history and error vectors
        diis_max = 8
        diis_orb_history = []   # list of orbital snapshots (flat vectors)
        diis_err_history = []   # list of error vectors (orbital change)
        noninteracting = (mode == "batch" and self.exchange == "hf"
                          and not np.any(self._angular_direct)
                          and not np.any(self._angular_exchange))

        for iteration in range(1, max_iter + 1):
            missing_shells.clear()
            max_orb_change = 0.0

            if mode == "batch" and self.exchange == "hf":
                # ============================================================
                # DBSR_HF hf_matrix algorithm with DIIS acceleration.
                # ============================================================
                use_diis = len(orbital_schedule) >= 5 and iteration >= 4
                sweep_input = (self._flatten_orbital_vector(
                    orbitals, orbital_schedule, ns_full) if use_diis else None)
                for flat_i, (sh, kappa, orb_idx) in enumerate(orbital_schedule):
                    # Build DBSR_HF-style Fock matrix for orbital i.
                    # skip_pq_exchange: V_PQ lands in the strictly-upper
                    # (P, Q) block of H, which eigh (lower=True) never reads
                    # — skipping its contraction is bit-identical and saves
                    # 1/4 of the exchange convolutions.
                    V_PP, V_QQ, V_PQ, V_QP = self._build_hf_matrix_dbsr(
                        flat_i, orbitals, skip_pq_exchange=True)

                    H = H0_cache[kappa].copy()
                    S_full = S0_cache[kappa]

                    # Add two-electron potential (direct + exchange combined)
                    n = min(V_PP.shape[0], ns_full)
                    H[:n, :n] += V_PP[:n, :n]
                    H[ns_full:ns_full+n, ns_full:ns_full+n] += V_QQ[:n, :n]
                    n_pq = min(V_PQ.shape[0], ns_full)
                    H[:n_pq, ns_full:ns_full+n_pq] += V_PQ[:n_pq, :n_pq]
                    H[ns_full:ns_full+n_pq, :n_pq] += V_QP[:n_pq, :n_pq]

                    virtual_keys = getattr(self, "virtual_orbital_keys", set())
                    target_virtual = (sh.n, kappa) in virtual_keys
                    other_orbitals = [np.concatenate((P, Q))
                                      for j, (P, Q, _) in enumerate(orbitals.get(kappa, []))
                                      if j != orb_idx and (target_virtual or
                                          (kappa_groups[kappa][j].n, kappa) not in virtual_keys)]
                    constraints = (np.column_stack(other_orbitals)
                                   if other_orbitals and not noninteracting else None)
                    result = solve_dirac(H, S_full, ns_full, kappa,
                                          nsp=nsp if has_kb else None,
                                          nsq=nsq if has_kb else None,
                                          orthogonal_to=constraints)

                    eig_idx = sh.n - sh.l - 1
                    if kappa > 0 and not has_kb:
                        eig_idx += 1
                    if constraints is not None:
                        # Lower principal roots are already represented by
                        # the constrained orbitals. Follow the requested
                        # node/root index in the remaining spectrum; maximum
                        # overlap can lock a diffuse valence electron onto
                        # a higher excited root during the initial screening.
                        removed_lower = sum(
                            other.n < sh.n and (target_virtual or
                                (other.n, kappa) not in virtual_keys)
                            for other in kappa_groups[kappa])
                        eig_idx = max(0, eig_idx - removed_lower)
                    if eig_idx >= result.n_bound:
                        # No genuine bound state for this subshell in this
                        # potential.  Record it: appending zeros here (as
                        # the other branch does) would silently produce an
                        # empty, zero-energy "converged" result.
                        missing_shells.add(sh.label)
                        continue

                    P_new, Q_new, E_new = result.get_orbital(eig_idx)
                    if (np.isnan(E_new) or np.any(np.isnan(P_new))
                            or np.any(np.isnan(Q_new))):
                        # HYP-1590: a NaN eigenvector with a finite
                        # eigenvalue poisons every downstream convolution
                        # just as surely — skip the update either way.
                        continue

                    # Track change + sign fix
                    orb_list = orbitals.get(kappa, [])
                    while orb_idx >= len(orb_list):
                        orb_list.append((np.zeros(ns_full), np.zeros(ns_full), sh.occupation))
                    P_old, Q_old, _ = orb_list[orb_idx]
                    p_max = np.max(np.abs(P_old))
                    if p_max > 1e-15:
                        # DBSR_HF sign convention: match sign of largest component
                        overlap = np.dot(P_new, P_old)
                        if overlap < 0:
                            P_new = -P_new
                            Q_new = -Q_new
                        change = np.max(np.abs(P_new[:len(P_old)] - P_old[:len(P_new)])) / p_max
                        max_orb_change = max(max_orb_change, change)

                    # Damping: mix new eigenvector with old
                    # alpha = fraction of new orbital to keep
                    if damping > 0 and iteration > 1 and p_max > 1e-15:
                        alpha = 1.0 - damping
                        # Reduce damping as convergence progresses
                        if residual < damping_release_residual:
                            alpha = 1.0  # no damping: let it converge freely
                        elif residual < 1e-1:
                            alpha = min(1.0, alpha + 0.2)
                        P_mixed = alpha * P_new + (1.0 - alpha) * P_old
                        Q_mixed = alpha * Q_new + (1.0 - alpha) * Q_old
                        norm_new = np.sqrt(np.sum(P_new**2) + np.sum(Q_new**2))
                        norm_mix = np.sqrt(np.sum(P_mixed**2) + np.sum(Q_mixed**2))
                        if norm_mix > 1e-15 and norm_new > 1e-15:
                            P_mixed *= norm_new / norm_mix
                            Q_mixed *= norm_new / norm_mix
                        P_new, Q_new = P_mixed, Q_mixed

                    # Update orbital immediately (DBSR_HF: Put_pv + Update_int)
                    orb_list[orb_idx] = (P_new, Q_new, sh.occupation)
                    orbitals[kappa] = orb_list
                    orbital_energies[sh.label] = E_new

                # --- DIIS extrapolation at end of iteration ---
                # Accelerate convergence via Pulay DIIS on orbital vectors.
                # Collect history always (for 5+ orbital systems after iter 4),
                # but only extrapolate when residual is in a useful range.
                if use_diis:
                    orb_vec = self._flatten_orbital_vector(orbitals, orbital_schedule, ns_full)
                    # The previous stored output is not this sweep's input:
                    # DIIS, orthonormalization, and pair rotations can all
                    # change it. Use F(x)-x, including the first history entry.
                    err_vec = orb_vec - sweep_input
                    diis_orb_history.append(orb_vec.copy())
                    diis_err_history.append(err_vec.copy())
                    if len(diis_orb_history) > diis_max:
                        diis_orb_history.pop(0)
                        diis_err_history.pop(0)
                    # Extrapolate: need 3+ history, moderate residual.
                    # Stop DIIS below 1e-3 to let undamped iterations converge.
                    if (len(diis_err_history) >= 3
                            and 1e-3 < max_orb_change < 0.3):
                        diis_vec = diis_extrapolate(diis_orb_history, diis_err_history)
                        if diis_vec is not None:
                            trial = {k: list(shells) for k, shells in orbitals.items()}
                            self._unflatten_orbital_vector(
                                diis_vec, trial, orbital_schedule, ns_full)
                            try:
                                self._orthonormalize_orbitals(dict(trial), S0_cache)
                            except (ValueError, np.linalg.LinAlgError):
                                # DIIS can extrapolate outside the independent
                                # orbital manifold. Keep the valid sweep and
                                # restart its history instead of accepting a
                                # rank-deficient radial basis into CI.
                                diis_orb_history.clear()
                                diis_err_history.clear()
                            else:
                                orbitals = trial

            elif mode == "orbital" and self.exchange == "hf":
                # ============================================================
                # Per-orbital mode with exchange caching + potential mixing.
                # ============================================================
                exchange_cache = {}

                for sh, kappa, orb_idx in orbital_schedule:
                    H = H0_cache[kappa].copy()
                    S_full = S0_cache[kappa]

                    V_J_PP, V_J_QQ = self._build_direct_potential(
                        orbitals, target_kappa=kappa, target_idx=orb_idx)

                    K_PP, K_QQ, K_PQ, K_QP = self._build_exchange_matrix_hf(
                        kappa, orb_idx, orbitals,
                        exchange_cache=exchange_cache)
                    Vi_PP, Vi_QQ = self._build_intra_shell_potential(
                        kappa, orb_idx, orbitals)
                    n = min(V_J_PP.shape[0], ns_full)
                    H[:n, :n] += V_J_PP[:n, :n] - K_PP[:n, :n] + Vi_PP[:n, :n]
                    H[ns_full:ns_full+n, ns_full:ns_full+n] += (
                        V_J_QQ[:n, :n] - K_QQ[:n, :n] + Vi_QQ[:n, :n])
                    n_pq = min(K_PQ.shape[0], ns_full)
                    H[:n_pq, ns_full:ns_full+n_pq] -= K_PQ[:n_pq, :n_pq]
                    H[ns_full:ns_full+n_pq, :n_pq] -= K_QP[:n_pq, :n_pq]

                    virtual_keys = getattr(self, "virtual_orbital_keys", set())
                    target_virtual = (sh.n, kappa) in virtual_keys
                    other_orbitals = [np.concatenate((P, Q))
                                      for j, (P, Q, _) in enumerate(orbitals.get(kappa, []))
                                      if j != orb_idx and (target_virtual or
                                          (kappa_groups[kappa][j].n, kappa) not in virtual_keys)]
                    constraints = np.column_stack(other_orbitals) if other_orbitals else None
                    result = solve_dirac(H, S_full, ns_full, kappa,
                                          nsp=nsp if has_kb else None,
                                          nsq=nsq if has_kb else None,
                                          orthogonal_to=constraints)

                    eig_idx = sh.n - sh.l - 1
                    if kappa > 0 and not has_kb:
                        eig_idx += 1
                    if constraints is not None:
                        # Lower principal roots are already represented by
                        # the constrained orbitals. Follow the requested
                        # node/root index in the remaining spectrum; maximum
                        # overlap can lock a diffuse valence electron onto
                        # a higher excited root during the initial screening.
                        removed_lower = sum(
                            other.n < sh.n and (target_virtual or
                                (other.n, kappa) not in virtual_keys)
                            for other in kappa_groups[kappa])
                        eig_idx = max(0, eig_idx - removed_lower)
                    n_phys = len(result.eigenvalues)
                    actual_idx = min(eig_idx, n_phys - 1)
                    if actual_idx >= 0 and actual_idx < n_phys:
                        sorted_idx = np.argsort(result.eigenvalues)
                        phys_idx = sorted_idx[actual_idx]
                        vec = result.coefficients[:, phys_idx]
                        P_new = np.zeros(ns_full)
                        Q_new = np.zeros(ns_full)
                        nsp_loc = result.nsp if hasattr(result, 'nsp') else ns_full
                        nsq_loc = result.nsq if hasattr(result, 'nsq') else ns_full
                        P_new[:nsp_loc] = vec[:nsp_loc]
                        Q_new[:nsq_loc] = vec[ns_full:ns_full + nsq_loc]
                        P_new, Q_new = canonicalize_radial_phase(P_new, Q_new)
                        E_new = result.eigenvalues[phys_idx]
                        if np.isnan(E_new) or np.any(np.isnan(vec)):
                            continue
                        orb_list = orbitals.get(kappa, [])
                        while orb_idx >= len(orb_list):
                            orb_list.append((np.zeros(ns_full), np.zeros(ns_full), sh.occupation))

                        P_old, Q_old, _ = orb_list[orb_idx]
                        p_max = np.max(np.abs(P_old))
                        if p_max > 1e-15:
                            # Fix sign: ensure new orbital has same sign convention
                            overlap = np.dot(P_new, P_old)
                            if overlap < 0:
                                P_new = -P_new
                                Q_new = -Q_new
                            change = np.max(np.abs(P_new[:len(P_old)] - P_old[:len(P_new)])) / p_max
                            max_orb_change = max(max_orb_change, change)

                        # Orbital damping: mix new eigenvector with old
                        if damping > 0 and iteration > 1 and p_max > 1e-15:
                            alpha = 1.0 - damping
                            if residual < 0.1:
                                alpha = min(1.0, alpha + 0.3)
                            if residual < damping_release_residual:
                                alpha = 1.0
                            P_mixed = alpha * P_new + (1.0 - alpha) * P_old
                            Q_mixed = alpha * Q_new + (1.0 - alpha) * Q_old
                            # Renormalize to preserve norm
                            norm_new = np.sqrt(np.sum(P_new**2) + np.sum(Q_new**2))
                            norm_mix = np.sqrt(np.sum(P_mixed**2) + np.sum(Q_mixed**2))
                            if norm_mix > 1e-15 and norm_new > 1e-15:
                                P_mixed *= norm_new / norm_mix
                                Q_mixed *= norm_new / norm_mix
                            orb_list[orb_idx] = (P_mixed, Q_mixed, sh.occupation)
                        else:
                            orb_list[orb_idx] = (P_new, Q_new, sh.occupation)

                        orbitals[kappa] = orb_list
                        orbital_energies[sh.label] = E_new
                        keys_to_remove = [ck for ck in exchange_cache
                                          if ck[0] == kappa and ck[1] == orb_idx]
                        for ck in keys_to_remove:
                            del exchange_cache[ck]

            else:
                # Slater local exchange — same Fock for all orbitals.
                #
                # LDA SCF with a shared Fock is prone to root-swapping
                # (valence bound state disappearing for one iteration because
                # the potential briefly fails to bind it).  Shooting's
                # Salvat-style DHFS avoids this by mixing the potential, not
                # the orbitals — we do the same here.  The Fock matrix blocks
                # V_J + V_x are blended with the previous iteration's blocks;
                # ``slater_mix`` defaults to ``0.2`` (80% old / 20% new), the
                # value DHFS.f uses as its opening factor.
                V_J_PP, V_J_QQ = self._build_direct_potential(orbitals)
                Vx_PP, Vx_QQ = self._build_exchange_matrix_slater(orbitals)
                Pot_PP_new = V_J_PP + Vx_PP
                Pot_QQ_new = V_J_QQ + Vx_QQ
                if V_J_PP_prev is None:
                    Pot_PP = Pot_PP_new
                    Pot_QQ = Pot_QQ_new
                else:
                    slater_mix = 0.2
                    Pot_PP = (1.0 - slater_mix) * V_J_PP_prev + slater_mix * Pot_PP_new
                    Pot_QQ = (1.0 - slater_mix) * V_J_QQ_prev + slater_mix * Pot_QQ_new
                V_J_PP_prev = Pot_PP
                V_J_QQ_prev = Pot_QQ
                for kappa, shells in kappa_groups.items():
                    H = H0_cache[kappa].copy()
                    S_full = S0_cache[kappa]
                    n = min(Pot_PP.shape[0], ns_full)
                    H[:n, :n] += Pot_PP[:n, :n]
                    H[ns_full:ns_full+n, ns_full:ns_full+n] += Pot_QQ[:n, :n]
                    result = solve_dirac(H, S_full, ns_full, kappa,
                                          nsp=nsp if has_kb else None,
                                          nsq=nsq if has_kb else None)
                    orb_list = []
                    for sh in shells:
                        eig_idx = sh.n - sh.l - 1
                        if kappa > 0 and not has_kb:
                            eig_idx += 1
                        if eig_idx < result.n_bound:
                            P_new, Q_new, E_new = result.get_orbital(eig_idx)
                            old_list = orbitals.get(kappa, [])
                            idx = len(orb_list)
                            if idx < len(old_list):
                                P_old = old_list[idx][0]
                                p_max = np.max(np.abs(P_old))
                                if p_max > 1e-15:
                                    change = np.max(np.abs(P_new[:len(P_old)] - P_old[:len(P_new)])) / p_max
                                    max_orb_change = max(max_orb_change, change)
                            orb_list.append((P_new, Q_new, sh.occupation))
                            orbital_energies[sh.label] = E_new
                        else:
                            missing_shells.add(sh.label)
                            orb_list.append(
                                (np.zeros(ns_full), np.zeros(ns_full), sh.occupation))
                    orbitals[kappa] = orb_list

            self._orthonormalize_orbitals(orbitals, S0_cache)
            if self.exchange == "hf" and self.mean_field_coeffs is None:
                self._canonicalize_closed_subshells(orbitals, orbital_energies, kappa_groups)
            if (self.exchange == "hf" and self.mean_field_coeffs is None
                    and max_orb_change < 0.1):
                max_orb_change = max(max_orb_change, self._optimize_same_kappa_rotations(
                    orbitals, core_hole_stationary=getattr(self, 'core_hole_stationary', True)))

            # Check convergence
            E_total_quick = sum(
                sh.occupation * orbital_energies.get(sh.label, 0.0)
                for sh in self.subshells)
            if iteration > 1:
                energy_change = abs(E_total_quick - E_old) / max(abs(E_old), 1e-10)
            else:
                energy_change = 1.0
            if self._log:
                self._log.scf_iteration(
                    iteration, E_total_quick,
                    delta_E=(E_total_quick - E_old) if iteration > 1 else 0.0,
                    max_dp=max_orb_change)
            E_old = E_total_quick
            residual = max_orb_change

            if noninteracting or (max_orb_change < tol and energy_change < tol * 0.01):
                converged = True
                if noninteracting:
                    residual = 0.0
                break

        if missing_shells:
            raise UnboundSubshellError(
                "no bound state for subshell(s) "
                + ", ".join(sorted(missing_shells))
                + f" in Z={self.Z} with the requested configuration "
                + ", ".join(sh.label for sh in self.subshells)
                + ". The configuration is not bound in this potential — "
                "returning a zero-orbital, zero-energy result as converged "
                "would be silently wrong."
            )

        if self.exchange == "hf" and self.mean_field_coeffs is None:
            self._canonicalize_closed_subshells(orbitals, orbital_energies, kappa_groups)

        # Compute total energy (with a transient-NaN tripwire, HYP-1590)
        total_energy, energy_components = self._final_energy_checked(
            orbitals, orbital_energies)

        energy_components["_core_hole_rotation_mode"] = (
            "local_stationary_v1" if getattr(self, "core_hole_stationary", True)
            else "frozen")
        energy_components["_eigensolver_refinement_policy"] = (
            "requested_bound_index_plus_l_ge10_newton_v2")
        energy_components["_diis_residual_policy"] = "actual_sweep_input_v1"
        energy_components["_nuclear_profile"] = self.nuclear_profile

        # Verify the constrained metric after damping/DIIS as well.
        max_overlap = _max_same_kappa_overlap(orbitals, self.bsp_data)
        energy_components["max_same_kappa_overlap"] = max_overlap
        if max_overlap > _SAME_KAPPA_OVERLAP_TOL:
            converged = False
            energy_components["orthogonality_warning"] = (
                f"same-kappa overlap {max_overlap:.2e} exceeds "
                f"the constraint tolerance {_SAME_KAPPA_OVERLAP_TOL:.0e}")

        # Post-SCF corrections via the injected CorrectionStack (E1 seam)
        return self._finalize_with_corrections(SCFResult(
            converged=converged,
            iterations=iteration,
            residual=residual,
            total_energy=total_energy,
            orbital_energies=orbital_energies,
            subshells=self.subshells,
            energy_components=energy_components,
            orbitals_raw=_canonicalized_orbitals_raw(orbitals),
            Z=self.Z,
            _bsp_data=self.bsp_data,
        ))

    def _finalize_with_corrections(self, result):
        """Apply the injected CorrectionStack to a converged SCFResult.

        The single post-SCF corrections seam (review E1) — delegates to
        :func:`atomx.corrections.stack.finalize_with_corrections`, which
        the shooting path shares.
        """
        from atomx.corrections.stack import finalize_with_corrections

        from atomx.calculation_spec import resolved_solver_spec
        result.calculation_spec = resolved_solver_spec(self, method='DHF' if self.exchange == 'hf' else 'DFS', engine='bspline',
            controls=getattr(self, '_calculation_controls', {}))
        result._nuclear_potential = self._nuclear_potential
        result.__dict__["_jj_configs"] = self.jj_configs
        stack = self.correction_stack
        if (stack is not None and stack.include_breit
                and stack.breit_mode == "operator"):
            # The explicit operator mode needs CI angular machinery in
            # methods/mcdhf/. The default DBSR_HF average is evaluated
            # directly by its correction backend and avoids this extra
            # m-scheme contraction.
            from atomx.methods.mcdhf.breit_ci import (
                attach_breit_operator_energy,
            )
            attach_breit_operator_energy(result, jj_configs=self.jj_configs)
        return finalize_with_corrections(result, stack)


def helium_subshells():
    return [Subshell(n=1, kappa=-1, occupation=2)]


def neon_subshells():
    return [
        Subshell(n=1, kappa=-1, occupation=2),
        Subshell(n=2, kappa=-1, occupation=2),
        Subshell(n=2, kappa=1, occupation=2),
        Subshell(n=2, kappa=-2, occupation=4),
    ]


def argon_subshells():
    return [
        Subshell(n=1, kappa=-1, occupation=2),
        Subshell(n=2, kappa=-1, occupation=2),
        Subshell(n=2, kappa=1, occupation=2),
        Subshell(n=2, kappa=-2, occupation=4),
        Subshell(n=3, kappa=-1, occupation=2),
        Subshell(n=3, kappa=1, occupation=2),
        Subshell(n=3, kappa=-2, occupation=4),
    ]


@dataclass
class DiracFock(Method):
    """Relativistic Dirac-Fock self-consistent field method.

    Parameters
    ----------
    exchange : Exchange
        HARTREE_FOCK (default, non-local) or SLATER (local).
    breit : bool
        Include Breit interaction (magnetic + retardation corrections).
    qed : bool
        Include QED: vacuum polarization (Uehling) + self-energy (Lamb).
    basis_order : int
        B-spline polynomial order (default 9).
    starting_potential : str or None
        "tfda" or "gsz" for faster SCF convergence.  Auto-selects for Z>=20.
    nuclear_mass_me : float or None
        Nuclear mass in electron masses. If supplied, include the one-body
        Breit recoil operator. Two-body, finite-size, and radiative recoil
        are not included.
    nms_mode : str
        ``"operator"`` (default), or ``"virial"`` to reproduce historical
        normal mass shifts estimated from the total energy.
    nuclear_mass_number : int or None
        Mass number used by the finite nuclear charge model. Independent of
        the recoil mass; None preserves the historical element defaults.
        Heavy-element radii remain empirical, and light-element reference
        radii are not isotope-resolved. Recorded in energy components.
    scf_damping : float or None
        Fraction of the previous orbital retained during SCF mixing.
        None uses the solver's automatic choice.
    scf_damping_release_residual : float
        Residual below which explicit damping is released. Defaults to 1e-2;
        a smaller value retains damping closer to convergence.
    seed_orbitals : dict or None
        Occupied B-spline orbitals from a calculation on the same radial
        basis. Useful for continuing a difficult HF solve from a converged
        local-exchange state; the HF equations are still solved anew.
    core_hole_stationary : bool
        Local stationary rotations between occupied same-kappa orbitals
        with a vacancy in the lower shell. Defaults to True. False reproduces
        historical frozen rotations, which can report false stationarity.
    """
    exchange: Exchange = Exchange.HARTREE_FOCK
    breit: bool = False
    qed: bool = False
    basis_order: int = 9
    starting_potential: str = None
    nuclear_mass_me: float | None = None
    nuclear_mass_number: int | None = None
    scf_damping: float | None = None
    seed_orbitals: Dict | None = field(default=None, repr=False)
    scf_damping_release_residual: float = 1e-2
    breit_mode: str = "dbsr_hf"
    nms_mode: str = "operator"
    include_sms: bool = False
    include_hydrogenic_mass_remainder: bool = False
    core_hole_stationary: bool = True

    @property
    def theory_level(self) -> TheoryLevel:
        if self.qed:
            return TheoryLevel.DIRAC_FOCK_QED
        if self.breit:
            return TheoryLevel.DIRAC_FOCK_BREIT
        if self.exchange == Exchange.SLATER:
            return TheoryLevel.DIRAC_FOCK_SLATER
        return TheoryLevel.DIRAC_HARTREE_FOCK

    def solve(self, Z, subshells, tol=1e-5, max_iter=40,
              jj_configs=None, use_physics_grid=True, ns=113,
              log=None, r_max=None, grid_hi=None, grid_he=None,
              grid_hmax=None, screening_potential=None):
        """Run Dirac-Fock SCF calculation."""
        # DFSolver is defined above in this module.
        if not use_physics_grid and any(
                value is not None for value in (grid_hi, grid_he, grid_hmax)):
            raise ValueError('grid shape controls require use_physics_grid=True')

        if self.scf_damping is not None and not 0 <= self.scf_damping < 1:
            raise ValueError("scf_damping must be in [0, 1)")
        if self.breit_mode not in {
                "dbsr_hf", "operator", "scalar_pairs", "scalar_legacy"}:
            raise ValueError(
                "breit_mode must be 'dbsr_hf', 'operator', 'scalar_pairs', "
                "or 'scalar_legacy'")
        if not 0 < self.scf_damping_release_residual < 1:
            raise ValueError("scf_damping_release_residual must be in (0, 1)")

        starting = self.starting_potential
        if starting is None and Z >= 20:
            starting = "tfda"

        correction_stack = None
        if (self.nuclear_mass_me is not None or self.breit or self.qed
                or self.include_sms or self.include_hydrogenic_mass_remainder):
            from atomx.corrections.qed import resolve_se_method
            from atomx.corrections.stack import CorrectionStack
            correction_stack = CorrectionStack(
                se_backend=(resolve_se_method("auto", Z) if self.qed else None),
                include_vp=self.qed,
                include_breit=self.breit,
                breit_mode=self.breit_mode,
                nuclear_mass_me=self.nuclear_mass_me,
                nms_mode=self.nms_mode,
                include_sms=self.include_sms,
                include_hydrogenic_mass_remainder=self.include_hydrogenic_mass_remainder,
            )

        solver = DFSolver(
            Z=Z, subshells=subshells,
            ns=ns, ks=self.basis_order,
            exchange=self.exchange.value,
            include_breit=self.breit,
            include_qed=self.qed,
            correction_stack=correction_stack,
            nuclear=self.nuclear.value,
            nuclear_mass_number=self.nuclear_mass_number,
            r_max=r_max, screening_potential=screening_potential,
            starting_potential=starting,
            jj_configs=jj_configs,
            log=log)
        solver.seed_orbitals = self.seed_orbitals
        solver.core_hole_stationary = self.core_hole_stationary

        if use_physics_grid:
            from atomx.numerics.bspline.basis import prepare_bspline_data
            from atomx.numerics.bspline.knots import make_physics_knots
            t_phys, ns_phys = make_physics_knots(
                ks=self.basis_order, Z=Z, nuclear=self.nuclear.value,
                r_max=r_max, hi=grid_hi, he=grid_he, hmax=grid_hmax)
            data = prepare_bspline_data(
                t_phys, ns_phys, self.basis_order, kinetic_balance=True)
            solver.t = t_phys
            solver.bsp_data = data
            solver.ns = ns_phys
            solver.r_max = float(t_phys[-1])
            solver.kinetic_balance = True
            solver._precompute()
            if log:
                log.basis_info(data)

        result = solver.solve(max_iter=max_iter, tol=tol,
                              damping=self.scf_damping,
                              damping_release_residual=self.scf_damping_release_residual)
        result._theory_level = self.theory_level.value
        result._method_name = self.name
        if not result._bsp_data:
            result._bsp_data = solver.bsp_data
        return result
