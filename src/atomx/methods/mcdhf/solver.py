"""MCDHF structure with constrained radial orbitals and exact jj CI.

The configuration-average DHF iteration supplies an initial orbital set.
For multiconfiguration B-spline calculations, the final optimization uses
analytic derivatives of the correlated Dirac–Coulomb CI functional in a
fixed positive-energy subspace. Explicit frozen-orbital protocols retain
that choice; the shooting route currently uses the averaged initializer.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from scipy import linalg

from atomx.corrections.qed import DEFAULT_SE_METHOD
from atomx.methods.ci import ci_solve
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mcdhf.config import CSF, get_shared_orbital_basis
from atomx.methods.mcdhf.matrix import GridCIWorkspace, build_ci_matrix
from atomx.numerics.bspline.dirac_assembly import canonicalize_radial_phase


@dataclass
class MCDHFResult:
    """Result of an MCDHF calculation.

    Satisfies the ``AtomicResult`` protocol from ``atomx.results``.
    """
    converged: bool
    iterations: int
    state_energies: np.ndarray
    mixing_coefficients: np.ndarray
    ground_energy: float
    orbital_energies: Dict[str, float]
    csfs: List[CSF]
    orbitals_dict: Dict = field(default_factory=dict)  # (n,kappa) -> (P, Q)
    bsp_data: Dict = field(default_factory=dict)
    Z: int = 0
    #: ``{(n, kappa): occupation}`` — ground-state Guerra-averaged
    #: effective occupations.  Populated by :class:`MCDHFSolver`;
    #: empty for results that haven't been through an SCF cycle.
    occupations: Dict = field(default_factory=dict)
    #: DODS / EOL per-(J, parity)-block orbital sets.  HYP-184 scaffold:
    #: each entry maps a ``(twoJ, parity)`` tuple to a separate
    #: ``{(n, kappa): (P, Q)}`` orbital dict optimised for that block.
    #: Empty by default — the solver currently emits one Guerra-averaged
    #: orbital set shared across all blocks (see ``MCDHFSolver``).  When
    #: populated, the transition calculator should apply a biorthogonal
    #: transformation when computing ``<F|T^L|I>`` for I and F in
    #: different blocks.
    per_block_orbitals: Dict = field(default_factory=dict)
    _bsp_data: Dict = field(default_factory=dict, repr=False)
    #: Radial grid metadata for the shooting-engine flavor.  In that
    #: flavor ``orbitals_dict`` already contains sampled P/Q arrays rather
    #: than B-spline coefficients.
    _grid: np.ndarray | None = field(default=None, repr=False)
    _difr: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        # Alias ``bsp_data`` → ``_bsp_data`` so downstream code (correction
        # adapters, hyperfine evaluators) can treat any AtomicResult
        # uniformly via the private name.
        if self.bsp_data and not self._bsp_data:
            self._bsp_data = self.bsp_data
        elif self._bsp_data and not self.bsp_data:
            self.bsp_data = self._bsp_data

    @property
    def n_states(self):
        return len(self.state_energies)

    @property
    def total_energy(self) -> float:
        return self.ground_energy

    @property
    def energy_components(self) -> Dict[str, float]:
        """Correction-aware energy breakdown (legacy SCFResult keys).

        Mirrors ``SCFResult.energy_components`` /
        :func:`atomx.corrections.stack.finalize_with_corrections` so the
        Breit/QED shifts folded into ``state_energies`` stay inspectable
        on the MCDHF route: ``E_Breit``, ``E_SE``, ``E_VP``, ``E_WK``,
        ``E_QED_total``, ``se_method`` (whichever were computed), plus
        ``E_total``.

        ``E_Breit`` is always the Breit shift that entered ``E_total``:

        * ``breit_mode="scalar"`` (and every non-CI route) — the
          configuration-average number from
          the exact Gaunt operator, which
          ``solve`` adds to every state;
        * ``breit_mode="operator"`` — the *measured* ground-state shift
          produced by the exact Gaunt operator inside the CI blocks
          (``E(with Breit) − E(without Breit)`` on the same orbitals), so
          the reported number is the number that is in the total.

        On the operator route the scalar is **not applied**, and is
        reported separately as ``E_Breit_scalar_not_applied`` — a key
        whose name says so.  Before this was split, ``E_Breit`` held the
        scalar (Ne: 0.050449 Ha) while the levels carried the operator
        (Ne: 0.017340 Ha): a reported value 3.03x the applied one and
        3.03x DBSR_HF, with nothing in the dict to warn a reader.
        See :attr:`breit_provenance`.
        """
        comps: Dict[str, float] = {"E_total": self.ground_energy}
        e_breit = self.__dict__.get("_E_Breit")
        if e_breit:
            comps["E_Breit"] = e_breit
        e_scalar = self.__dict__.get("_E_Breit_scalar_not_applied")
        if e_scalar:
            comps["E_Breit_scalar_not_applied"] = e_scalar
        qed = self.__dict__.get("_E_QED") or {}
        for key in ("E_SE", "E_VP", "E_WK", "E_QED_total", "se_method"):
            if key in qed:
                comps[key] = qed[key]
        nms = self.__dict__.get("_nms_state_shifts")
        if nms is not None:
            comps["nms"] = float(nms[0])
        sms = self.__dict__.get("_sms_state_shifts")
        if sms is not None:
            comps["sms"] = float(sms[0])
        return comps

    @property
    def orbital_optimization(self) -> Dict:
        """Model and stationarity diagnostics for the radial optimization."""
        variational = self.__dict__.get("_variational_diagnostics")
        if variational is not None:
            return dict(variational, initializer_converged=self.__dict__.get("_inner_converged"))
        return {
            "model": self.__dict__.get("_radial_optimization", "unspecified"),
            "functional": "configuration-average / frozen-orbital Dirac-Fock",
            "full_correlated_orbital_gradient": False,
            "inner_converged": self.__dict__.get("_inner_converged"),
            "inner_residual": self.__dict__.get("_inner_residual"),
        }

    @property
    def breit_provenance(self) -> Dict:
        """How the Breit interaction entered this result.

        ``{}`` when Breit was off.  Otherwise ``mode`` is ``"operator"``
        (exact Gaunt operator inside every ``(2J, parity)`` CI block) or
        ``"scalar"`` (the comparison-only uniform shift), plus what was
        omitted — see :data:`atomx.methods.mcdhf.breit_ci.RETARDATION_NOTE`.
        """
        return dict(self.__dict__.get("_breit_provenance") or {})

    @property
    def energy_breakdown(self) -> Dict[str, float]:
        """Alias for :attr:`energy_components` (AtomicResult protocol)."""
        return self.energy_components

    @property
    def qed_provenance(self) -> Dict:
        """Operator content and remaining approximations of applied QED."""
        return dict(self.__dict__.get("_qed_provenance") or {})

    @property
    def subshells(self) -> List[Subshell]:
        """SCF-compatible subshell list using ground-state averaged occupations."""
        keys = set(self.occupations) | set(self.orbitals_dict)
        out: List[Subshell] = []
        for n, kappa in sorted(keys, key=lambda nk: (nk[0], abs(nk[1]), -nk[1])):
            occ = float(self.occupations.get((n, kappa), 0.0))
            out.append(Subshell(n=n, kappa=kappa, occupation=occ))
        return out

    @property
    def orbitals_raw(self) -> Dict[int, list]:
        """SCF-compatible ``kappa -> [(P, Q, occ), ...]`` orbital mapping."""
        by_kappa: Dict[int, list] = {}
        for sh in self.subshells:
            pq = self.orbitals_dict.get((sh.n, sh.kappa))
            if pq is None:
                continue
            P, Q = pq
            by_kappa.setdefault(sh.kappa, []).append((P, Q, sh.occupation))
        return by_kappa

    @property
    def theory_level(self):
        from atomx.types import TheoryLevel
        return TheoryLevel.MCDHF

    @property
    def method_name(self) -> str:
        return "MCDHF"

    @property
    def coupling_scheme(self):
        from atomx.types import CouplingScheme
        return CouplingScheme.JJ

    @property
    def waves(self):
        """Build :class:`atomx.results.Waves` for either radial engine."""
        from atomx.methods.dirac_fock import _eval_bspline_on_grid
        from atomx.results import OrbitalWavefunction, Waves
        if not self.orbitals_dict:
            return Waves(grid=np.array([]))

        if self.bsp_data:
            grid = self.bsp_data["gr"].ravel()
            basis_type = "bspline"
        elif self._grid is not None:
            grid = np.asarray(self._grid)
            basis_type = "grid"
        else:
            return Waves(grid=np.array([]))

        orbitals = {}
        for (n, kappa), (P_coeffs, Q_coeffs) in self.orbitals_dict.items():
            l = kappa if kappa > 0 else -kappa - 1
            j2 = 2 * abs(kappa) - 1
            l_char = "spdfghiklmno"[l] if l < 12 else f"l{l}"
            label = f"{n}{l_char}{j2}/2"
            if self.bsp_data:
                P_grid = _eval_bspline_on_grid(P_coeffs, self.bsp_data, "p")
                Q_grid = _eval_bspline_on_grid(Q_coeffs, self.bsp_data, "q")
            else:
                P_grid = np.asarray(P_coeffs)
                Q_grid = np.asarray(Q_coeffs)
            orbitals[label] = OrbitalWavefunction(
                P=P_grid, Q=Q_grid,
                n=n, l=l, kappa=kappa,
                # Guerra-averaged ground-state effective occupation;
                # 0.0 if the solver did not populate ``occupations``
                # (back-compat for callers constructing MCDHFResult by hand).
                occupation=float(self.occupations.get((n, kappa), 0.0)),
                energy=self.orbital_energies.get(label, 0.0),
                label=label)
        return Waves(grid=grid, orbitals=orbitals, basis_type=basis_type)

    def excitation_energies(self, unit="hartree"):
        from atomx.constants import HARTREE_CM1, HARTREE_EV
        de = self.state_energies - self.state_energies[0]
        if unit == "ev":
            return de * HARTREE_EV
        elif unit == "cm-1":
            return de * HARTREE_CM1
        return de

    def one_body_density_matrix(self, state=None):
        """Spherically averaged radial one-body density, including coherence.

        Returns ``(orbital_keys, density)`` with trace equal to the electron
        number. By default average exactly degenerate ground states, as for
        the reported occupations; an integer selects a particular state.
        """
        from atomx.methods.mcdhf.exact_ci import csf_signature, one_body_tensor_terms

        keys = tuple(sorted(self.orbitals_dict))
        index = {key: i for i, key in enumerate(keys)}
        density = np.zeros((len(keys), len(keys)))
        states = (np.flatnonzero(abs(self.state_energies - self.state_energies[0]) < 1e-10)
                  if state is None else [state])
        signatures = [csf_signature(csf) for csf in self.csfs]
        for st in states:
            mixing = self.mixing_coefficients[:, st]
            active = np.flatnonzero(abs(mixing) > 1e-15)
            for r in active:
                for s in active:
                    weight = mixing[r] * mixing[s] / len(states)
                    for a, b, angular in one_body_tensor_terms(signatures[r], signatures[s], 0):
                        if a[1] == b[1]:
                            density[index[a], index[b]] += weight * angular * np.sqrt(
                                2 * abs(a[1]) / (signatures[r].total_J2 + 1))
        return keys, 0.5 * (density + density.T)

    @property
    def state_J2(self) -> np.ndarray:
        """For each eigenstate, the ``2J`` of its dominant CSF (jj-CSF only).

        For an AL-flat CSF list (every CSF has ``total_J2=0``) this
        returns all zeros — matches the legacy single-config behaviour.
        """
        if self.mixing_coefficients.ndim != 2 or self.mixing_coefficients.size == 0:
            return np.zeros(0, dtype=int)
        n_states = self.mixing_coefficients.shape[1]
        out = np.zeros(n_states, dtype=int)
        for k in range(n_states):
            j = int(np.argmax(np.abs(self.mixing_coefficients[:, k])))
            out[k] = int(getattr(self.csfs[j], "total_J2", 0))
        return out

    @property
    def state_parity(self) -> np.ndarray:
        """Parity (±1) of each eigenstate's dominant CSF."""
        if self.mixing_coefficients.ndim != 2 or self.mixing_coefficients.size == 0:
            return np.zeros(0, dtype=int)
        n_states = self.mixing_coefficients.shape[1]
        out = np.zeros(n_states, dtype=int)
        for k in range(n_states):
            j = int(np.argmax(np.abs(self.mixing_coefficients[:, k])))
            out[k] = int(getattr(self.csfs[j], "parity", 1))
        return out

    @property
    def level_table(self) -> list:
        """Compact J-resolved level table.

        Returns a list of dicts ``{index, E_hartree, dE_eV, twoJ, parity,
        config}`` sorted by energy.  ``config`` is the dominant CSF
        label; ``dE_eV`` is the excitation energy above the ground
        state.  Intended for benchmark output and downstream FAC
        comparison.
        """
        from atomx.constants import HARTREE_EV
        n_states = len(self.state_energies)
        if n_states == 0:
            return []
        e0 = float(self.state_energies[0])
        twoJ = self.state_J2
        par = self.state_parity
        rows = []
        for k in range(n_states):
            dom = int(np.argmax(np.abs(self.mixing_coefficients[:, k]))) \
                if self.mixing_coefficients.size else 0
            cfg = self.csfs[dom].label if dom < len(self.csfs) else ""
            rows.append({
                "index": k,
                "E_hartree": float(self.state_energies[k]),
                "dE_eV": (float(self.state_energies[k]) - e0) * HARTREE_EV,
                "twoJ": int(twoJ[k]),
                "parity": int(par[k]),
                "config": cfg,
            })
        return rows


class MCDHFSolver:
    """Multi-Configuration Dirac-Hartree-Fock solver.

    Uses Guerra average-level occupations for the DHF orbital seed. The
    B-spline correlated orbital step targets the ground-state CI energy.
    """

    def __init__(self, Z, csfs, ns=50, ks=9, r_min=None, r_max=None,
                 include_breit=False, include_qed=False,
                 qed_se_method=None,
                 correction_stack=None,
                 nuclear="fermi", use_physics_grid=True,
                 starting_potential=None,
                 per_block_orbital_sets=False,
                 radial_optimization="mcdhf",
                 engine="bspline",
                 breit_mode="operator",
                 screening_potential=None,
                 grid_hi=None, grid_he=None, grid_hmax=None,
                 variational_projector="nuclear"):
        self.Z = Z
        self.csfs = csfs
        self.n_csf = len(csfs)
        self.ns = ns
        self.ks = ks
        self.include_breit = include_breit
        self.include_qed = include_qed
        # The Breit interaction enters as the exact Gaunt operator in
        # every (2J, parity) CI block (atomx.methods.mcdhf.breit_ci), so
        # it splits levels.  That is the only mode: the scalar
        # configuration-average Breit was removed in 2026-09 (3.00x /
        # 3.03x / 3.10x DBSR_HF for He / Ne / Ar — see
        # corrections/breit.py), and `breit_mode="scalar"` now refuses
        # rather than quietly applying a 3x-wrong number.
        #
        # The operator is verified against DBSR_HF, the reference the
        # repo is modelled on (``dbsr_hf an=Z term=AV mbreit=1``):
        #
        #   He 1s2           DBSR_HF 0.000063777432919  operator
        #                    6.377743260e-05 (+5e-9 relative)
        #   Ne 1s2 2s2 2p6   DBSR_HF 0.016643630840490  operator
        #                    0.017340451404 (+4.2 %, omitted retardation)
        #   Ar 1s2 ... 3p6   DBSR_HF 0.132364619357042  operator
        #                    0.140544488 (+6.2 %, omitted retardation)
        if breit_mode not in {"operator", "scalar"}:
            raise ValueError(
                "breit_mode must be 'operator' or 'scalar' "
                f"(got {breit_mode!r})")
        # ``"scalar"`` is the historical uniform shift, now with the
        # fixed scalar (v=L normalised by 1/(2L+1), within 0.13 %/0.33 %/
        # 0.66 % of DBSR_HF for He/Ne/Ar).  It cannot split levels by
        # construction, so it stays the comparison path; "operator" is
        # the default.
        self.breit_mode = breit_mode
        # Single configuration surface for post-SCF corrections (review
        # E1).  MCDHF's Guerra-averaged occupations are not expressible
        # through the result-based adapters yet, so the stack is consumed
        # as configuration by _apply_corrections_to_orbitals below.
        # Self-energy backend for the multi-configuration path.  The
        # single-configuration routes thread Atom's qed_backend through
        # DFSolver/ShootingSolver; the MCDHF solver used to resolve
        # DEFAULT_SE_METHOD here regardless, so an explicit backend was
        # silently ignored on multi-configuration runs.
        self.qed_se_method = qed_se_method or DEFAULT_SE_METHOD
        if correction_stack is not None:
            self.correction_stack = correction_stack
        elif include_breit or include_qed:
            from atomx.corrections.qed import resolve_se_method
            from atomx.corrections.stack import CorrectionStack
            self.correction_stack = CorrectionStack(
                se_backend=(resolve_se_method(self.qed_se_method, Z)
                            if include_qed else None),
                include_vp=bool(include_qed),
                include_wk_vp=False,
                include_breit=bool(include_breit),
            )
        else:
            self.correction_stack = None
        if self.correction_stack is not None:
            if self.correction_stack.include_hydrogenic_mass_remainder:
                raise NotImplementedError(
                    'Hydrogenic mass remainder requires spectroscopic principal-shell '
                    'identity; MCDHF correlation-orbital labels cannot supply it')
            self.include_breit = bool(self.correction_stack.include_breit)
            self.include_qed = bool(self.correction_stack.se_backend is not None
                                    or self.correction_stack.include_vp
                                    or self.correction_stack.include_wk_vp)
        # Electron-gas (plasma) screening, forwarded to whichever inner
        # solver the selected engine uses (DFSolver or ShootingHFSolver).
        self.screening_potential = screening_potential
        self.orbital_basis = get_shared_orbital_basis(csfs)
        self.r_min = r_min
        self.r_max = r_max
        self.grid_hi = grid_hi
        self.grid_he = grid_he
        self.grid_hmax = grid_hmax
        self.nuclear = nuclear
        self.use_physics_grid = use_physics_grid
        self.starting_potential = starting_potential
        if radial_optimization not in {"mcdhf", "fac_frozen", "fac_frozen_local"}:
            raise ValueError(
                "radial_optimization must be 'mcdhf', 'fac_frozen', or 'fac_frozen_local' "
                f"(got {radial_optimization!r})"
            )
        self.radial_optimization = radial_optimization
        if variational_projector not in ("nuclear", "fock"):
            raise ValueError(
                "variational_projector must be 'nuclear' or 'fock', "
                f"got {variational_projector!r}")
        self.variational_projector = variational_projector
        # DODS / EOL per-(J, parity)-block orbital optimisation toggle —
        # HYP-184 scaffold.  When True, ``solve()`` will (eventually)
        # run a separate Guerra-averaged DHF per (J, parity) block on
        # the CSF list filtered to that block, store the resulting
        # orbital sets on ``MCDHFResult.per_block_orbitals``, and rely
        # on the transition calculator to apply a biorthogonal
        # transformation when computing inter-block matrix elements.
        # The implementation is currently a no-op (one shared Guerra
        # orbital set across all blocks) and is tracked under HYP-184
        # follow-up item (2).
        self.per_block_orbital_sets = bool(per_block_orbital_sets)
        # Radial ENGINE for the inner SCF + CI kernel pair.  "bspline"
        # (default) is the historical DFSolver + cell-integral route;
        # "shooting" runs ShootingHFSolver with Guerra jj_configs pair
        # weights and the grid-flavour CI kernel (GridCIWorkspace) — the
        # SAME consistency pairing (identical Slater integrator and
        # nuclear potential between the SCF functional and the CI matrix
        # elements) that fixed the +74 Ha Pb bug on the B-spline side.
        if engine not in {"bspline", "shooting"}:
            raise ValueError(
                f"engine must be 'bspline' or 'shooting' (got {engine!r})")
        self.engine = engine
        if engine == "shooting":
            if self.radial_optimization != "mcdhf":
                raise ValueError(
                    "radial_optimization='fac_frozen'/'fac_frozen_local' "
                    "are DFSolver protocols; engine='shooting' supports "
                    "'mcdhf' only")
            if self.per_block_orbital_sets:
                raise ValueError(
                    "per_block_orbital_sets runs its per-block solves "
                    "through DFSolver; it requires engine='bspline'")
            if nuclear not in ("point", "fermi"):
                raise ValueError(
                    f"engine='shooting' supports nuclear='point' or "
                    f"'fermi' (got {nuclear!r})")

    @property
    def _breit_in_ci(self) -> bool:
        """True when the Breit operator is carried by the CI blocks.

        Only then may the one-shot scalar Breit shift be skipped; with
        ``breit_mode="scalar"`` the historical uniform shift still runs.
        """
        return bool(self.include_breit) and self.breit_mode == "operator"

    @staticmethod
    def _floored_occ_map(csf) -> Dict:
        """Per-CSF occupations with the jj-PARTNER floor (``ñ``).

        Returns ``{(n, kappa): q̃}`` where an orbital ABSENT from the CSF
        is injected at 0.001 iff its same-``(n, l)`` jj partner
        (``kappa' = -kappa - 1``) IS occupied — e.g. a Pb (6p₁/₂)² CSF
        carries ``6p₃/₂ → 0.001``.  Rationale: when the CI weights
        collapse onto such a CSF, a hard-zero entry makes every
        ``WW(i, ·)`` product vanish for the orphaned jj component, its
        Fock operator loses the entire electron-repulsion term, and the
        orbital dives to a hydrogenic bare-nucleus state (Pb 6p₃/₂ →
        −95.6 Ha, mislabeled 4p-scale energy).  With the floor,
        ``WW(i, j)/ϖᵢ → q_j`` — the correct neutral-atom mean field.

        The floor is *not* the general cure for that failure mode.  It
        only keeps the Guerra average defined for an orbital that is
        genuinely populated somewhere (``ϖᵢ ≥ 0.001 > 0``); an orbital
        with no population at all in any CSF still has ``WW(i, ·) ≡ 0``
        and would still be solved against a bare nucleus.  That case is
        handled by :meth:`_mean_field_matrices`' asymmetric mean field
        (a spectator virtual is solved in its parent configurations'
        core — GRASP's layer intuition: an excited layer sees the core
        left behind by the electron it replaced), *not* by widening
        this floor — widening it would re-screen the occupied orbitals
        with phantom correlation density, which is the opposite error.
        The floor stays jj-partner-only on purpose: it repairs the
        numerator/denominator mismatch in ``WW(i, j)/ϖᵢ`` for a partner
        that is populated in some CSFs, and does nothing else.
        """
        occ = {(sh.n, sh.kappa): float(sh.occupation) for sh in csf.subshells}
        if sum(occ.values()) <= 1:
            # An auxiliary jj partner must never create electron repulsion
            # in a one-electron Hamiltonian. Zero-population target fields
            # are already supplied by the parent-core construction below.
            return occ
        for (n, kappa), q in list(occ.items()):
            partner = -kappa - 1
            if partner == 0 or q <= 0.0:
                continue  # s₁/₂ has no partner / unoccupied source
            occ.setdefault((n, partner), 0.001)
        return occ

    def _guerra_populations(self, weights) -> Dict:
        """Raw Guerra populations ``ϖᵢ = Σ_c w_c ñᵢ^c`` (unclamped).

        Same per-CSF occupations (:meth:`_floored_occ_map`) and same
        summation order as :meth:`_averaged_subshells`, so a population
        is bit-identical to the value that method would average — but
        *before* the 0.001 solvability floor is applied.  The distinction
        matters for the asymmetric mean field: ``ϖᵢ`` is what turns the
        Guerra product into the conditional average
        ``WW(i, j)/ϖᵢ = ⟨q_j⟩_{i}``, and a population that is exactly
        zero means that average does not exist (0/0) — see
        :meth:`_mean_field_matrices`.
        """
        if weights is None:
            weights = np.ones(self.n_csf) / self.n_csf
        occ_maps = [self._floored_occ_map(csf) for csf in self.csfs]
        keys = set()
        for m in occ_maps:
            keys.update(m)
        return {
            key: sum(w * m.get(key, 0.0)
                     for w, m in zip(weights, occ_maps, strict=False))
            for key in keys
        }

    def _reference_occupations(self) -> Dict:
        """Occupation map of the FIRST CSF — the layer reference ``q^ref``.

        :meth:`_fac_frozen_subshells` already treats ``self.csfs[0]`` as
        the reference configuration (optimise it, keep everything else as
        a tiny-occupation virtual), and the R7 asymmetric mean field uses
        the same convention so both radial protocols agree on what "the
        reference" is.  Raw CSF occupations, *without* the jj-partner
        floor: the floor exists to keep a solved orbital's Guerra average
        from becoming 0/0, not to invent population in the reference.
        """
        ref: Dict = {}
        if not self.csfs:
            return ref
        for sh in self.csfs[0].subshells:
            ref[(sh.n, sh.kappa)] = float(sh.occupation)
        return ref

    def _mean_field_matrices(self, avg_subshells, weights, jj_configs):
        """Asymmetric mean-field coefficients ``(C, normalization)`` — R7.

        The Guerra product ``WW(i, j) = Σ_c w_c ñᵢ^c ñⱼ^c`` is symmetric
        and can only express one mean field for both directions: the
        per-electron weight of source ``j`` in target ``i``'s Fock
        operator is ``WW(i, j)/ϖᵢ`` and it scales with the target's own
        population ``ϖᵢ``.  When ``ϖᵢ`` collapses the target loses its
        entire electron-repulsion term and is solved against a bare
        nucleus (Be ``correlation="s"``: 2p → −2.0005 Ha = the hydrogenic
        Z=4 2p; 3p/3d likewise), while the reverse — the occupied core
        being re-screened by phantom correlation density — must *not*
        happen.  That asymmetry is not expressible as a single symmetric
        product, so it is passed to :class:`DFSolver` explicitly.

        Returns
        -------
        coeffs : ndarray, shape ``(n_orb, n_orb)``
            Raw coefficient of source ``j`` in target ``i``'s direct and
            exchange potential (``DFSolver.mean_field_coeffs``).
        normalization : ndarray, shape ``(n_orb,)``
            Per-target divisor (``DFSolver.mean_field_normalization``);
            the physical per-electron weight is ``coeffs[i, j] /
            normalization[i]``.

        Rules
        -----
        * Target ``i`` with a Guerra population (``ϖᵢ > 0``) **or** present
          in the reference CSF: ``coeffs[i, :] = WW(i, :)`` and
          ``normalization[i] = qᵢ`` — literally today's arithmetic, so
          every closed-shell number and every occupied-orbital row is
          bit-identical to the pre-R7 code.
        * Target ``i`` with ``ϖᵢ = 0`` that is *not* in the reference: a
          spectator virtual, solved in the field of the **parent
          configurations' own core** — the parent's occupations with the
          spectator removed — not the reference's.  Concretely: let
          ``Sᵢ = {c : qᵢ^c > 0}`` (raw occupations, no jj-partner floor)
          and let the holes of parent ``c`` be the shells where
          ``q_c < q^ref``.  Among parents with the fewest holes, keep
          those whose deepest hole is shallowest (a removed shell's
          ``(n, l)`` — the valence-promotion channel dominates); the row
          is the mean over that set:

          ``coeffs[i, j] = ⟨q_j^c⟩_c`` for ``j ≠ i``, ``coeffs[i, i] =
          ⟨qᵢ^c (qᵢ^c − 1)⟩_c`` (same-shell partner count),
          ``normalization[i] = 1``.

          The parent core is the conditional field the orbital actually
          lives in.  The R7 predecessor used the capped reference
          ``min(q^ref_j, max_{c∈Sᵢ} q_j^c)`` and a ``q_ref − 1``
          diagonal, and both carry defects under this codebase's
          angular conventions.  (1) ``DFSolver._get_exchange_coef``
          returns 0.0 for ``i == j`` structurally and the direct self
          density is doubled (DBSR_HF ``c = c + c`` at ``j ==
          target_i``), so a self weight ``w`` contributes ``w × (+J_i)``
          with no compensating ``+K_i``: the historical ``q_ref − 1``
          delivered pure ``−J_i`` anti-screening.  Measured on H-like
          Z=1 2p with hand-set seam rows: self weight 0 reproduces the
          Dirac closed form to 5.4e-10 Ha; −0.5 shifts the level by
          −0.105 Ha; −1 by −0.237 Ha, landing 2p₁/₂ at −0.3616 Ha with
          the fine-structure ordering inverted.  (2) Even with the
          diagonal at 0 the capped *reference* over-screens a valence
          spectator: the reference holds electrons in shells that the
          parents — the configurations the spectator actually appears
          in — have promoted away.  Measured on Be ``correlation="s"``
          with the converged 1s²2s² core frozen: the capped row
          ``[2, 2]`` puts 2p₁/₂ at +0.006 Ha (unbound in the box;
          FAC's own frozen-core protocol — ``OptimizeRadial`` on the
          ground group — puts 1s²2s²2p₁ at 1.708 eV, the same
          barely-bound state), while the parent core ``[2, 1]`` (the
          2s→2p parents 1s²2s¹2p¹, a 2s hole) puts it at −0.212 Ha,
          inside the suite's bracket and next to the relaxed value.
          GRASP's layer intuition agrees: an excited layer sees the
          core left behind by the electron it replaced.

          The two historical anchor cases are recovered as limits:
          H-like (parent ``2p¹``: hole at 1s, core ``[]``) gives the
          zero row — exact Dirac, no self term; He (parent ``1s¹2p¹``:
          core ``[1]``) reproduces the R7 cap's ``ε(2p) = −0.125`` and
          the 0.0469 Ha 1s2p splitting.  The self term is the
          conditional factorial moment, reducing to the FAC-calibrated
          occupied-shell value ``q(q−1)`` for parents that doubly
          occupy the spectator's shell, and to 0 for single-occupancy
          parents.  Orbitals present only through the jj-partner floor
          (no raw parent) keep the capped-reference fallback — no
          conditional distribution exists for them.

        Why the threshold is exactly zero rather than a small positive
        number: the numerator ``WW(i, j)`` is *unclamped* while the
        divisor is the solver's floored occupation ``max(ϖᵢ, 0.001)``, so
        the Guerra average is exact for every ``ϖᵢ ≥ 0.001`` and merely
        under-scaled (by ``ϖᵢ/0.001``, today's pre-existing behaviour,
        left untouched) below it.  Only ``ϖᵢ = 0`` makes the average
        0/0 — and a zero population is structural (the orbital appears
        only in CSFs that cannot mix with the state, e.g. a different
        ``(J, parity)`` block), not a numerically noisy quantity, so the
        branch cannot flicker between outer iterations.
        """
        from atomx.config.jj_coupling import compute_WW

        order = [(s.n, s.kappa) for s in avg_subshells]
        n_orb = len(order)
        coeffs = np.empty((n_orb, n_orb), dtype=np.float64)
        for i in range(n_orb):
            for j in range(n_orb):
                coeffs[i, j] = compute_WW(i, j, jj_configs)
        normalization = np.array(
            [float(s.occupation) for s in avg_subshells], dtype=np.float64)

        populations = self._guerra_populations(weights)
        ref_occ = self._reference_occupations()
        occ_maps = [self._floored_occ_map(csf) for csf in self.csfs]
        raw_maps = [
            {(sh.n, sh.kappa): float(sh.occupation) for sh in csf.subshells}
            for csf in self.csfs
        ]
        for i, key in enumerate(order):
            if populations.get(key, 0.0) > 0.0:
                continue  # populated: Guerra mean field (today's numbers)
            if ref_occ.get(key, 0.0) > 0.0:
                continue  # an occupied orbital always keeps its own row
            normalization[i] = 1.0
            # Parent configurations (RAW occupations — the jj-partner
            # floor would invent phantom parents whose core mirrors the
            # reference and drags the row back toward over-screening).
            # The spectator is solved in the mean core of the
            # configurations it actually appears in: row[j] = <q_j^c>,
            # self = <q_i^c (q_i^c - 1)> (same-shell partner count).
            raw_parents = [m for m in raw_maps if m.get(key, 0.0) > 0.0]
            if not raw_parents:
                # Present only through the jj-partner floor: no
                # conditional distribution exists; keep the R7
                # capped-reference fallback.
                parents = [m for m in occ_maps if m.get(key, 0.0) > 0.0]
                for j, key_j in enumerate(order):
                    q_ref = ref_occ.get(key_j, 0.0)
                    if parents:
                        q_ref = min(
                            q_ref,
                            max(m.get(key_j, 0.0) for m in parents),
                        )
                    coeffs[i, j] = q_ref
                continue

            def _holes(m):
                return [
                    (nj, ref_occ.get(nj, 0.0) - m.get(nj, 0.0))
                    for nj in ref_occ
                    if m.get(nj, 0.0) < ref_occ.get(nj, 0.0) - 1e-12
                ]

            def _shell_nl(nk):
                n, kappa = nk
                return (n, kappa if kappa > 0 else -kappa - 1)

            # Valence-promotion channel: fewest holes, then shallowest
            # deepest hole (a removed shell's (n, l)).  Ties average.
            min_holes = min(len(_holes(m)) for m in raw_parents)
            cand = [m for m in raw_parents if len(_holes(m)) == min_holes]
            deepest = {
                id(m): min(_shell_nl(nk) for nk in _holes(m)) for m in cand
            }
            best = max(deepest.values())
            sel = [m for m in cand if deepest[id(m)] == best]
            n_sel = float(len(sel))
            for j, key_j in enumerate(order):
                if key_j == key:
                    coeffs[i, j] = sum(
                        m.get(key_j, 0.0) * (m.get(key_j, 0.0) - 1.0)
                        for m in sel
                    ) / n_sel
                else:
                    coeffs[i, j] = sum(
                        m.get(key_j, 0.0) for m in sel
                    ) / n_sel
        return coeffs, normalization

    def _collapsed_targets(self, avg_subshells, weights) -> List[str]:
        """Labels of spectator virtuals with no Guerra population (R7).

        The same condition :meth:`_mean_field_matrices` branches on:
        ``ϖᵢ = 0`` for an orbital that the reference CSF does not occupy.
        Such a target's Fock operator has no electron-repulsion term at
        all under the symmetric Guerra product, so any engine that cannot
        consume the asymmetric seam must refuse it rather than report
        hydrogenic orbitals as a converged result.
        """
        populations = self._guerra_populations(weights)
        ref_occ = self._reference_occupations()
        return [
            sh.label for sh in avg_subshells
            if populations.get((sh.n, sh.kappa), 0.0) <= 0.0
            and ref_occ.get((sh.n, sh.kappa), 0.0) <= 0.0
        ]

    def _averaged_subshells(self, weights=None):
        """Guerra Eq (10): ϖᵢ = Σⱼ cⱼ² ñᵢʲ — fractional occupations.

        All orbitals appearing in any CSF are included (with a final
        minimum occupation 0.001) so they get solved even when their
        weight is near-zero — essential for transition calculations.

        The average runs over the jj-partner-floored per-CSF occupations
        ``ñ`` (:meth:`_floored_occ_map`) — the SAME occupations
        ``_jj_configs`` feeds to ``compute_WW`` — so the effective mean
        field ``WW(i, j)/ϖᵢ`` stays an exact weighted average of the
        other shells' occupations even when the CI weights collapse onto
        CSFs where orbital *i* is absent.  With inconsistent floors the
        ratio under-counts screening and near-phantom jj components dive
        toward hydrogenic energies.

        The raw, unclamped populations are :meth:`_guerra_populations`;
        the R7 asymmetric mean field (:meth:`_mean_field_matrices`) keys
        off them to decide which targets have a Guerra average at all.
        """
        avg_occ = self._guerra_populations(weights)

        subshells = []
        for (n, kappa), occ in sorted(avg_occ.items(),
                                       key=lambda x: (x[0][0], abs(x[0][1]))):
            max_q = 2 * abs(kappa)
            # Ensure all orbitals are solved (min occupation for numerical stability)
            occ = max(occ, 0.001)
            occ = min(occ, max_q)
            subshells.append(Subshell(n=n, kappa=kappa, occupation=occ))

        return subshells

    def _physical_subshells(self, weights):
        """CSF number-operator expectations, including exactly empty virtuals."""
        occupations = dict.fromkeys(self.orbital_basis, 0.0)
        for weight, csf in zip(weights, self.csfs, strict=True):
            for sh in csf.subshells:
                key = (sh.n, sh.kappa)
                occupations[key] = occupations.get(key, 0.0) + weight * sh.occupation
        return [Subshell(n=n, kappa=k, occupation=q)
                for (n, k), q in sorted(occupations.items(),
                                       key=lambda item: (item[0][0], abs(item[0][1])))]

    def _fac_frozen_subshells(self):
        """FAC-like radial set: optimise the first CSF and keep virtuals tiny.

        FAC structure scripts commonly call ``OptimizeRadial`` on the closed
        ground group and then generate excited orbitals in that frozen
        central potential. This helper builds the analogous subshell list:
        occupations from the first CSF, plus tiny non-zero occupancies for
        every additional orbital needed by the CI basis so DFSolver solves
        those orbitals without relaxing the core toward the excited manifold.

        ``self.csfs[0]`` is the layer reference for both radial protocols:
        this method takes its occupations directly, and the Guerra
        protocol's asymmetric mean field (:meth:`_mean_field_matrices`)
        uses the same map for every orbital whose own Guerra population
        has collapsed (see :meth:`_reference_occupations`).
        """
        ref_occ = self._reference_occupations()

        subshells = []
        for n, kappa in sorted(self.orbital_basis, key=lambda x: (x[0], abs(x[1]), x[1])):
            occ = ref_occ.get((n, kappa), 0.001)
            occ = min(max(occ, 0.001), 2 * abs(kappa))
            subshells.append(Subshell(n=n, kappa=kappa, occupation=occ))
        return subshells

    def _jj_configs(self, avg_subshells, weights):
        """Build JJConfig list carrying c_k² weights for DFSolver.compute_WW.

        The orbital index order in each JJConfig.subshells must match
        ``avg_subshells`` so that DFSolver's ``compute_WW(i, j)`` picks the
        right occupation. Occupations are the jj-partner-floored ``ñ`` of
        :meth:`_floored_occ_map` — identical to what
        :meth:`_averaged_subshells` averages — so ``WW(i, j)/ϖᵢ`` stays a
        consistent mean field even when the CI weights collapse onto CSFs
        where a jj component is absent (Pb (6p₁/₂)² ground: without the
        partner floor the orphaned 6p₃/₂ saw zero electron repulsion and
        collapsed to a hydrogenic −95.6 Ha state).
        """
        from atomx.config.jj_coupling import JJConfig
        order = [(s.n, s.kappa) for s in avg_subshells]
        configs = []
        for i, csf in enumerate(self.csfs):
            occ_map = self._floored_occ_map(csf)
            positional = [
                Subshell(n=n, kappa=kappa,
                         occupation=float(occ_map.get((n, kappa), 0.0)))
                for (n, kappa) in order
            ]
            configs.append(JJConfig(subshells=positional, weight=float(weights[i])))
        return configs

    def solve(self, max_iter=30, scf_tol=1e-5, ci_tol=1e-8, scf_max_iter=40):
        """Run the MCDHF calculation.

        The key improvement: directly uses converged DHF orbitals from
        SCFResult.orbitals_raw, avoiding monkey-patching.
        """
        weights = np.ones(self.n_csf) / self.n_csf
        E_old = 0.0
        E_hist: list = []
        w_prev = None
        frozen_weights = None
        converged = False
        inner_converged = False
        inner_residual = float("inf")
        orbital_energies: Dict[str, float] = {}
        mixing = np.eye(self.n_csf)
        state_energies = np.zeros(self.n_csf)
        iteration = 0
        orbitals_dict: Dict = {}
        dhf = None
        shoot_result = None
        seed_orbitals = None  # warm-start carry-over between outer iterations
        #: Keyword arguments of the SAME Dirac-Coulomb-only CI as the last
        #: outer iteration (same CSFs, orbitals, workspace, nuclear
        #: model).  Used once, for reporting only, to measure the shift
        #: the Breit *operator* actually applied — see
        #: ``E_Breit_operator`` below.
        dc_ci_kwargs = None

        # Configuration-average iterations only supply an orbital seed for
        # the full correlated functional below. Converging that different
        # fixed point can spend hundreds of diagonalizations on spectator
        # orbitals without improving the eventual variational solution.
        use_correlated_gradient = (self.radial_optimization == "mcdhf"
                                   and self.engine == "bspline" and self.n_csf > 1)
        initializer_iterations = min(max_iter, 3) if use_correlated_gradient else max_iter
        for iteration in range(1, initializer_iterations + 1):
            # Step 1: Effective occupations. The default is Guerra Eq 10.
            # ``fac_frozen`` intentionally follows FAC's structure protocol:
            # optimise the first/ground configuration and keep excited
            # orbitals as tiny-occupation virtuals in that frozen potential.
            if self.radial_optimization in {"fac_frozen", "fac_frozen_local"}:
                avg_subshells = self._fac_frozen_subshells()
                jj_configs = None
            else:
                avg_subshells = self._averaged_subshells(weights)
                jj_configs = self._jj_configs(avg_subshells, weights)

            if self.engine == "shooting":
                # Step 2 (shooting): Froese-Fischer non-local HF on the
                # shooting grid with the SAME fractional occupations and
                # the Guerra jj_configs pair weights (ShootingHFSolver
                # mirrors DFSolver's WW(a,b)/q_a Fock conventions).  No
                # warm-start between outer iterations: the solver exposes
                # no orbital-seed input — every solve() bootstraps from
                # its own local-exchange (Gu) starter on a fresh SGRID —
                # and one inner solve is cheap next to the B-spline path,
                # so re-solving per outer iteration is acceptable.
                from atomx.numerics.shooting.hf_shooting import ShootingHFSolver

                # R7 is only fixed on the B-spline path, which is the one
                # that consumes the asymmetric mean field
                # (``_mean_field_matrices`` -> ``DFSolver.mean_field_coeffs``).
                # ShootingHFSolver builds its Fock operator from the same
                # symmetric Guerra product, so a target with no Guerra
                # population would still be solved against a bare nucleus
                # there.  Refuse that case loudly rather than return
                # hydrogenic correlation orbitals with converged=True.
                collapsed = self._collapsed_targets(avg_subshells, weights)
                if collapsed:
                    raise NotImplementedError(
                        "MCDHFSolver(engine='shooting'): Guerra population "
                        "of " + ", ".join(collapsed) + " is exactly zero, "
                        "i.e. these are spectator virtuals whose Fock "
                        "operator has no electron-repulsion term at all. "
                        "The R7 asymmetric mean field that repairs this is "
                        "implemented for engine='bspline' only; "
                        "ShootingHFSolver has no coefficient seam and would "
                        "return hydrogenic, bare-nucleus orbitals for them."
                    )

                shoot_solver = ShootingHFSolver(
                    Z=self.Z, subshells=avg_subshells,
                    nuclear=self.nuclear,
                    jj_configs=jj_configs,
                    screening_potential=self.screening_potential,
                )
                # tol: the shooting default 1e-7 (per-sweep relative
                # energy change, 0.5 damping) leaves ~2e-5 Ha SCF
                # residual — enough to dominate ci-vs-statistical
                # deltas.  1e-9 keeps the inner functional converged
                # well below ci_tol.
                shoot_result = shoot_solver.solve(
                    max_iter=max(scf_max_iter, 60),
                    tol=min(scf_tol, 1.0e-9))
                orbital_energies = shoot_result.orbital_energies

                # Step 3: grid orbitals keyed (n, kappa); rows of
                # _pa/_qa follow the solver's subshell order.  Phase
                # canonicalisation applies to grid arrays exactly as to
                # B-spline coefficient vectors (largest-|P| lobe sign).
                orbitals_dict = {}
                for i, sh in enumerate(shoot_solver.subshells):
                    P, Q = canonicalize_radial_phase(
                        shoot_result._pa[i], shoot_result._qa[i])
                    orbitals_dict[(sh.n, sh.kappa)] = (P, Q)

                if not orbitals_dict:
                    break  # no orbitals found

                # Steps 4 + 5: grid-flavour CI.  The workspace carries
                # the shooting solver's own r*V_nuc and SLAG6 grid, so
                # the CI kernel evaluates matrix elements with the SAME
                # numerics as the inner SCF energy functional (the
                # consistency principle noted in the bspline branch
                # below).  nuclear_potential/convol_cache stay None —
                # the workspace owns both roles.
                grid_ws = GridCIWorkspace.from_shooting(
                    shoot_solver, shoot_result)
                ci_res = ci_solve(
                    self.csfs, orbitals_dict, self.orbital_basis,
                    grid_ws, self.Z, grid_ws.n,
                    include_breit=self._breit_in_ci,
                )
                # Reporting-only companion solve (no Breit).  Deferred to
                # the end of ``solve`` so it never runs inside the SCF
                # loop; the stashed arguments are this iteration's.
                dc_ci_kwargs = dict(
                    csfs=self.csfs, orbitals_dict=orbitals_dict,
                    orbital_basis=self.orbital_basis, bsp_data=grid_ws,
                    Z=self.Z, ns=grid_ws.n, include_breit=False)
            else:
                # Step 2: Solve DHF with fractional occupations. Skip
                # Breit/QED here — they are perturbative in atomx's
                # DFSolver (do not enter the Fock matrix), so they only
                # affect the final energy. We compute them ONCE after the
                # MCDHF outer loop converges.
                exchange_model = "hf"
                if self.radial_optimization == "fac_frozen_local":
                    exchange_model = "gu"

                kwargs = {}
                if self.r_min is not None:
                    kwargs["r_min"] = self.r_min
                if self.r_max is not None:
                    kwargs["r_max"] = self.r_max

                # Build Guerra-weighted JJConfig list so DFSolver.compute_WW
                # realises ϖᵢⱼ = Σⱼ cⱼ² nᵢʲ nⱼʲ — the correct
                # averaged-monoconfiguration Fock operator (Guerra Eq 10).
                # In FAC-frozen mode ``jj_configs`` is intentionally ``None``:
                # DFSolver then uses the tiny virtual occupations in
                # ``avg_subshells`` directly, matching the FAC-like radial
                # experiment where excited orbitals are solved but do not
                # relax the closed core appreciably.
                #
                # R7 asymmetric mean field: hand DFSolver the explicit
                # coefficient matrix (Guerra product, except for targets
                # whose population has collapsed — those are solved in the
                # reference configuration's frozen potential).  With
                # ``jj_configs is None`` (FAC-frozen protocols) the
                # DFSolver's own q_i*q_j convention already is that frozen
                # potential, so no seam is supplied.
                # Supply the asymmetric seam ONLY when a spectator
                # virtual actually needs it.  With no collapsed target
                # every row's Guerra arithmetic is already correct, and
                # handing DFSolver the seam would route those rows through
                # ``coeffs[i, j] / normalization[i]`` instead of the
                # historical raw ``WW(i, j)`` — a different normalisation
                # that cancels for a closed shell (q_i uniform) but does
                # NOT for an open one.  Measured: He-like
                # ``Atom(2, config=["1s2", "1s1 2p1"])`` lost three
                # orders of magnitude of 2p potential that way and the
                # textbook ``E(1P1) - E(3P1) = 2 G^1(1s,2p)/3`` identity
                # broke (ratio 1.000 -> 15.054, G^1 0.1405 -> 0.000186 Ha).
                # Gating on ``_collapsed_targets`` keeps every
                # non-spectator case bit-identical to the pre-R7 code by
                # construction rather than by coincidence.
                mean_field_coeffs = mean_field_norm = None
                if jj_configs is not None and self._collapsed_targets(
                        avg_subshells, weights):
                    mean_field_coeffs, mean_field_norm = (
                        self._mean_field_matrices(
                            avg_subshells, weights, jj_configs))

                dhf = DFSolver(
                    Z=self.Z, subshells=avg_subshells,
                    ns=self.ns, ks=self.ks,
                    exchange=exchange_model,
                    include_breit=False, include_qed=False,
                    nuclear=self.nuclear,
                    starting_potential=(self.starting_potential
                                        or ("tfda" if self.Z >= 20 else None)),
                    jj_configs=jj_configs,
                    mean_field_coeffs=mean_field_coeffs,
                    mean_field_normalization=mean_field_norm,
                    screening_potential=self.screening_potential,
                    **kwargs,
                )

                # Install the DBSR-HF-style semi-exponential physics grid
                # before _precompute(), so the inner DHF matches the
                # DFSolver path used for averaging in
                # {"statistical","equal","fractional"}.
                if self.use_physics_grid:
                    from atomx.numerics.bspline.basis import prepare_bspline_data
                    from atomx.numerics.bspline.knots import make_physics_knots
                    t_phys, ns_phys = make_physics_knots(
                        ks=self.ks, Z=self.Z, nuclear=self.nuclear,
                        r_max=self.r_max, hi=self.grid_hi, he=self.grid_he,
                        hmax=self.grid_hmax)
                    data = prepare_bspline_data(t_phys, ns_phys, self.ks,
                                                kinetic_balance=True)
                    dhf.t = t_phys
                    dhf.bsp_data = data
                    dhf.ns = ns_phys
                    dhf._precompute()

                # Warm-start from the previous outer iteration's converged
                # orbitals. Only applied when the basis dimension matches;
                # the grid is rebuilt identically each outer iter, so it
                # does.
                if seed_orbitals is not None:
                    dhf.seed_orbitals = seed_orbitals

                if self.radial_optimization in {"fac_frozen", "fac_frozen_local"}:
                    reference_occupations = self._reference_occupations()
                    dhf.virtual_orbital_keys = {key for key in self.orbital_basis
                                               if reference_occupations.get(key, 0.0) == 0.0}
                dhf_result = dhf.solve(max_iter=scf_max_iter, tol=scf_tol)
                inner_converged = bool(dhf_result.converged)
                inner_residual = float(dhf_result.residual)
                orbital_energies = dhf_result.orbital_energies
                # Stash for the next outer iteration
                seed_orbitals = dhf_result.orbitals_raw

                # Step 3: Extract converged orbitals from SCFResult, keyed
                # by (n, kappa) via the result's own subshell bookkeeping —
                # never by positional per-kappa counters (see
                # SCFResult.orbitals_by_subshell).
                orbitals_dict = {}
                for (n, kappa), (P, Q, _occ) in (
                        dhf_result.orbitals_by_subshell().items()):
                    P, Q = canonicalize_radial_phase(P, Q)
                    orbitals_dict[(n, kappa)] = (P, Q)

                if not orbitals_dict:
                    break  # no orbitals found

                # Steps 4 + 5: CI matrix build + diagonalise via methods/ci.
                # Share the inner solver's nuclear potential and
                # cell-integral cache: CI matrix elements must be evaluated
                # with the SAME finite-nucleus model and the SAME
                # (cell-algorithm) Slater integrator as the SCF energy
                # functional, or the CI total drifts from the DHF total on
                # identical orbitals (74 Ha at Z=82 with the old
                # point-nucleus + Y^k-quadrature kernel).
                ci_res = ci_solve(
                    self.csfs, orbitals_dict, self.orbital_basis,
                    dhf.bsp_data, self.Z, dhf.ns,
                    nuclear_potential=dhf._nuclear_potential,
                    convol_cache=getattr(dhf, "_convol_cache", None),
                    include_breit=self._breit_in_ci,
                )
                # Reporting-only companion solve (no Breit); see the
                # shooting branch above.
                dc_ci_kwargs = dict(
                    csfs=self.csfs, orbitals_dict=orbitals_dict,
                    orbital_basis=self.orbital_basis, bsp_data=dhf.bsp_data,
                    Z=self.Z, ns=dhf.ns,
                    nuclear_potential=dhf._nuclear_potential,
                    convol_cache=getattr(dhf, "_convol_cache", None),
                    include_breit=False)
            if shoot_result is not None:
                inner_converged = bool(shoot_result.converged)
                inner_residual = float(getattr(shoot_result, "residual", float("nan")))
            state_energies = ci_res.state_energies
            mixing = ci_res.mixing_coefficients

            # Step 6: Update weights = cⱼ² (ground state mixing). In
            # FAC-frozen radial mode the Fock weights stay pinned to the
            # first configuration, but keeping ``weights`` updated preserves
            # convergence checks and result occupations.
            #
            # Repeated-parentage CSFs are exactly degenerate under the
            # orthogonal-parentage kernel, so the raw ground eigenvector is
            # an arbitrary rotation within that subspace — feeding it to
            # the Guerra occupations made open-shell solves (Co 3d7)
            # oscillate forever.  Average |c|^2 over the degenerate
            # multiplet: deterministic, and identical to the raw vector
            # whenever the ground state is non-degenerate.
            deg = np.abs(state_energies - state_energies[0]) < 1e-9
            if deg.sum() > 1:
                w_new = (mixing[:, deg] ** 2).mean(axis=1)
            else:
                w_new = mixing[:, 0] ** 2
            w_new = w_new / w_new.sum()

            E_new = state_energies[0]
            energy_change = abs(E_new - E_old)

            if frozen_weights is not None:
                # final pass on the cycle-averaged weights: accept.
                E_old = E_new
                converged = inner_converged
                break

            if iteration > 1 and energy_change < ci_tol and inner_converged:
                E_old = E_new
                converged = inner_converged
                break

            # Period-2 limit-cycle detection: open-shell configurations
            # with two near-degenerate AL ground candidates (e.g. Co
            # 3d7 4s2, two jj CSFs 2.9e-4 Ha apart) flip dominance every
            # iteration — each candidate's self-consistent orbitals favor
            # the other, so the bare weights -> |c_ground|^2 iteration
            # never settles (damping only makes the cycle aperiodic).
            # When E returns to its value two iterations ago, freeze the
            # weights at the cycle average — the configuration-average
            # answer over the two competing states — and do one final
            # self-consistent pass.
            if (len(E_hist) >= 2 and w_prev is not None
                    and abs(E_new - E_hist[-2]) < 10.0 * ci_tol
                    and energy_change > ci_tol):
                frozen_weights = 0.5 * (w_prev + w_new)
                frozen_weights = frozen_weights / frozen_weights.sum()
                weights = frozen_weights
            else:
                weights = w_new
            E_hist.append(E_new)
            w_prev = w_new
            E_old = E_new

        variational_diagnostics = None
        if use_correlated_gradient and orbitals_dict:
            from atomx.methods.mcdhf.variational import (
                CorrelatedOrbitalFunctional,
                optimize_orbitals,
            )
            functional = CorrelatedOrbitalFunctional(
                self.csfs, dhf.bsp_data, self.Z, dhf._nuclear_potential,
                getattr(dhf, "_convol_cache", None))
            # The nuclear spectrum is a poor Hessian approximation for a
            # screened heavy-atom valence shell, and — unless
            # variational_projector="fock" — it is also the fixed no-pair
            # boundary itself (critical review F3: the projector used to be
            # nuclear unconditionally, with only the preconditioner patched).
            # Project the initializer's actual Fock fields into the no-pair
            # space; optimize_orbitals reports no_pair_projector_leakage in
            # its diagnostics so the two boundaries can be compared.
            preconditioners = {}
            for flat_i, sh in enumerate(dhf.subshells):
                if (sh.n, sh.kappa) not in functional.keys:
                    continue
                pp, qq, _, qp = dhf._build_hf_matrix_dbsr(
                    flat_i, dhf_result.orbitals_raw, skip_pq_exchange=True)
                h = functional.one_body[sh.kappa].copy()
                ns = functional.ns
                h[:ns, :ns] += pp
                h[ns:, ns:] += qq
                h[ns:, :ns] += qp
                preconditioners[sh.n, sh.kappa] = np.tril(h) + np.tril(h, -1).T
            orbitals_dict, dc_ci, variational_diagnostics = optimize_orbitals(
                functional, orbitals_dict, max_iter=32 * max_iter,
                tolerance=scf_tol, energy_tolerance=ci_tol,
                preconditioner_matrices=preconditioners,
                projector_source=self.variational_projector)
            # The initializer's convergence is separately reported. The
            # final CI/orbital stationarity decides the variational result.
            converged = bool(variational_diagnostics["converged"])
            dc_ci_kwargs["orbitals_dict"] = orbitals_dict
            if self._breit_in_ci:
                ci_kwargs = dict(dc_ci_kwargs, include_breit=True)
                final_ci = ci_solve(**ci_kwargs)
            else:
                final_ci = dc_ci
            state_energies = final_ci.state_energies
            mixing = final_ci.mixing_coefficients
            degenerate = abs(dc_ci.state_energies - dc_ci.state_energies[0]) < 1e-10
            occupations_for_energy = self._physical_subshells(
                np.mean(dc_ci.mixing_coefficients[:, degenerate] ** 2, axis=1))
            one_electron = sum(s.occupation for s in self.csfs[0].subshells) == 1
            for sh in occupations_for_energy:
                key = (sh.n, sh.kappa)
                if one_electron:
                    vector = np.concatenate(orbitals_dict[key])
                    orbital_energies[sh.label] = float(vector @ functional.one_body[sh.kappa] @ vector)
                elif sh.occupation > 1e-10:
                    vector = np.concatenate(orbitals_dict[key])
                    orbital_energies[sh.label] = float(
                        vector @ functional.last_gradient[key] / (2 * sh.occupation))
            variational_diagnostics["orbital_energies"] = (
                "one-electron nuclear eigenvalues" if one_electron else
                "diagonal Lagrange multipliers divided by occupation")

        # Compute ground-state Guerra-averaged effective occupations,
        # so MCDHFResult.waves can carry physically meaningful
        # per-orbital occupations (used by Atom.density, etc.).
        degenerate = abs(state_energies - state_energies[0]) < 1e-10
        ground_weights = np.mean(mixing[:, degenerate] ** 2, axis=1)
        ground_weights = ground_weights / ground_weights.sum()
        # Physical number-operator expectations. The artificial populations
        # used to keep virtual orbitals solvable never belong in a result.
        ground_subshells = self._physical_subshells(ground_weights)
        occupations = {(sh.n, sh.kappa): float(sh.occupation) for sh in ground_subshells}

        # One-shot perturbative Breit + QED on the final converged orbitals.
        # These corrections do not enter the Fock matrix in atomx's DHF, so
        # doing them once here (rather than inside each outer iteration's
        # DFSolver) is both physically equivalent and N× faster, where N is
        # the MCDHF outer-iteration count.
        E_Breit = 0.0
        qed_components: Dict[str, float] = {}
        corr_data = None
        i0 = 0
        stack = self.correction_stack
        if stack is not None and (dhf is not None or shoot_result is not None):
            from atomx.corrections.breit import compute_breit_energy
            # The correction functions are engine-uniform through the
            # atomx.corrections.workspace seam: B-spline results pass the
            # solver's bsp_data, shooting results a radial-grid workspace.
            # The grid flavour slices off the r = 0 origin node (exactly
            # like corrections.adapters._grid_workspace_of) so -Z/r,
            # Uehling and Y^k/r^{k+1} potentials stay finite.
            if self.engine == "shooting":
                from atomx.corrections.workspace import make_grid_workspace
                i0 = int(np.searchsorted(shoot_result._grid, 0.0,
                                         side="right"))
                corr_data = make_grid_workspace(
                    shoot_result._grid[i0:], shoot_result._difr[i0:])
            else:
                i0 = 0
                corr_data = dhf.bsp_data
            # Flatten orbitals on the final basis for the correction APIs.
            orb_list = []
            kappa_counter: Dict[int, int] = {}
            for sh in ground_subshells:
                idx = kappa_counter.get(sh.kappa, 0)
                kappa_counter[sh.kappa] = idx + 1
                cached = orbitals_dict.get((sh.n, sh.kappa))
                if cached is not None:
                    P, Q = cached
                    orb_list.append((P[i0:], Q[i0:],
                                     sh.occupation, sh.kappa, sh.label))
            if stack.include_breit:
                E_Breit = float(compute_breit_energy(orb_list, corr_data,
                                                     self.Z))

        # DODS / EOL per-(J, parity)-block orbital relaxation (HYP-184
        # scaffold).  When ``per_block_orbital_sets=True``, build a
        # block → orbital-set map; otherwise leave empty so the
        # transition calculator falls back to the shared Guerra set.
        per_block_orbitals: Dict = {}
        if self.per_block_orbital_sets:
            per_block_orbitals = self._solve_per_block_orbitals(
                mixing=mixing,
                shared_orbitals=orbitals_dict,
                dhf=dhf,
            )
            if per_block_orbitals and dhf is not None:
                state_energies, mixing = self._diagonalize_per_block_ci(
                    state_energies=state_energies,
                    mixing=mixing,
                    per_block_orbitals=per_block_orbitals,
                    bsp_data=dhf.bsp_data,
                    ns=dhf.ns,
                    nuclear_potential=dhf._nuclear_potential,
                    convol_cache=getattr(dhf, "_convol_cache", None),
                )

        # The one-shot corrections.  Two of them are now genuinely
        # different objects: the Breit *operator* (when it is in the CI,
        # the levels are already split and nothing is added here) and the
        # QED shift: full one-body coherence for local VP, diagonal
        # orbital self-energy for the current model-SE backend.
        breit_operator_in_ci = bool(self._breit_in_ci)

        # REPORTING ONLY.  When the operator is in the CI, ``E_Breit``
        # must be the shift that is actually in the total, not the
        # configuration-average scalar that the operator replaced.
        # ``E(CI with Gaunt) − E(CI without)`` on identical CSFs,
        # orbitals and workspace is that shift by definition.  This is
        # the only extra work the reporting costs, it runs once outside
        # the SCF loop, and it touches nothing the CI already computed.
        # Skipped for the per-block-orbital scaffold, whose states come
        # from per-block orbital sets a single companion solve cannot
        # reproduce; there the scalar is reported under its
        # not-applied key rather than passed off as the applied value.
        E_Breit_operator = None
        if (breit_operator_in_ci and dc_ci_kwargs is not None
                and not per_block_orbitals):
            e_with = float(np.asarray(state_energies, dtype=float)[0])
            e_without = float(np.asarray(
                ci_solve(**dc_ci_kwargs).state_energies, dtype=float)[0])
            E_Breit_operator = e_with - e_without

        if not breit_operator_in_ci and E_Breit:
            state_energies = state_energies + E_Breit
        qed_state_shifts = np.zeros_like(np.asarray(state_energies,
                                                   dtype=float))
        qed_state_components = {}
        if self.include_qed:
            qed_state_components = self._qed_state_components(
                mixing=mixing, orbitals_dict=orbitals_dict,
                corr_data=corr_data, i0=i0, per_block_orbitals=per_block_orbitals)
            qed_state_shifts = sum(qed_state_components.values(), qed_state_shifts)
            state_energies = state_energies + qed_state_shifts

        nms_state_shifts = np.zeros_like(np.asarray(state_energies, dtype=float))
        nms_provenance = None
        if stack is not None and stack.nuclear_mass_me is not None:
            from atomx.corrections.adapters import NormalMassShiftCorrection
            from atomx.corrections.recoil import (
                recoil_provenance,
                recoil_state_parameters,
            )
            # Validate the same mass/mode contract as the DHF adapter.
            NormalMassShiftCorrection(stack.nuclear_mass_me, stack.nms_mode)
            if stack.nms_mode == "virial":
                nms_state_shifts = -np.asarray(state_energies) / stack.nuclear_mass_me
                nms_provenance = {"estimator": "virial theorem ⟨T_e⟩ = −E_total",
                                  "nuclear_mass_me": stack.nuclear_mass_me}
            else:
                if self.per_block_orbital_sets:
                    raise NotImplementedError("recoil with per-block orbital sets is not implemented")
                if corr_data is None:
                    raise ValueError("one-body recoil requires retained radial orbitals")
                radial_orbitals = {key: (p[i0:], q[i0:])
                                   for key, (p, q) in orbitals_dict.items()}
                nms_state_shifts = recoil_state_parameters(
                    self.csfs, mixing, radial_orbitals, corr_data, self.Z
                ) / stack.nuclear_mass_me
                nms_provenance = recoil_provenance(stack.nuclear_mass_me)
                nms_provenance["evaluation"] = "first-order CI state expectation including radial coherence"
            state_energies = state_energies + nms_state_shifts

        sms_state_shifts = np.zeros_like(np.asarray(state_energies, dtype=float))
        sms_provenance = None
        if stack is not None and stack.include_sms:
            from atomx.corrections.specific_recoil import (
                momentum_recoil_parameters,
                specific_recoil_provenance,
            )
            if self.per_block_orbital_sets:
                raise NotImplementedError("two-body recoil with per-block orbitals is not implemented")
            if corr_data is None:
                raise ValueError("two-body recoil requires retained radial orbitals")
            radial_orbitals = {key: (p[i0:], q[i0:])
                               for key, (p, q) in orbitals_dict.items()}
            sms_state_shifts = momentum_recoil_parameters(
                self.csfs, mixing, radial_orbitals, corr_data, Z=self.Z
            ) / stack.nuclear_mass_me
            sms_provenance = specific_recoil_provenance(stack.nuclear_mass_me)
            state_energies = state_energies + sms_state_shifts

        # Perturbative corrections or separately optimized blocks can
        # reorder close levels. Keep energies, amplitudes and shifts paired,
        # then report occupations of the actual final ground-state ensemble.
        order = np.argsort(state_energies, kind="stable")
        state_energies = np.asarray(state_energies)[order]
        mixing = mixing[:, order]
        qed_state_shifts = qed_state_shifts[order]
        qed_state_components = {key: value[order] for key, value in qed_state_components.items()}
        nms_state_shifts = nms_state_shifts[order]
        sms_state_shifts = sms_state_shifts[order]
        degenerate = abs(state_energies - state_energies[0]) < 1e-10
        final_weights = np.mean(mixing[:, degenerate] ** 2, axis=1)
        final_subshells = self._physical_subshells(final_weights)
        occupations = {(sh.n, sh.kappa): float(sh.occupation) for sh in final_subshells}
        if qed_state_components:
            qed_components = {key: float(value[0]) for key, value in qed_state_components.items()}
            qed_components["E_QED_total"] = float(qed_state_shifts[0])
            if stack.se_backend is not None:
                from atomx.corrections.qed import resolve_se_method
                qed_components["se_method"] = resolve_se_method(stack.se_backend, self.Z)

        result = MCDHFResult(
            converged=converged,
            iterations=iteration,
            state_energies=state_energies,
            mixing_coefficients=mixing,
            ground_energy=state_energies[0],
            orbital_energies=orbital_energies,
            csfs=self.csfs,
            orbitals_dict=dict(orbitals_dict),
            bsp_data=dict(dhf.bsp_data) if dhf is not None else {},
            Z=self.Z,
            occupations=occupations,
            per_block_orbitals=per_block_orbitals,
            _grid=(np.asarray(shoot_result._grid)
                   if shoot_result is not None else None),
            _difr=(np.asarray(shoot_result._difr)
                   if shoot_result is not None else None),
        )
        result._nuclear_potential = dhf._nuclear_potential if dhf is not None else None
        result.__dict__["_variational_diagnostics"] = variational_diagnostics
        result.__dict__["_radial_optimization"] = self.radial_optimization
        result.__dict__["_inner_converged"] = inner_converged
        result.__dict__["_inner_residual"] = inner_residual
        # Attach correction components for downstream energy breakdown.
        # ``E_Breit`` is the shift that is in ``E_total``: the measured CI
        # operator shift on the operator route, the (fixed) scalar on the
        # scalar route.  The non-applied one is reported under a key whose
        # name says so.
        if E_Breit:
            if breit_operator_in_ci:
                result.__dict__["_E_Breit_scalar_not_applied"] = E_Breit
                if E_Breit_operator is not None:
                    result.__dict__["_E_Breit"] = E_Breit_operator
            else:
                result.__dict__["_E_Breit"] = E_Breit
        if self.include_breit:
            from atomx.methods.mcdhf.breit_ci import RETARDATION_NOTE
            result.__dict__["_breit_mode"] = self.breit_mode
            if breit_operator_in_ci:
                result.__dict__["_breit_provenance"] = {
                    "mode": "operator",
                    "operator": "gaunt-exact-in-blocks",
                    "scalar_shift_applied": False,
                    "applied_shift_source": (
                        "E(CI with Gaunt) − E(CI without Gaunt) on the "
                        "converged orbitals"
                        if E_Breit_operator is not None
                        else "not measured (per-block orbital sets), so "
                             "E_Breit is not reported"),
                    "note": RETARDATION_NOTE,
                }
            else:
                result.__dict__["_breit_provenance"] = {
                    "mode": "scalar",
                    "operator": "none",
                    "scalar_shift_applied": bool(E_Breit),
                    "note": (
                        "comparison path only: the fixed scalar Breit was "
                        "evaluated once on the Guerra-averaged orbitals "
                        "and added to every CI state as the same scalar, "
                        "so it cannot split levels."),
                }
        if qed_components:
            result.__dict__["_E_QED"] = qed_components
            result.__dict__["_qed_state_components"] = qed_state_components
            result.__dict__["_qed_provenance"] = {
                "vacuum_polarization": "full radial one-body matrix; first-order CI expectation",
                "self_energy": ("full radial local-model matrix; nonlocal SE absent"
                                if self.correction_stack.se_backend == "shabaev_local"
                                else "diagonal orbital model; off-diagonal self-energy absent"),
                "degeneracy": "correction not diagonalized within degenerate manifolds",
            }
            result.__dict__["_qed_state_shifts"] = np.asarray(
                qed_state_shifts, dtype=float)
        if sms_provenance is not None:
            result.__dict__["_sms_state_shifts"] = sms_state_shifts
            result.__dict__["_sms_provenance"] = sms_provenance
        if nms_provenance is not None:
            result.__dict__["_nms_state_shifts"] = nms_state_shifts
            result.__dict__["_nms_provenance"] = nms_provenance
        return result


    def _qed_state_components(self, *, mixing, orbitals_dict, corr_data, i0,
                              per_block_orbitals=None):
        """First-order QED components with full local-VP radial coherence.

        A central potential preserves kappa, but is generally not diagonal
        in principal/radial orbital labels. The local Uehling and model WK
        potentials therefore use the full one-body CSF matrix. The current
        self-energy backend supplies a full matrix when available and diagonal
        orbital shifts otherwise; provenance distinguishes these cases.
        """
        from atomx.corrections.qed import (
            self_energy_radial_matrix,
            short_range_radial_matrix,
            uehling_potential,
        )
        from atomx.methods.mcdhf.one_body import scalar_one_body_expectations

        stack = self.correction_stack
        if stack is None or corr_data is None:
            return {}
        if per_block_orbitals:
            groups = {}
            for state, dominant in enumerate(np.argmax(abs(mixing), axis=0)):
                csf = self.csfs[dominant]
                groups.setdefault((csf.total_J2, csf.parity), []).append(state)
            result = {}
            for block, states in groups.items():
                local = dict(orbitals_dict)
                local.update(per_block_orbitals.get(block, {}))
                components = self._qed_state_components(
                    mixing=mixing[:, states], orbitals_dict=local,
                    corr_data=corr_data, i0=i0)
                for name, values in components.items():
                    result.setdefault(name, np.zeros(mixing.shape[1]))[states] = values
            return result
        keys = tuple(sorted(orbitals_dict))
        orbitals = [(p[i0:], q[i0:], key[1])
                    for key in keys for p, q in [orbitals_dict[key]]]
        result = {}
        potentials = {}
        if stack.include_vp:
            potentials["E_VP"] = lambda r: uehling_potential(r, self.Z)
        if stack.include_wk_vp:
            from atomx.corrections.qedmod.flambaum_ginges import V_wichmann_kroll
            potentials["E_WK"] = lambda r: V_wichmann_kroll(r, self.Z)
        for name, potential in potentials.items():
            radial = short_range_radial_matrix(orbitals, corr_data, potential)
            result[name] = scalar_one_body_expectations(self.csfs, mixing, keys, radial)
        if stack.se_backend is not None:
            radial = self_energy_radial_matrix(
                [(p, q, kappa, Subshell(n, kappa, 1.).label)
                 for (n, kappa), (p, q, _) in zip(keys, orbitals, strict=True)],
                self.Z, corr_data, method=stack.se_backend)
            result["E_SE"] = scalar_one_body_expectations(
                self.csfs, mixing, keys, radial)
        return result

    def _qed_state_shifts(self, *, mixing, orbitals_dict, corr_data, i0):
        """Sum the separately inspectable first-order QED components."""
        components = self._qed_state_components(
            mixing=mixing, orbitals_dict=orbitals_dict, corr_data=corr_data, i0=i0)
        return sum(components.values(), np.zeros(np.asarray(mixing).shape[1]))

    def _diagonalize_per_block_ci(self, *, state_energies, mixing,
                                  per_block_orbitals, bsp_data, ns,
                                  nuclear_potential=None, convol_cache=None):
        """Replace same-(J, parity) CI subblocks with block-local orbitals."""
        energies = np.array(state_energies, dtype=float, copy=True)
        mixed = np.array(mixing, dtype=float, copy=True)
        if mixed.ndim != 2 or mixed.size == 0:
            return energies, mixed

        state_blocks: Dict[tuple[int, int], list[int]] = {}
        for state in range(mixed.shape[1]):
            dom = int(np.argmax(np.abs(mixed[:, state])))
            csf = self.csfs[dom]
            block = (int(getattr(csf, "total_J2", 0)), int(getattr(csf, "parity", 1)))
            state_blocks.setdefault(block, []).append(state)

        for block, orbitals in per_block_orbitals.items():
            csf_indices = [
                i for i, csf in enumerate(self.csfs)
                if int(getattr(csf, "total_J2", 0)) == block[0]
                and int(getattr(csf, "parity", 1)) == block[1]
            ]
            state_indices = sorted(state_blocks.get(block, []), key=lambda i: energies[i])
            if not csf_indices or not state_indices:
                continue
            H = build_ci_matrix(
                [self.csfs[i] for i in csf_indices],
                orbitals,
                self.orbital_basis,
                bsp_data,
                self.Z,
                ns,
                nuclear_potential=nuclear_potential,
                convol_cache=convol_cache,
                include_breit=self._breit_in_ci,
            )
            vals, vecs = linalg.eigh(H)
            count = min(len(state_indices), len(vals))
            for local_state in range(count):
                global_state = state_indices[local_state]
                energies[global_state] = float(vals[local_state])
                mixed[:, global_state] = 0.0
                for local_csf, global_csf in enumerate(csf_indices):
                    mixed[global_csf, global_state] = vecs[local_csf, local_state]

        return energies, mixed

    def _solve_per_block_orbitals(self, *, mixing, shared_orbitals, dhf):
        """DODS / EOL per-(J, parity)-block orbital optimisation — scaffold.

        HYP-184 follow-up item (2).  In the canonical Guerra averaged-level
        MCDHF, ``MCDHFSolver.solve`` produces one shared orbital set
        optimised against the *flat* CI mixing weights.  For Fe XVII
        2p⁵3l this means the excited-state 3l electron is solved in
        the closed-shell 2p⁶ field rather than the open 2p⁵3l field,
        pulling 2p→3l excitation energies ~5–8 % low and producing the
        canonical ~1 Å redshift on the 3C wavelength (16.31 Å vs
        Bernitt 15.014 Å).

        The fix is to run a separate Guerra-averaged DHF *per
        (J, parity) block*, using only that block's CSFs (with their
        block-internal mixing weights), and to expose each
        block-specific orbital set on
        :attr:`MCDHFResult.per_block_orbitals` as
        ``{(twoJ, parity): {(n, kappa): (P, Q)}}``.  The transition
        calculator then applies a biorthogonal transformation between
        the initial-state and final-state orbital sets when computing
        radiative matrix elements (see
        :mod:`atomx.properties.transitions.calculator`).

        **Current status.**  Implemented: each dominant ``(twoJ, parity)``
        block gets its own Guerra-averaged DHF solve on that block's CSFs
        (the block's CI weights, renormalised), and the per-block orbital
        sets are returned for the caller to store on
        :attr:`MCDHFResult.per_block_orbitals`.  Like the shared-orbital
        pass, each block solve consumes the R7 asymmetric mean field
        (:meth:`_mean_field_matrices`), so a block-local correlation
        orbital with no Guerra population is solved in that block's
        reference (first-CSF) potential rather than against a bare
        nucleus.

        Parameters
        ----------
        mixing : np.ndarray
            ``(n_csf, n_state)`` CI mixing coefficients.  Used to
            extract per-block weights ``w^block_i = c_i^state_in_block``.
        shared_orbitals : dict
            ``{(n, kappa): (P, Q)}`` Guerra-averaged orbital set
            already solved in :meth:`solve`.  Not consumed: each block
            re-bootstraps from the solver's own starter (per-block
            seeding is a follow-up).
        dhf : DFSolver
            The DFSolver instance used in the shared-orbital pass.
            Carries the bsp_data / grid that per-block solves should
            reuse for orbital-pair compatibility.

        Returns
        -------
        dict
            ``{(twoJ, parity): {(n, kappa): (P, Q)}}`` — empty when
            there is no mixing to block up.
        """
        del shared_orbitals
        if dhf is None or mixing is None or np.asarray(mixing).size == 0:
            return {}

        mixing_arr = np.asarray(mixing, dtype=float)
        dominant_blocks: Dict[tuple[int, int], list[int]] = {}
        for state in range(mixing_arr.shape[1]):
            dom = int(np.argmax(np.abs(mixing_arr[:, state])))
            csf = self.csfs[dom]
            block = (int(getattr(csf, "total_J2", 0)), int(getattr(csf, "parity", 1)))
            dominant_blocks.setdefault(block, []).append(state)

        out: Dict = {}
        for block, states in dominant_blocks.items():
            weights = np.zeros(self.n_csf, dtype=float)
            for state in states:
                weights += mixing_arr[:, state] ** 2
            block_mask = np.array([
                int(getattr(csf, "total_J2", 0)) == block[0]
                and int(getattr(csf, "parity", 1)) == block[1]
                for csf in self.csfs
            ], dtype=bool)
            weights = np.where(block_mask, weights, 0.0)
            total = float(weights.sum())
            if total <= 0.0:
                continue
            weights /= total

            avg_subshells = self._averaged_subshells(weights=weights)
            block_jj_configs = self._jj_configs(avg_subshells, weights)
            block_coeffs, block_norm = self._mean_field_matrices(
                avg_subshells, weights, block_jj_configs)
            kwargs = {}
            if self.r_min is not None:
                kwargs["r_min"] = self.r_min
            if self.r_max is not None:
                kwargs["r_max"] = self.r_max
            block_dhf = DFSolver(
                Z=self.Z, subshells=avg_subshells,
                ns=self.ns, ks=self.ks,
                exchange="hf",
                include_breit=False, include_qed=False,
                nuclear=self.nuclear,
                starting_potential=(self.starting_potential
                                    or ("tfda" if self.Z >= 20 else None)),
                jj_configs=block_jj_configs,
                mean_field_coeffs=block_coeffs,
                mean_field_normalization=block_norm,
                screening_potential=self.screening_potential,
                **kwargs,
            )
            if self.use_physics_grid:
                from atomx.numerics.bspline.basis import prepare_bspline_data
                from atomx.numerics.bspline.knots import make_physics_knots
                t_phys, ns_phys = make_physics_knots(
                    ks=self.ks, Z=self.Z, nuclear=self.nuclear)
                data = prepare_bspline_data(t_phys, ns_phys, self.ks,
                                            kinetic_balance=True)
                block_dhf.t = t_phys
                block_dhf.bsp_data = data
                block_dhf.ns = ns_phys
                block_dhf._precompute()

            block_result = block_dhf.solve(max_iter=20, tol=1.0e-5)
            # Keyed by (n, kappa) via SCFResult's own subshell bookkeeping
            # (never positional per-kappa counters).
            orbitals_dict: Dict = {}
            for (n, kappa), (P, Q, _occ) in block_result.orbitals_by_subshell().items():
                P, Q = canonicalize_radial_phase(P, Q)
                orbitals_dict[(n, kappa)] = (P, Q)
            if orbitals_dict:
                out[block] = orbitals_dict

        return out
