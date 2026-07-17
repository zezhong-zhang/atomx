"""MCDHF solver: Average-Level orbital optimization + CI diagonalization.

Implements the Guerra averaged monoconfiguration approach:
  Eq (9):  ρ(r) = Σᵢ ϖᵢ (Pᵢ² + Qᵢ²)
  Eq (10): ϖᵢ = Σⱼ cⱼ² nᵢʲ

Algorithm:
1. Compute effective occupations ϖᵢ from CI mixing coefficients cⱼ
2. Solve DHF with fractional occupations (averaged monoconfig)
3. Build CI matrix using converged shared orbitals
4. Diagonalize CI → state energies + new cⱼ
5. Iterate until energy converges
"""

import numpy as np
from scipy import linalg
from dataclasses import dataclass, field
from typing import List, Dict

from atomx.methods.dirac_fock import DFSolver, Subshell, SCFResult
from atomx.methods.mcdhf.config import CSF, get_shared_orbital_basis
from atomx.methods.mcdhf.matrix import build_ci_matrix
from atomx.methods.ci import ci_solve
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
    def energy_breakdown(self) -> Dict[str, float]:
        return {"E_total": self.ground_energy}

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
        """Build :class:`atomx.results.Waves` container from orbital coefficients."""
        from atomx.results import Waves, OrbitalWavefunction
        from atomx.methods.dirac_fock import _eval_bspline_on_grid
        if not self.orbitals_dict or not self.bsp_data:
            return Waves(grid=np.array([]))

        gr = self.bsp_data["gr"]
        grid = gr.ravel()

        orbitals = {}
        for (n, kappa), (P_coeffs, Q_coeffs) in self.orbitals_dict.items():
            l = kappa if kappa > 0 else -kappa - 1
            j2 = 2 * abs(kappa) - 1
            l_char = "spdfghiklmno"[l] if l < 12 else f"l{l}"
            label = f"{n}{l_char}{j2}/2"
            P_grid = _eval_bspline_on_grid(P_coeffs, self.bsp_data, "p")
            Q_grid = _eval_bspline_on_grid(Q_coeffs, self.bsp_data, "q")
            orbitals[label] = OrbitalWavefunction(
                P=P_grid, Q=Q_grid,
                n=n, l=l, kappa=kappa,
                # Guerra-averaged ground-state effective occupation;
                # 0.0 if the solver did not populate ``occupations``
                # (back-compat for callers constructing MCDHFResult by hand).
                occupation=float(self.occupations.get((n, kappa), 0.0)),
                energy=self.orbital_energies.get(label, 0.0),
                label=label)
        return Waves(grid=grid, orbitals=orbitals, basis_type="bspline")

    def excitation_energies(self, unit="hartree"):
        from atomx.constants import HARTREE_EV, HARTREE_CM1
        de = self.state_energies - self.state_energies[0]
        if unit == "ev":
            return de * HARTREE_EV
        elif unit == "cm-1":
            return de * HARTREE_CM1
        return de

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

    Uses Average-Level (AL) approach with Guerra effective occupations.
    """

    def __init__(self, Z, csfs, ns=50, ks=9, r_min=None, r_max=None,
                 include_breit=False, include_qed=False,
                 correction_stack=None,
                 nuclear="fermi", use_physics_grid=True,
                 starting_potential=None,
                 per_block_orbital_sets=False,
                 radial_optimization="mcdhf"):
        self.Z = Z
        self.csfs = csfs
        self.n_csf = len(csfs)
        self.ns = ns
        self.ks = ks
        self.include_breit = include_breit
        self.include_qed = include_qed
        # Single configuration surface for post-SCF corrections (review
        # E1).  MCDHF's Guerra-averaged occupations are not expressible
        # through the result-based adapters yet, so the stack is consumed
        # as configuration by _apply_corrections_to_orbitals below.
        if correction_stack is not None:
            self.correction_stack = correction_stack
        elif include_breit or include_qed:
            from atomx.corrections.qed import DEFAULT_SE_METHOD, resolve_se_method
            from atomx.corrections.stack import CorrectionStack
            self.correction_stack = CorrectionStack(
                se_backend=(resolve_se_method(DEFAULT_SE_METHOD, Z)
                            if include_qed else None),
                include_vp=bool(include_qed),
                include_wk_vp=False,
                include_breit=bool(include_breit),
            )
        else:
            self.correction_stack = None
        self.orbital_basis = get_shared_orbital_basis(csfs)
        self.r_min = r_min
        self.r_max = r_max
        self.nuclear = nuclear
        self.use_physics_grid = use_physics_grid
        self.starting_potential = starting_potential
        if radial_optimization not in {"mcdhf", "fac_frozen", "fac_frozen_local"}:
            raise ValueError(
                "radial_optimization must be 'mcdhf', 'fac_frozen', or 'fac_frozen_local' "
                f"(got {radial_optimization!r})"
            )
        self.radial_optimization = radial_optimization
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

    def _averaged_subshells(self, weights=None):
        """Guerra Eq (10): ϖᵢ = Σⱼ cⱼ² nᵢʲ — fractional occupations.

        All orbitals appearing in any CSF are included (with minimum
        occupation 0.001) so they get solved even when their weight is
        near-zero. This is essential for transition calculations.
        """
        if weights is None:
            weights = np.ones(self.n_csf) / self.n_csf

        avg_occ = {}
        for i, csf in enumerate(self.csfs):
            for sh in csf.subshells:
                key = (sh.n, sh.kappa)
                avg_occ[key] = avg_occ.get(key, 0.0) + weights[i] * sh.occupation

        subshells = []
        for (n, kappa), occ in sorted(avg_occ.items(),
                                       key=lambda x: (x[0][0], abs(x[0][1]))):
            max_q = 2 * abs(kappa)
            # Ensure all orbitals are solved (min occupation for numerical stability)
            occ = max(occ, 0.001)
            occ = min(occ, max_q)
            subshells.append(Subshell(n=n, kappa=kappa, occupation=occ))

        return subshells

    def _fac_frozen_subshells(self):
        """FAC-like radial set: optimise the first CSF and keep virtuals tiny.

        FAC structure scripts commonly call ``OptimizeRadial`` on the closed
        ground group and then generate excited orbitals in that frozen
        central potential. This helper builds the analogous subshell list:
        occupations from the first CSF, plus tiny non-zero occupancies for
        every additional orbital needed by the CI basis so DFSolver solves
        those orbitals without relaxing the core toward the excited manifold.
        """
        ref_occ = {}
        if self.csfs:
            for sh in self.csfs[0].subshells:
                ref_occ[(sh.n, sh.kappa)] = float(sh.occupation)

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
        right occupation. For CSFs that don't include a given orbital, we
        insert it with occupation 0 so the positional index is preserved.
        """
        from atomx.config.jj_coupling import JJConfig
        order = [(s.n, s.kappa) for s in avg_subshells]
        configs = []
        for i, csf in enumerate(self.csfs):
            occ_map = {(sh.n, sh.kappa): sh.occupation for sh in csf.subshells}
            positional = [
                Subshell(n=n, kappa=kappa,
                         occupation=float(occ_map.get((n, kappa), 0)))
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
        orbital_energies: Dict[str, float] = {}
        mixing = np.eye(self.n_csf)
        state_energies = np.zeros(self.n_csf)
        iteration = 0
        orbitals_dict: Dict = {}
        dhf = None
        seed_orbitals = None  # warm-start carry-over between outer iterations

        for iteration in range(1, max_iter + 1):
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

            # Step 2: Solve DHF with fractional occupations. Skip Breit/QED
            # here — they are perturbative in atomx's DFSolver (do not enter
            # the Fock matrix), so they only affect the final energy. We
            # compute them ONCE after the MCDHF outer loop converges.
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
            # experiment where excited orbitals are solved but do not relax
            # the closed core appreciably.

            dhf = DFSolver(
                Z=self.Z, subshells=avg_subshells,
                ns=self.ns, ks=self.ks,
                exchange=exchange_model,
                include_breit=False, include_qed=False,
                nuclear=self.nuclear,
                starting_potential=(self.starting_potential
                                    or ("tfda" if self.Z >= 20 else None)),
                jj_configs=jj_configs,
                **kwargs,
            )

            # Install the DBSR-HF-style semi-exponential physics grid before
            # _precompute(), so the inner DHF matches the DFSolver path used
            # for averaging in {"statistical","equal","fractional"}.
            if self.use_physics_grid:
                from atomx.numerics.bspline.knots import make_physics_knots
                from atomx.numerics.bspline.basis import prepare_bspline_data
                t_phys, ns_phys = make_physics_knots(
                    ks=self.ks, Z=self.Z, nuclear=self.nuclear)
                data = prepare_bspline_data(t_phys, ns_phys, self.ks,
                                            kinetic_balance=True)
                dhf.t = t_phys
                dhf.bsp_data = data
                dhf.ns = ns_phys
                dhf._precompute()

            # Warm-start from the previous outer iteration's converged
            # orbitals. Only applied when the basis dimension matches; the
            # grid is rebuilt identically each outer iter, so it does.
            if seed_orbitals is not None:
                dhf.seed_orbitals = seed_orbitals

            dhf_result = dhf.solve(max_iter=scf_max_iter, tol=scf_tol)
            orbital_energies = dhf_result.orbital_energies
            # Stash for the next outer iteration
            seed_orbitals = dhf_result.orbitals_raw

            # Step 3: Extract converged orbitals from SCFResult
            orbitals_dict = {}
            kappa_counter = {}
            for sh in avg_subshells:
                idx = kappa_counter.get(sh.kappa, 0)
                kappa_counter[sh.kappa] = idx + 1
                orb_list = dhf_result.orbitals_raw.get(sh.kappa, [])
                if idx < len(orb_list):
                    P, Q, occ = orb_list[idx]
                    P, Q = canonicalize_radial_phase(P, Q)
                    orbitals_dict[(sh.n, sh.kappa)] = (P, Q)

            if not orbitals_dict:
                break  # no orbitals found

            # Steps 4 + 5: CI matrix build + diagonalise via methods/ci.
            ci_res = ci_solve(
                self.csfs, orbitals_dict, self.orbital_basis,
                dhf.bsp_data, self.Z, dhf.ns,
            )
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
                converged = True
                break

            if iteration > 1 and energy_change < ci_tol:
                E_old = E_new
                converged = True
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

        # Compute ground-state Guerra-averaged effective occupations,
        # so MCDHFResult.waves can carry physically meaningful
        # per-orbital occupations (used by Atom.density, etc.).
        ground_weights = mixing[:, 0] ** 2
        ground_weights = ground_weights / ground_weights.sum()
        ground_subshells = self._averaged_subshells(weights=ground_weights)
        occupations = {(sh.n, sh.kappa): float(sh.occupation) for sh in ground_subshells}

        # One-shot perturbative Breit + QED on the final converged orbitals.
        # These corrections do not enter the Fock matrix in atomx's DHF, so
        # doing them once here (rather than inside each outer iteration's
        # DFSolver) is both physically equivalent and N× faster, where N is
        # the MCDHF outer-iteration count.
        E_Breit = 0.0
        qed_components: Dict[str, float] = {}
        stack = self.correction_stack
        if dhf is not None and stack is not None:
            from atomx.corrections.breit import compute_breit_energy
            from atomx.corrections.qed import compute_qed_corrections
            # Flatten orbitals on the final basis for the correction APIs.
            orb_list = []
            kappa_counter: Dict[int, int] = {}
            for sh in ground_subshells:
                idx = kappa_counter.get(sh.kappa, 0)
                kappa_counter[sh.kappa] = idx + 1
                cached = orbitals_dict.get((sh.n, sh.kappa))
                if cached is not None:
                    P, Q = cached
                    orb_list.append((P, Q, sh.occupation, sh.kappa, sh.label))
            if stack.include_breit:
                E_Breit = float(compute_breit_energy(orb_list, dhf.bsp_data, self.Z))
            if stack.se_backend is not None or stack.include_vp:
                qed_components = dict(compute_qed_corrections(
                    orb_list, dhf.bsp_data, self.Z,
                    se_method=stack.se_backend or "auto",
                    include_wichmann_kroll=stack.include_wk_vp,
                ))

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
                )

        # The one-shot corrections are computed from the shared
        # Guerra-averaged orbital set, so to this order they shift every
        # CI state by the same scalar.  Fold the shift into
        # ``state_energies`` as well, preserving the AtomicResult
        # invariant ``state_energies[0] == total_energy`` (excitation
        # energies are unchanged by a uniform shift).
        e_corr = E_Breit + qed_components.get("E_QED_total", 0.0)
        if e_corr:
            state_energies = state_energies + e_corr

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
        )
        # Attach correction components for downstream energy breakdown.
        if E_Breit:
            result.__dict__["_E_Breit"] = E_Breit
        if qed_components:
            result.__dict__["_E_QED"] = qed_components
        return result


    def _diagonalize_per_block_ci(self, *, state_energies, mixing, per_block_orbitals, bsp_data, ns):
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

        **Current status — scaffold only.**  Returns an empty dict;
        the caller leaves the shared orbital set in place.

        Parameters
        ----------
        mixing : np.ndarray
            ``(n_csf, n_state)`` CI mixing coefficients.  Used to
            extract per-block weights ``w^block_i = c_i^state_in_block``.
        shared_orbitals : dict
            ``{(n, kappa): (P, Q)}`` Guerra-averaged orbital set
            already solved in :meth:`solve`.  Useful as a warm-start
            seed for the per-block DHF.
        dhf : DFSolver
            The DFSolver instance used in the shared-orbital pass.
            Carries the bsp_data / grid that per-block solves should
            reuse for orbital-pair compatibility.

        Returns
        -------
        dict
            ``{(twoJ, parity): {(n, kappa): (P, Q)}}`` — empty when
            the implementation is not yet filled in.
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
                jj_configs=self._jj_configs(avg_subshells, weights),
                **kwargs,
            )
            if self.use_physics_grid:
                from atomx.numerics.bspline.knots import make_physics_knots
                from atomx.numerics.bspline.basis import prepare_bspline_data
                t_phys, ns_phys = make_physics_knots(
                    ks=self.ks, Z=self.Z, nuclear=self.nuclear)
                data = prepare_bspline_data(t_phys, ns_phys, self.ks,
                                            kinetic_balance=True)
                block_dhf.t = t_phys
                block_dhf.bsp_data = data
                block_dhf.ns = ns_phys
                block_dhf._precompute()

            block_result = block_dhf.solve(max_iter=20, tol=1.0e-5)
            orbitals_dict: Dict = {}
            kappa_counter: Dict[int, int] = {}
            for sh in avg_subshells:
                idx = kappa_counter.get(sh.kappa, 0)
                kappa_counter[sh.kappa] = idx + 1
                orb_list = block_result.orbitals_raw.get(sh.kappa, [])
                if idx < len(orb_list):
                    P, Q, _occ = orb_list[idx]
                    P, Q = canonicalize_radial_phase(P, Q)
                    orbitals_dict[(sh.n, sh.kappa)] = (P, Q)
            if orbitals_dict:
                out[block] = orbitals_dict

        return out
