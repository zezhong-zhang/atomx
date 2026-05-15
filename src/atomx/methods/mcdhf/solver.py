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
                 nuclear="fermi", use_physics_grid=True,
                 starting_potential=None):
        self.Z = Z
        self.csfs = csfs
        self.n_csf = len(csfs)
        self.ns = ns
        self.ks = ks
        self.include_breit = include_breit
        self.include_qed = include_qed
        self.orbital_basis = get_shared_orbital_basis(csfs)
        self.r_min = r_min
        self.r_max = r_max
        self.nuclear = nuclear
        self.use_physics_grid = use_physics_grid
        self.starting_potential = starting_potential

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
        converged = False
        orbital_energies: Dict[str, float] = {}
        mixing = np.eye(self.n_csf)
        state_energies = np.zeros(self.n_csf)
        iteration = 0
        orbitals_dict: Dict = {}
        dhf = None
        seed_orbitals = None  # warm-start carry-over between outer iterations

        for iteration in range(1, max_iter + 1):
            # Step 1: Effective occupations (Guerra Eq 10)
            avg_subshells = self._averaged_subshells(weights)

            # Step 2: Solve DHF with fractional occupations. Skip Breit/QED
            # here — they are perturbative in atomx's DFSolver (do not enter
            # the Fock matrix), so they only affect the final energy. We
            # compute them ONCE after the MCDHF outer loop converges.
            kwargs = {}
            if self.r_min is not None:
                kwargs["r_min"] = self.r_min
            if self.r_max is not None:
                kwargs["r_max"] = self.r_max

            # Build Guerra-weighted JJConfig list so DFSolver.compute_WW
            # realises ϖᵢⱼ = Σⱼ cⱼ² nᵢʲ nⱼʲ — the correct
            # averaged-monoconfiguration Fock operator (Guerra Eq 10).
            jj_configs = self._jj_configs(avg_subshells, weights)

            dhf = DFSolver(
                Z=self.Z, subshells=avg_subshells,
                ns=self.ns, ks=self.ks,
                exchange="hf",
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

            # Step 6: Update weights = cⱼ² (ground state mixing)
            weights = mixing[:, 0] ** 2
            weights /= weights.sum()

            # Convergence check
            E_new = state_energies[0]
            energy_change = abs(E_new - E_old)
            E_old = E_new

            if iteration > 1 and energy_change < ci_tol:
                converged = True
                break

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
        if dhf is not None and (self.include_breit or self.include_qed):
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
            if self.include_breit:
                E_Breit = float(compute_breit_energy(orb_list, dhf.bsp_data, self.Z))
            if self.include_qed:
                qed_components = dict(compute_qed_corrections(
                    orb_list, dhf.bsp_data, self.Z,
                ))

        result = MCDHFResult(
            converged=converged,
            iterations=iteration,
            state_energies=state_energies,
            mixing_coefficients=mixing,
            ground_energy=state_energies[0] + E_Breit + qed_components.get("E_QED_total", 0.0),
            orbital_energies=orbital_energies,
            csfs=self.csfs,
            orbitals_dict=dict(orbitals_dict),
            bsp_data=dict(dhf.bsp_data) if dhf is not None else {},
            Z=self.Z,
            occupations=occupations,
        )
        # Attach correction components for downstream energy breakdown.
        if E_Breit:
            result.__dict__["_E_Breit"] = E_Breit
        if qed_components:
            result.__dict__["_E_QED"] = qed_components
        return result
