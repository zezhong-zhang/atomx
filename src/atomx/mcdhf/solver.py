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

from atomx.scf.solver import DFSolver, Subshell, SCFResult
from atomx.mcdhf.config import CSF, get_shared_orbital_basis
from atomx.mcdhf.matrix import build_ci_matrix


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
    def wavefunctions(self):
        """Build Wavefunctions container from orbital coefficients."""
        from atomx.results import Wavefunctions, OrbitalWavefunction
        from atomx.scf.solver import _eval_bspline_on_grid
        if not self.orbitals_dict or not self.bsp_data:
            return Wavefunctions(grid=np.array([]))

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
                P=P_grid, Q=Q_grid, n=n, l=l,
                kappa=kappa,
                occupation=0.0,  # fractional in MCDHF
                energy=self.orbital_energies.get(label, 0.0),
                label=label)
        return Wavefunctions(grid=grid, orbitals=orbitals, basis_type="bspline")

    def excitation_energies(self, unit="hartree"):
        from atomx.constants import HARTREE_EV, HARTREE_CM1
        de = self.state_energies - self.state_energies[0]
        if unit == "ev":
            return de * HARTREE_EV
        elif unit == "cm-1":
            return de * HARTREE_CM1
        return de


class MCDHFSolver:
    """Multi-Configuration Dirac-Hartree-Fock solver.

    Uses Average-Level (AL) approach with Guerra effective occupations.
    """

    def __init__(self, Z, csfs, ns=50, ks=9, r_min=None, r_max=None,
                 include_breit=False, include_qed=False):
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

    def solve(self, max_iter=30, scf_tol=1e-5, ci_tol=1e-8, scf_max_iter=40):
        """Run the MCDHF calculation.

        The key improvement: directly uses converged DHF orbitals from
        SCFResult.orbitals_raw, avoiding monkey-patching.
        """
        weights = np.ones(self.n_csf) / self.n_csf
        E_old = 0.0
        converged = False
        orbital_energies = {}
        mixing = np.eye(self.n_csf)
        state_energies = np.zeros(self.n_csf)

        for iteration in range(1, max_iter + 1):
            # Step 1: Effective occupations (Guerra Eq 10)
            avg_subshells = self._averaged_subshells(weights)

            # Step 2: Solve DHF with fractional occupations
            kwargs = {}
            if self.r_min is not None:
                kwargs["r_min"] = self.r_min
            if self.r_max is not None:
                kwargs["r_max"] = self.r_max

            dhf = DFSolver(
                Z=self.Z, subshells=avg_subshells,
                ns=self.ns, ks=self.ks,
                exchange="hf", **kwargs
            )
            dhf_result = dhf.solve(max_iter=scf_max_iter, tol=scf_tol)
            orbital_energies = dhf_result.orbital_energies

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

            # Step 4: Build CI matrix
            H_ci = build_ci_matrix(
                self.csfs, orbitals_dict, self.orbital_basis,
                dhf.bsp_data, self.Z, dhf.ns
            )

            # Step 5: Diagonalize CI
            state_energies, mixing = linalg.eigh(H_ci)

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

        return MCDHFResult(
            converged=converged,
            iterations=iteration,
            state_energies=state_energies,
            mixing_coefficients=mixing,
            ground_energy=state_energies[0],
            orbital_energies=orbital_energies,
            csfs=self.csfs,
            orbitals_dict=dict(orbitals_dict),
            bsp_data=dict(dhf.bsp_data),
        )
