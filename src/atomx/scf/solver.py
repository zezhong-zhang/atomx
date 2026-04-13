"""Dirac-Fock self-consistent field solver with non-local HF exchange.

Two modes:
- DFS (Dirac-Fock-Slater): local exchange V_x ∝ -ρ^{1/3} — fast, approximate
- DHF (Dirac-Hartree-Fock): non-local exchange via Y^k — accurate, slower

Total energy computed as:
    E = Σ_a q_a ε_a^{1e} + (1/2) Σ_{a,b} [q_a*q_b*F^0(a,b) + Σ_k c^k * G^k(a,b)]
"""

import numpy as np
from scipy import linalg
from dataclasses import dataclass, field
from typing import List, Dict

from atomx.constants import C, C_SQ
from atomx.bspline.knots import make_knots
from atomx.bspline.integrals import overlap_matrix, nuclear_matrix, _assemble_integral
from atomx.dirac.matrix import build_dirac_matrices, prepare_bspline_data
from atomx.dirac.solver import solve_dirac
from atomx.scf.slater import (
    compute_yk, eval_orbital_on_grid,
    compute_exchange_matrix, compute_slater_fk, compute_slater_gk,
)
from atomx.scf.angular_coeffs import (
    exchange_coefficient, exchange_k_range, direct_coefficient,
)


@dataclass
class Subshell:
    """An atomic subshell (n, kappa) with occupation.

    Supports fractional occupations for averaged monoconfiguration mode
    (Guerra Eq 10: ϖᵢ = Σⱼ cⱼ² nᵢʲ).
    """
    n: int
    kappa: int
    occupation: float  # float to support fractional (Guerra averaged monoconfig)
    label: str = ""

    def __post_init__(self):
        l = self.kappa if self.kappa > 0 else -self.kappa - 1
        j2 = 2 * abs(self.kappa) - 1
        max_occ = j2 + 1
        if self.occupation > max_occ + 0.01:
            raise ValueError(f"Occupation {self.occupation} exceeds max {max_occ} for kappa={self.kappa}")
        if not self.label:
            l_char = "spdfghiklmno"[l] if l < 12 else f"l{l}"
            self.label = f"{self.n}{l_char}{j2}/2"

    @property
    def l(self):
        return self.kappa if self.kappa > 0 else -self.kappa - 1

    @property
    def j2(self):
        return 2 * abs(self.kappa) - 1


@dataclass
class SCFResult:
    """Result of an SCF calculation."""
    converged: bool
    iterations: int
    residual: float
    total_energy: float
    orbital_energies: Dict[str, float]
    subshells: List[Subshell]
    energy_components: Dict[str, float] = field(default_factory=dict)
    orbitals_raw: Dict = field(default_factory=dict)  # kappa -> [(P, Q, occ)]


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
        "slater" for local Slater exchange with alpha_x parameter.
    alpha_x : float
        Slater exchange parameter (only used when exchange="slater").
    """

    def __init__(self, Z, subshells, ns=50, ks=9, r_min=None, r_max=None,
                 exchange="hf", alpha_x=2.0/3.0,
                 include_breit=False, include_qed=False):
        self.Z = Z
        self.subshells = subshells
        self.ns = ns
        self.ks = ks
        self.r_min = r_min if r_min is not None else 0.05 / Z
        self.r_max = r_max if r_max is not None else max(40.0, 60.0 / Z)
        self.exchange = exchange
        self.alpha_x = alpha_x
        self.include_breit = include_breit
        self.include_qed = include_qed
        self.N_electrons = sum(sh.occupation for sh in subshells)

        self.t = make_knots(ns, ks, self.r_min, self.r_max)
        self.bsp_data = prepare_bspline_data(self.t, ns, ks)
        self._precompute()

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

    def _eval_on_grid(self, coeffs):
        """Evaluate orbital using P-component basis (backward compat)."""
        return self._eval_P_on_grid(coeffs)

    def _build_direct_potential(self, orbitals):
        """Build Hartree potential matrices for P and Q blocks separately.

        With KB: V_J_PP uses ksp-order basis, V_J_QQ uses ksq-order.
        Returns (V_J_PP, V_J_QQ).
        """
        nv = self.bsp_data["nv"]
        ns = self.ns
        gr = self.bsp_data["gr"]
        grw = self.bsp_data["grw"]
        n_gauss = gr.shape[1]

        # Build total density using component-specific evaluation
        rho_total = np.zeros((nv, n_gauss))
        for kappa, orb_list in orbitals.items():
            for P, Q, occ in orb_list:
                fP = self._eval_P_on_grid(P)
                fQ = self._eval_Q_on_grid(Q)
                rho_total += occ * (fP ** 2 + fQ ** 2)

        yk = compute_yk(rho_total, gr, grw, nv, n_gauss, k=0)
        yk_weights = grw * yk

        # V_J for P-block
        ksp = self.bsp_data.get("ksp", self.ks)
        bsp_p = self.bsp_data["bsp_p"]
        V_J_PP = _assemble_integral(nv, ksp, ksp, ns, bsp_p, bsp_p, yk_weights)
        V_J_PP = 0.5 * (V_J_PP + V_J_PP.T)

        # V_J for Q-block
        ksq = self.bsp_data.get("ksq", self.ks)
        bsp_q = self.bsp_data.get("bsp_q", bsp_p)
        V_J_QQ = _assemble_integral(nv, ksq, ksq, ns, bsp_q, bsp_q, yk_weights)
        V_J_QQ = 0.5 * (V_J_QQ + V_J_QQ.T)

        return V_J_PP, V_J_QQ

    def _build_exchange_matrix_hf(self, target_kappa, orbitals):
        """Build non-local HF exchange matrix for a given kappa channel.

        For the closed-shell Fock operator:
          F = h + V_Hartree(total) - V_exchange
          V_exchange = Σ_b Σ_k [Λ(k)/(2j_b+1)] * K^k(b)

        The coefficient Λ/(2j_b+1) gives one exchange per spinorbital,
        averaged over m_j. This correctly cancels self-interaction:
        for same-subshell, V_J(q electrons) - V_K gives V_J(q-1 electrons).
        """
        from atomx.scf.angular_coeffs import exchange_lambda

        ns = self.ns
        K_PP = np.zeros((ns, ns))
        K_QQ = np.zeros((ns, ns))

        for kappa_b, orb_list in orbitals.items():
            tj_b = 2 * abs(kappa_b) - 1
            for P_b, Q_b, occ_b in orb_list:
                k_values = exchange_k_range(target_kappa, kappa_b)
                for k in k_values:
                    lam = exchange_lambda(target_kappa, kappa_b, k)
                    if lam < 1e-15:
                        continue
                    c_k = lam / (tj_b + 1.0)
                    Kk_PP, Kk_QQ = compute_exchange_matrix(
                        P_b, Q_b, self.bsp_data, k)
                    # K_PP is (nsp x nsp), K_QQ is (nsq x nsq) — pad to ns
                    n_pp = min(Kk_PP.shape[0], ns)
                    n_qq = min(Kk_QQ.shape[0], ns)
                    K_PP[:n_pp, :n_pp] += c_k * Kk_PP[:n_pp, :n_pp]
                    K_QQ[:n_qq, :n_qq] += c_k * Kk_QQ[:n_qq, :n_qq]

        K_PP = 0.5 * (K_PP + K_PP.T)
        K_QQ = 0.5 * (K_QQ + K_QQ.T)
        return K_PP, K_QQ

    def _build_exchange_matrix_slater(self, orbitals):
        """Build local Slater exchange — separate PP and QQ matrices for KB."""
        nv = self.bsp_data["nv"]
        ns = self.ns
        gr = self.bsp_data["gr"]
        grw = self.bsp_data["grw"]
        n_gauss = gr.shape[1]

        rho = np.zeros((nv, n_gauss))
        for kappa, orb_list in orbitals.items():
            for P, Q, occ in orb_list:
                fP = self._eval_P_on_grid(P)
                fQ = self._eval_Q_on_grid(Q)
                rho += occ * (fP ** 2 + fQ ** 2)

        vx = np.zeros_like(rho)
        for iv in range(nv):
            for m in range(n_gauss):
                if rho[iv, m] > 1e-30:
                    vx[iv, m] = -self.alpha_x * (3.0 / np.pi * rho[iv, m]) ** (1.0 / 3.0)
        vx_w = grw * vx

        ksp = self.bsp_data.get("ksp", self.ks)
        bsp_p = self.bsp_data["bsp_p"]
        Vx_PP = _assemble_integral(nv, ksp, ksp, ns, bsp_p, bsp_p, vx_w)
        Vx_PP = 0.5 * (Vx_PP + Vx_PP.T)

        ksq = self.bsp_data.get("ksq", self.ks)
        bsp_q = self.bsp_data.get("bsp_q", bsp_p)
        Vx_QQ = _assemble_integral(nv, ksq, ksq, ns, bsp_q, bsp_q, vx_w)
        Vx_QQ = 0.5 * (Vx_QQ + Vx_QQ.T)

        return Vx_PP, Vx_QQ

    def _compute_total_energy(self, orbitals, orbital_energies=None):
        """Compute total energy using orbital eigenvalue formula (KB-safe).

        E_total = Σ_a q_a ε_a - E_J + E_K

        where ε_a is the Fock eigenvalue. This avoids matrix expectation
        values that can be numerically inaccurate with kinetic balance.

        E_1e is derived as: Σ_a q_a ε_a - 2*(E_J - E_K)
        """
        orb_list = self._flatten_orbitals(orbitals)
        bsp_data = self.bsp_data

        # Sum of orbital eigenvalues: Σ q_a ε_a
        E_orbital_sum = 0.0
        if orbital_energies:
            for P, Q, occ, kappa, label in orb_list:
                eps = orbital_energies.get(label, 0.0)
                E_orbital_sum += occ * eps
        else:
            # Fallback: compute from matrix expectation values
            for P, Q, occ, kappa, label in orb_list:
                H0, S0 = build_dirac_matrices(self.ns, kappa, self.Z, bsp_data)
                vec = np.concatenate([P, Q])
                norm_sq = vec @ S0 @ vec
                if norm_sq > 1e-20:
                    E_orbital_sum += occ * (vec @ H0 @ vec) / norm_sq

        # Direct Coulomb: E_J = (1/2) Σ_{a,b} q_a * q_b * F^0(a,b)
        E_J = 0.0
        for i, (Pa, Qa, oa, ka, la) in enumerate(orb_list):
            for j, (Pb, Qb, ob, kb, lb) in enumerate(orb_list):
                f0 = compute_slater_fk(Pa, Qa, Pb, Qb, bsp_data, k=0)
                E_J += 0.5 * oa * ob * f0

        # Exchange: E_K = (1/2) Σ_{a,b,k} w_k * G^k(a,b)
        # w_k = q_a * q_b / ((2j_a+1)(2j_b+1)) * Lambda(k)
        # The exchange removes self-interaction from E_J.
        from atomx.scf.angular_coeffs import exchange_lambda
        E_K = 0.0
        if self.exchange == "hf":
            for i, (Pa, Qa, oa, ka, la) in enumerate(orb_list):
                tj_a = 2 * abs(ka) - 1
                for j, (Pb, Qb, ob, kb, lb) in enumerate(orb_list):
                    tj_b = 2 * abs(kb) - 1
                    k_values = exchange_k_range(ka, kb)
                    for k in k_values:
                        lam = exchange_lambda(ka, kb, k)
                        if lam < 1e-15:
                            continue
                        gk = compute_slater_gk(Pa, Qa, Pb, Qb, bsp_data, k)
                        w_k = oa * ob / ((tj_a + 1.0) * (tj_b + 1.0)) * lam
                        E_K += 0.5 * w_k * gk
        else:
            # Slater exchange energy: E_x = alpha * (3/4)(3/pi)^{1/3} ∫ ρ^{4/3} dr
            nv = bsp_data["nv"]
            gr = bsp_data["gr"]
            grw = bsp_data["grw"]
            n_gauss = gr.shape[1]
            rho = np.zeros((nv, n_gauss))
            for P, Q, occ, _, _ in orb_list:
                fP = self._eval_on_grid(P)
                fQ = self._eval_on_grid(Q)
                rho += occ * (fP ** 2 + fQ ** 2)
            for iv in range(nv):
                for m in range(n_gauss):
                    if rho[iv, m] > 1e-30:
                        E_K += grw[iv, m] * (self.alpha_x * 0.75
                               * (3.0 / np.pi) ** (1.0 / 3.0)
                               * rho[iv, m] ** (4.0 / 3.0))

        # E_total = Σ q ε - E_J + E_K (eigenvalue-based, KB-safe)
        # E_1e = Σ q ε - 2*(E_J - E_K) (derived)
        E_total = E_orbital_sum - E_J + E_K
        E_1e = E_orbital_sum - 2.0 * (E_J - E_K)
        components = {"E_1e": E_1e, "E_J": E_J, "E_K": E_K}
        return E_total, components

    def solve(self, max_iter=100, tol=1e-6, mixing=0.3):
        """Run the SCF iteration (DBSR_HF-style sequential orbital updates).

        Strategy (following DBSR_HF):
        - Solve orbital-by-orbital, immediately updating the density
        - Pure eigenvector replacement (no mixing on orbitals)
        - Light potential mixing only for stability in early iterations
        - Dual convergence: max orbital change AND energy change

        Parameters
        ----------
        max_iter : int
        tol : float
            Orbital convergence tolerance (relative).
        mixing : float
            Potential mixing for first few iterations only.
        """
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
        for kappa, shells in kappa_groups.items():
            H, S = build_dirac_matrices(ns, kappa, self.Z, self.bsp_data)
            result = solve_dirac(H, S, ns, kappa,
                                  nsp=nsp_init if has_kb else None,
                                  nsq=nsq_init if has_kb else None)
            orb_list = []
            for sh in shells:
                idx = sh.n - sh.l - 1
                if kappa > 0 and not has_kb:
                    idx += 1
                if idx < result.n_bound:
                    P, Q, E = result.get_orbital(idx)
                    orb_list.append((P, Q, sh.occupation))
                    orbital_energies[sh.label] = E
            orbitals[kappa] = orb_list

        converged = False
        E_old = 0.0
        residual = 1.0

        for iteration in range(1, max_iter + 1):
            max_orb_change = 0.0

            # Sweep over all orbitals sequentially (DBSR_HF style)
            has_kb = self.bsp_data.get("ksp", self.ks) != self.bsp_data.get("ksq", self.ks)
            nsp = self.bsp_data.get("nsp", ns)
            nsq = self.bsp_data.get("nsq", ns)

            for sh, kappa, orb_idx in orbital_schedule:
                # Build Fock: H_0 + V_J + V_exchange
                V_J_PP, V_J_QQ = self._build_direct_potential(orbitals)
                H, S_full = build_dirac_matrices(ns, kappa, self.Z, self.bsp_data)

                if self.exchange == "hf":
                    K_PP, K_QQ = self._build_exchange_matrix_hf(kappa, orbitals)
                    H[:ns, :ns] += V_J_PP - K_PP[:ns, :ns]
                    H[ns:, ns:] += V_J_QQ - K_QQ[:ns, :ns]
                else:
                    Vx_PP, Vx_QQ = self._build_exchange_matrix_slater(orbitals)
                    H[:ns, :ns] += V_J_PP + Vx_PP[:ns, :ns]
                    H[ns:, ns:] += V_J_QQ + Vx_QQ[:ns, :ns]

                result = solve_dirac(H, S_full, ns, kappa,
                                      nsp=nsp if has_kb else None,
                                      nsq=nsq if has_kb else None)

                # Update only the target orbital
                eig_idx = sh.n - sh.l - 1
                if kappa > 0 and not has_kb:
                    eig_idx += 1  # skip spurious only without KB
                if eig_idx < result.n_bound:
                    P_new, Q_new, E_new = result.get_orbital(eig_idx)

                    # Track orbital change
                    orb_list = orbitals.get(kappa, [])
                    if orb_idx < len(orb_list):
                        P_old = orb_list[orb_idx][0]
                        p_max = np.max(np.abs(P_old))
                        if p_max > 1e-15:
                            change = np.max(np.abs(P_new - P_old)) / p_max
                            max_orb_change = max(max_orb_change, change)

                    # Pure eigenvector replacement (DBSR_HF style: no mixing)
                    while orb_idx >= len(orb_list):
                        orb_list.append((np.zeros(ns), np.zeros(ns), sh.occupation))
                    orb_list[orb_idx] = (P_new, Q_new, sh.occupation)
                    orbitals[kappa] = orb_list
                    orbital_energies[sh.label] = E_new

            # Check energy convergence
            E_total_quick = sum(
                occ * orbital_energies.get(sh.label, 0.0)
                for sh in self.subshells
                for occ in [sh.occupation]
            )
            if iteration > 1:
                energy_change = abs(E_total_quick - E_old) / max(abs(E_old), 1e-10)
            else:
                energy_change = 1.0
            E_old = E_total_quick
            residual = max_orb_change

            if max_orb_change < tol and energy_change < tol * 0.01:
                converged = True
                break

        # Compute total energy
        total_energy, energy_components = self._compute_total_energy(orbitals, orbital_energies)

        # QED and Breit corrections (perturbative, after SCF convergence)
        orb_list = self._flatten_orbitals(orbitals)
        if self.include_breit or self.include_qed:
            from atomx.scf.breit import compute_breit_energy
            from atomx.scf.qed import compute_qed_corrections

            if self.include_breit:
                E_Breit = compute_breit_energy(orb_list, self.bsp_data, self.Z)
                energy_components["E_Breit"] = E_Breit
                total_energy += E_Breit

            if self.include_qed:
                qed = compute_qed_corrections(orb_list, self.bsp_data, self.Z)
                energy_components.update(qed)
                total_energy += qed["E_QED_total"]

        return SCFResult(
            converged=converged,
            iterations=iteration,
            residual=residual,
            total_energy=total_energy,
            orbital_energies=orbital_energies,
            subshells=self.subshells,
            energy_components=energy_components,
            orbitals_raw=dict(orbitals),  # expose converged orbitals
        )


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
