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
from atomx.bspline.convol import (
    compute_slater_fk_cell, compute_slater_gk_cell,
    ConvolCache,
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
                 include_breit=False, include_qed=False,
                 nuclear="point",
                 starting_potential=None,
                 jj_configs=None):
        """
        Parameters
        ----------
        exchange : str
            "hf" for non-local Hartree-Fock exchange,
            "slater" for local Slater exchange,
            "tfda" for Thomas-Fermi-Dirac-Amaldi (no SCF iteration),
            "gsz" for Green-Sellin-Zachor parametric potential.
        nuclear : str
            Nuclear model: "point", "fermi", "uniform".
        starting_potential : str or None
            Starting potential for SCF: None (hydrogenic), "tfda", "gsz".
        jj_configs : list of JJConfig, optional
            Multi-configuration jj-coupled configurations with weights.
            Used for WW-based energy coefficients (DBSR_HF av_energy_coef).
            If None, uses simple q*(q-1)/2 and q_a*q_b formulas.
        """
        self.Z = Z
        self.subshells = subshells
        self.jj_configs = jj_configs
        self.ns = ns
        self.ks = ks
        self.r_min = r_min if r_min is not None else 0.05 / Z
        self.r_max = r_max if r_max is not None else max(40.0, 60.0 / Z)
        self.exchange = exchange
        self.alpha_x = alpha_x
        self.include_breit = include_breit
        self.include_qed = include_qed
        self.nuclear = nuclear
        self.starting_potential = starting_potential
        self.N_electrons = sum(sh.occupation for sh in subshells)

        # Nuclear potential function (None = point nucleus -Z/r)
        from atomx.nuclear import make_nuclear_potential
        self._nuclear_potential = make_nuclear_potential(Z, nuclear)

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
            from atomx.scf.jj_config import compute_WW
            return compute_WW(i, j, self.jj_configs)
        # Single-config with fractional occupations: q*(q-1) is correct
        qi = self.subshells[i].occupation
        if i == j:
            return qi * (qi - 1.0)
        qj = self.subshells[j].occupation
        return qi * qj

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

    def _build_direct_potential(self, orbitals, target_kappa=None, target_idx=None):
        """Build Hartree potential via cell-based convolution (DBSR_HF algorithm).

        Uses rkb convolution for each orbital's F^0 contribution to the direct
        potential matrix. Self-consistent with cell-based energy computation.

        Falls back to Y^k for DFS (Slater local exchange) mode, which doesn't
        benefit from cell algorithm precision.
        """
        # DFS mode: use fast Y^k approach (no cell-algorithm precision needed)
        if self.exchange != "hf":
            return self._build_direct_potential_yk(orbitals, target_kappa, target_idx)

        from atomx.bspline.convol import (
            _build_rkb_sym, _density_sym, _convol_case1, _convol_case1_mixed,
        )

        bd = self.bsp_data
        ksp = bd.get("ksp", bd["ks"])
        ksq = bd.get("ksq", bd["ks"])
        ns_full = bd["ns"]

        if not hasattr(self, '_convol_cache'):
            self._convol_cache = ConvolCache(bd)

        # All 4 integral types for direct potential (DBSR_HF convention)
        rkb_pp, _, _ = self._convol_cache.get_rkb(0, "pppp")
        rkb_qq, _, _ = self._convol_cache.get_rkb(0, "qqqq")
        rkb_pq, _, _ = self._convol_cache.get_rkb(0, "pqpq")
        rkb_qp, _, _ = self._convol_cache.get_rkb(0, "qpqp")

        V_J_PP = np.zeros((ns_full, ns_full))
        V_J_QQ = np.zeros((ns_full, ns_full))

        # Build flat index map
        flat_idx = {}
        idx = 0
        kappa_counter = {}
        for sh in self.subshells:
            orb_idx_local = kappa_counter.get(sh.kappa, 0)
            kappa_counter[sh.kappa] = orb_idx_local + 1
            flat_idx[(sh.kappa, orb_idx_local)] = idx
            idx += 1

        target_flat = flat_idx.get((target_kappa, target_idx), -1) if target_kappa is not None else -1
        target_q = self.subshells[target_flat].occupation if target_flat >= 0 else 1.0

        for kappa, orb_list in orbitals.items():
            for orb_i, (P_b, Q_b, occ_b) in enumerate(orb_list):
                j_flat = flat_idx.get((kappa, orb_i), -1)
                if target_flat >= 0 and j_flat >= 0:
                    ww = self._get_WW(target_flat, j_flat)
                    coeff = ww / target_q
                else:
                    coeff = occ_b
                    if target_kappa == kappa and target_idx == orb_i:
                        coeff = occ_b - 1.0
                if abs(coeff) < 1e-10:
                    continue

                # DBSR_HF direct potential has 4 component contributions:
                # PP block gets: pppp(P-density) + pqpq(Q-density)
                # QQ block gets: qqqq(Q-density) + qpqp(P-density)

                # pppp → PP: contract with P-density
                d_P = _density_sym(P_b[:ns_full], P_b[:ns_full], ns_full, ksp)
                v_pp = _convol_case1(rkb_pp, d_P, ns_full, ksp)
                for i in range(ns_full):
                    for s in range(ksp):
                        if i + s < ns_full:
                            V_J_PP[i, i + s] += coeff * v_pp[i, s]
                            if s > 0:
                                V_J_PP[i + s, i] += coeff * v_pp[i, s]

                # pqpq → PP: rkb(ns,ns,ksp,ksq), contract Q-density(ksq) → result(ksp)
                d_Q = _density_sym(Q_b[:ns_full], Q_b[:ns_full], ns_full, ksq)
                v_pq = _convol_case1_mixed(rkb_pq, d_Q, ns_full, ksp, ksq)
                for i in range(ns_full):
                    for s in range(ksp):
                        if i + s < ns_full:
                            V_J_PP[i, i + s] += coeff * v_pq[i, s]
                            if s > 0:
                                V_J_PP[i + s, i] += coeff * v_pq[i, s]

                # qpqp → QQ: rkb(ns,ns,ksq,ksp), contract P-density(ksp) → result(ksq)
                v_qp = _convol_case1_mixed(rkb_qp, d_P, ns_full, ksq, ksp)
                for i in range(ns_full):
                    for s in range(ksq):
                        if i + s < ns_full:
                            V_J_QQ[i, i + s] += coeff * v_qp[i, s]
                            if s > 0:
                                V_J_QQ[i + s, i] += coeff * v_qp[i, s]

                # qqqq → QQ: contract with Q-density
                v_qq = _convol_case1(rkb_qq, d_Q, ns_full, ksq)
                for i in range(ns_full):
                    for s in range(ksq):
                        if i + s < ns_full:
                            V_J_QQ[i, i + s] += coeff * v_qq[i, s]
                            if s > 0:
                                V_J_QQ[i + s, i] += coeff * v_qq[i, s]

        return V_J_PP, V_J_QQ

    def _build_direct_potential_yk(self, orbitals, target_kappa=None, target_idx=None):
        """Y^k-based Hartree potential (fast, for DFS mode)."""
        nv = self.bsp_data["nv"]
        ns = self.ns
        gr = self.bsp_data["gr"]
        grw = self.bsp_data["grw"]
        n_gauss = gr.shape[1]

        rho_total = np.zeros((nv, n_gauss))
        for kappa, orb_list in orbitals.items():
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

        Returns (K_PP, K_QQ, K_PQ, K_QP) for all 4 component blocks:
        - pppp -> PP diagonal block
        - qqqq -> QQ diagonal block
        - pqpq -> PQ off-diagonal block (cross-component exchange)
        - qpqp -> QP off-diagonal block (cross-component exchange)
        """
        from atomx.scf.angular_coeffs import exchange_lambda
        from atomx.bspline.convol import _density_full, _convol_case4

        ns = self.ns
        bd = self.bsp_data
        ksp = bd.get("ksp", self.ks)
        nsp = bd.get("nsp", ns)
        ksq = bd.get("ksq", self.ks)
        nsq = bd.get("nsq", ns)
        ns_full = bd.get("ns", ns)

        if not hasattr(self, '_convol_cache'):
            self._convol_cache = ConvolCache(bd)

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
                k_values = exchange_k_range(target_kappa, kappa_b)
                for k in k_values:
                    lam = exchange_lambda(target_kappa, kappa_b, k)
                    if lam < 1e-15:
                        continue
                    c_k = occ_b * lam / ((tj_a + 1.0) * (tj_b + 1.0))

                    # pppp -> PP block
                    rkb_pp, _, _ = self._convol_cache.get_rkb(k, "pppp")
                    d_P = _density_full(P_b[:nsp], P_b[:nsp], nsp)
                    Kk_PP = _convol_case4(rkb_pp, d_P, nsp, ksp)
                    n = min(Kk_PP.shape[0], ns_full)
                    K_PP[:n, :n] += c_k * Kk_PP[:n, :n]

                    # qqqq -> QQ block
                    rkb_qq, _, _ = self._convol_cache.get_rkb(k, "qqqq")
                    d_Q = _density_full(Q_b[:nsq], Q_b[:nsq], nsq)
                    Kk_QQ = _convol_case4(rkb_qq, d_Q, nsq, ksq)
                    n = min(Kk_QQ.shape[0], ns_full)
                    K_QQ[:n, :n] += c_k * Kk_QQ[:n, :n]

                    # pqpq -> PQ block: density = P_b * Q_b (cross-component)
                    rkb_pq, _, _ = self._convol_cache.get_rkb(k, "pqpq")
                    d_PQ = _density_full(P_b[:nsp], Q_b[:nsq], max(nsp, nsq))
                    Kk_PQ = _convol_case4(rkb_pq, d_PQ, ns_full, ksp)
                    K_PQ[:ns_full, :ns_full] += c_k * Kk_PQ[:ns_full, :ns_full]

                    # qpqp -> QP block: density = Q_b * P_b
                    rkb_qp, _, _ = self._convol_cache.get_rkb(k, "qpqp")
                    d_QP = _density_full(Q_b[:nsq], P_b[:nsp], max(nsp, nsq))
                    Kk_QP = _convol_case4(rkb_qp, d_QP, ns_full, ksq)
                    K_QP[:ns_full, :ns_full] += c_k * Kk_QP[:ns_full, :ns_full]

        K_PP = 0.5 * (K_PP + K_PP.T)
        K_QQ = 0.5 * (K_QQ + K_QQ.T)
        # PQ and QP: symmetrize each individually
        K_PQ = 0.5 * (K_PQ + K_PQ.T)
        K_QP = 0.5 * (K_QP + K_QP.T)
        return K_PP, K_QQ, K_PQ, K_QP

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

    def _build_intra_shell_potential(self, target_kappa, target_idx, orbitals):
        """Same-shell F^k contribution for k>0 via cell-based convolution.

        For j=1/2 shells (s, p1/2), only k=0 is allowed -> returns zeros.
        For j>=1 (p3/2, d, f, ...), adds the intra-shell potential:
            coeff = -(q-1) * Lambda(k) / (2j * (2j+1))

        Uses cell-based convolution (rkb + icase=1) for consistency with
        the direct potential and exchange, avoiding Y^k/cell mismatch.
        """
        from atomx.scf.angular_coeffs import exchange_lambda
        from atomx.bspline.convol import _density_sym, _convol_case1

        ns = self.ns
        tj_a = 2 * abs(target_kappa) - 1

        Vi_PP = np.zeros((ns, ns))
        Vi_QQ = np.zeros((ns, ns))

        if tj_a <= 1:
            return Vi_PP, Vi_QQ

        orb_list_t = orbitals.get(target_kappa, [])
        if target_idx >= len(orb_list_t):
            return Vi_PP, Vi_QQ

        P_a, Q_a, q_a = orb_list_t[target_idx]
        if q_a * (q_a - 1.0) < 1e-15:
            return Vi_PP, Vi_QQ  # no intra-shell pairs for q<=1

        bd = self.bsp_data
        ksp = bd.get("ksp", self.ks)
        nsp = bd.get("nsp", ns)
        ksq = bd.get("ksq", self.ks)
        nsq = bd.get("nsq", ns)

        l_a = target_kappa if target_kappa > 0 else -target_kappa - 1

        if not hasattr(self, '_convol_cache'):
            self._convol_cache = ConvolCache(bd)

        for k in range(2, tj_a + 1):
            if (k + 2 * l_a) % 2 != 0:
                continue
            lam = exchange_lambda(target_kappa, target_kappa, k)
            if lam < 1e-15:
                continue
            coeff = -(q_a - 1.0) * lam / (tj_a * (tj_a + 1.0))

            # Cell-based F^k contribution (icase=1)
            rkb_pp, _, _ = self._convol_cache.get_rkb(k, "pppp")
            d_P = _density_sym(P_a, P_a, nsp, ksp)
            v_banded = _convol_case1(rkb_pp, d_P, nsp, ksp)
            for i in range(nsp):
                for s in range(ksp):
                    if i + s < nsp:
                        Vi_PP[i, i + s] += coeff * v_banded[i, s]
                        if s > 0:
                            Vi_PP[i + s, i] += coeff * v_banded[i, s]

            rkb_qq, _, _ = self._convol_cache.get_rkb(k, "qqqq")
            d_Q = _density_sym(Q_a, Q_a, nsq, ksq)
            v_banded_q = _convol_case1(rkb_qq, d_Q, nsq, ksq)
            for i in range(nsq):
                for s in range(ksq):
                    if i + s < nsq:
                        Vi_QQ[i, i + s] += coeff * v_banded_q[i, s]
                        if s > 0:
                            Vi_QQ[i + s, i] += coeff * v_banded_q[i, s]

        return Vi_PP, Vi_QQ

    def _compute_total_energy(self, orbitals, orbital_energies=None):
        """Compute total energy using direct Slater integral formula (DBSR_HF style).

        E = Σ_a q_a <a|h|a> + E_2e

        where E_2e uses DBSR_HF av_energy_coef formulas:
          same-shell:  q(q-1)/2 * [F^0(a,a) - Σ_{k>0} Lambda(k)/(2j*(2j+1)) * F^k(a,a)]
          cross-shell: q_a*q_b * [F^0(a,b) - Σ_k Lambda(k)/((2j_a+1)(2j_b+1)) * G^k(a,b)]
        """
        from atomx.scf.angular_coeffs import exchange_lambda

        orb_list = self._flatten_orbitals(orbitals)
        bsp_data = self.bsp_data

        if self.exchange != "hf":
            # DFS (Slater local exchange) — use eigenvalue formula
            return self._compute_total_energy_slater(orbitals, orbital_energies,
                                                      orb_list, bsp_data)

        # One-electron: Σ q_a <a|h|a>
        ns_e = bsp_data.get("ns", self.ns)  # max(nsp, nsq)
        E_1e = 0.0
        for P, Q, occ, kappa, label in orb_list:
            H0, S0 = build_dirac_matrices(ns_e, kappa, self.Z, bsp_data, self._nuclear_potential)
            vec = np.concatenate([P[:ns_e], Q[:ns_e]])
            norm_sq = vec @ S0 @ vec
            if norm_sq > 1e-20:
                E_1e += occ * (vec @ H0 @ vec) / norm_sq

        # Two-electron (DBSR_HF av_energy_coef formula)
        # Track direct (E_J = F^0) and exchange (E_K = F^{k>0}, G^k) separately
        # Use cell-based integrals for consistency with DBSR_HF
        convol_cache = ConvolCache(bsp_data)
        _fk = lambda Pa, Qa, Pb, Qb, k: compute_slater_fk_cell(
            Pa, Qa, Pb, Qb, bsp_data, k, convol_cache)
        _gk = lambda Pa, Qa, Pb, Qb, k: compute_slater_gk_cell(
            Pa, Qa, Pb, Qb, bsp_data, k, convol_cache)

        # Two-electron coefficients use WW function for multi-config average
        E_J = 0.0   # direct Coulomb (F^0 terms)
        E_K = 0.0   # exchange (F^{k>0} and G^k terms)
        for i, (Pa, Qa, oa, ka, la) in enumerate(orb_list):
            tj_a = 2 * abs(ka) - 1
            l_a = ka if ka > 0 else -ka - 1
            for j, (Pb, Qb, ob, kb, lb) in enumerate(orb_list):
                if j > i:
                    continue
                tj_b = 2 * abs(kb) - 1
                l_b = kb if kb > 0 else -kb - 1

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

    def _compute_total_energy_slater(self, orbitals, orbital_energies,
                                      orb_list, bsp_data):
        """DFS total energy using eigenvalue formula (Slater local exchange)."""
        E_orbital_sum = 0.0
        if orbital_energies:
            for P, Q, occ, kappa, label in orb_list:
                eps = orbital_energies.get(label, 0.0)
                E_orbital_sum += occ * eps
        else:
            for P, Q, occ, kappa, label in orb_list:
                H0, S0 = build_dirac_matrices(self.ns, kappa, self.Z, bsp_data, self._nuclear_potential)
                vec = np.concatenate([P, Q])
                norm_sq = vec @ S0 @ vec
                if norm_sq > 1e-20:
                    E_orbital_sum += occ * (vec @ H0 @ vec) / norm_sq

        E_J = 0.0
        for i, (Pa, Qa, oa, ka, la) in enumerate(orb_list):
            for j, (Pb, Qb, ob, kb, lb) in enumerate(orb_list):
                f0 = compute_slater_fk(Pa, Qa, Pb, Qb, bsp_data, k=0)
                E_J += 0.5 * oa * ob * f0

        nv = bsp_data["nv"]
        gr = bsp_data["gr"]
        grw = bsp_data["grw"]
        n_gauss = gr.shape[1]
        rho = np.zeros((nv, n_gauss))
        for P, Q, occ, _, _ in orb_list:
            fP = self._eval_on_grid(P)
            fQ = self._eval_on_grid(Q)
            rho += occ * (fP ** 2 + fQ ** 2)
        E_K = 0.0
        for iv in range(nv):
            for m in range(n_gauss):
                if rho[iv, m] > 1e-30:
                    E_K += grw[iv, m] * (self.alpha_x * 0.75
                           * (3.0 / np.pi) ** (1.0 / 3.0)
                           * rho[iv, m] ** (4.0 / 3.0))

        E_total = E_orbital_sum - E_J - (1.0/3.0) * E_K
        E_1e = E_orbital_sum - 2.0 * E_J - (4.0/3.0) * E_K
        return E_total, {"E_1e": E_1e, "E_J": E_J, "E_K": E_K}

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
                    orb_list.append((P, Q, sh.occupation))
                    orbital_energies[sh.label] = E
            orbitals[kappa] = orb_list

        # For parametric potentials (TFDA, GSZ), no SCF iteration needed --
        # the initial orbitals ARE the final orbitals
        if self.exchange in ("tfda", "gsz"):
            total_energy, energy_components = self._compute_total_energy(
                orbitals, orbital_energies)
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
                converged=True, iterations=1, residual=0.0,
                total_energy=total_energy, orbital_energies=orbital_energies,
                subshells=self.subshells, energy_components=energy_components,
                orbitals_raw=dict(orbitals))

        converged = False
        E_old = 0.0
        residual = 1.0
        has_kb = self.bsp_data.get("ksp", self.ks) != self.bsp_data.get("ksq", self.ks)
        nsp = self.bsp_data.get("nsp", ns)
        nsq = self.bsp_data.get("nsq", ns)

        # OPT 1: Cache one-electron matrices per kappa (they never change)
        H0_cache = {}
        S0_cache = {}
        for kappa in kappa_groups:
            H0, S0 = build_dirac_matrices(ns_full_init, kappa, self.Z, self.bsp_data, self._nuclear_potential)
            H0_cache[kappa] = H0
            S0_cache[kappa] = S0

        for iteration in range(1, max_iter + 1):
            max_orb_change = 0.0

            # OPT 2: Exchange cache — K^k(b) depends only on source orbital
            # and k, NOT on target kappa. Shared across the sweep.
            # Invalidated when the source orbital changes.
            exchange_cache = {}

            ns_full = max(nsp, nsq)  # matrix block dimension

            for sh, kappa, orb_idx in orbital_schedule:
                H = H0_cache[kappa].copy()  # OPT 1: cached one-electron
                S_full = S0_cache[kappa]

                if self.exchange == "hf":
                    # Direct potential (pppp→PP, qqqq→QQ diagonal blocks)
                    V_J_PP, V_J_QQ = self._build_direct_potential(
                        orbitals, target_kappa=kappa, target_idx=orb_idx)
                    # Exchange (pppp→PP, qqqq→QQ + pqpq→PQ, qpqp→QP off-diag)
                    K_PP, K_QQ, K_PQ, K_QP = self._build_exchange_matrix_hf(
                        kappa, orb_idx, orbitals,
                        exchange_cache=exchange_cache)
                    # Intra-shell F^k for k>0
                    Vi_PP, Vi_QQ = self._build_intra_shell_potential(
                        kappa, orb_idx, orbitals)
                    # PP block
                    n_pp = min(V_J_PP.shape[0], ns_full)
                    H[:n_pp, :n_pp] += V_J_PP[:n_pp, :n_pp]
                    n_kpp = min(K_PP.shape[0], ns_full)
                    H[:n_kpp, :n_kpp] -= K_PP[:n_kpp, :n_kpp]
                    n_vpp = min(Vi_PP.shape[0], ns_full)
                    H[:n_vpp, :n_vpp] += Vi_PP[:n_vpp, :n_vpp]
                    # QQ block
                    n_qq = min(V_J_QQ.shape[0], ns_full)
                    H[ns_full:ns_full+n_qq, ns_full:ns_full+n_qq] += V_J_QQ[:n_qq, :n_qq]
                    n_kqq = min(K_QQ.shape[0], ns_full)
                    H[ns_full:ns_full+n_kqq, ns_full:ns_full+n_kqq] -= K_QQ[:n_kqq, :n_kqq]
                    n_vqq = min(Vi_QQ.shape[0], ns_full)
                    H[ns_full:ns_full+n_vqq, ns_full:ns_full+n_vqq] += Vi_QQ[:n_vqq, :n_vqq]
                    # PQ and QP off-diagonal blocks (cross-component exchange)
                    n_pq = min(K_PQ.shape[0], ns_full)
                    H[:n_pq, ns_full:ns_full+n_pq] -= K_PQ[:n_pq, :n_pq]
                    H[ns_full:ns_full+n_pq, :n_pq] -= K_QP[:n_pq, :n_pq]
                else:
                    V_J_PP, V_J_QQ = self._build_direct_potential(orbitals)
                    Vx_PP, Vx_QQ = self._build_exchange_matrix_slater(orbitals)
                    n_pp = min(V_J_PP.shape[0], ns_full)
                    H[:n_pp, :n_pp] += V_J_PP[:n_pp, :n_pp] + Vx_PP[:n_pp, :n_pp]
                    n_qq = min(V_J_QQ.shape[0], ns_full)
                    H[ns_full:ns_full+n_qq, ns_full:ns_full+n_qq] += V_J_QQ[:n_qq, :n_qq] + Vx_QQ[:n_qq, :n_qq]

                result = solve_dirac(H, S_full, ns_full, kappa,
                                      nsp=nsp if has_kb else None,
                                      nsq=nsq if has_kb else None)

                # Select target orbital by index
                eig_idx = sh.n - sh.l - 1
                if kappa > 0 and not has_kb:
                    eig_idx += 1  # skip spurious only without KB

                # Get orbital from FULL physical spectrum (bound + discretized continuum)
                # This handles outer orbitals that become weakly bound/unbound during SCF
                n_phys = len(result.eigenvalues)
                actual_idx = min(eig_idx, n_phys - 1)
                if actual_idx >= 0 and actual_idx < n_phys:
                    # Use eigenvalues/coefficients from the physical spectrum
                    sorted_idx = np.argsort(result.eigenvalues)
                    phys_idx = sorted_idx[actual_idx]
                    vec = result.coefficients[:, phys_idx]
                    P_new = np.zeros(ns_full)
                    Q_new = np.zeros(ns_full)
                    nsp_loc = result.nsp if hasattr(result, 'nsp') else ns_full
                    nsq_loc = result.nsq if hasattr(result, 'nsq') else ns_full
                    # P occupies vec[0:nsp_loc], Q occupies vec[ns_full:ns_full+nsq_loc]
                    # (NOT vec[nsp_loc:] — that would include the zero-padded P entry)
                    P_new[:nsp_loc] = vec[:nsp_loc]
                    Q_new[:nsq_loc] = vec[ns_full:ns_full + nsq_loc]
                    E_new = result.eigenvalues[phys_idx]

                    # NaN guard: skip if eigenvalue solver produced garbage
                    if np.isnan(E_new) or np.any(np.isnan(vec)):
                        continue

                    # Track orbital change
                    orb_list = orbitals.get(kappa, [])
                    if orb_idx < len(orb_list):
                        P_old = orb_list[orb_idx][0]
                        p_max = np.max(np.abs(P_old))
                        if p_max > 1e-15:
                            change = np.max(np.abs(P_new[:len(P_old)] - P_old[:len(P_new)])) / p_max
                            max_orb_change = max(max_orb_change, change)

                    # Pure eigenvector replacement (DBSR_HF style: no mixing)
                    while orb_idx >= len(orb_list):
                        orb_list.append((np.zeros(ns_full), np.zeros(ns_full), sh.occupation))
                    orb_list[orb_idx] = (P_new, Q_new, sh.occupation)
                    orbitals[kappa] = orb_list
                    orbital_energies[sh.label] = E_new

                    # OPT 2: Invalidate exchange cache for this source orbital
                    keys_to_remove = [ck for ck in exchange_cache
                                      if ck[0] == kappa and ck[1] == orb_idx]
                    for ck in keys_to_remove:
                        del exchange_cache[ck]

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
