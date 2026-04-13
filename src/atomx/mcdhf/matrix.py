"""CI Hamiltonian matrix construction for MCDHF.

Slater-Condon rules:
- 0-diff (diagonal): full 1e + 2e energy of the configuration
- 1-diff (a→b):   <a|h|b> + Σ_c q_c [R^0(ac,bc) - Σ_k c^k R^k(ac,cb)]
- 2-diff (ab→cd): Σ_k [direct R^k(ac,bd) - exchange R^k(ac,db)]
- 3+ diff:        0  (orthogonality)
"""

import numpy as np
from atomx.scf.slater import (
    compute_slater_fk, compute_slater_gk, compute_slater_rk,
    eval_orbital_on_grid, compute_yk,
)
from atomx.scf.angular_coeffs import exchange_lambda, exchange_k_range
from atomx.dirac.matrix import build_dirac_matrices


def build_ci_matrix(csfs, orbitals_dict, orbital_basis, bsp_data, Z, ns):
    """Build the CI Hamiltonian matrix H_{rs} in the CSF basis."""
    n_csf = len(csfs)
    H = np.zeros((n_csf, n_csf))
    for r in range(n_csf):
        for s in range(r, n_csf):
            h_rs = _ci_matrix_element(csfs[r], csfs[s], orbitals_dict,
                                       orbital_basis, bsp_data, Z, ns)
            H[r, s] = h_rs
            if r != s:
                H[s, r] = h_rs
    return H


def _ci_matrix_element(csf_r, csf_s, orbitals_dict, orbital_basis,
                        bsp_data, Z, ns):
    n_diff = csf_r.n_electron_diff(csf_s)
    if n_diff == 0:
        return _diagonal_element(csf_r, orbitals_dict, bsp_data, Z, ns)
    elif n_diff == 1:
        return _one_diff_element(csf_r, csf_s, orbitals_dict, bsp_data, Z, ns)
    elif n_diff == 2:
        return _two_diff_element(csf_r, csf_s, orbitals_dict, bsp_data)
    return 0.0


def _get_orb(key, orbitals_dict, ns):
    """Get orbital (P, Q) or zeros if not found."""
    if key in orbitals_dict:
        return orbitals_dict[key]
    return np.zeros(ns), np.zeros(ns)


def _diagonal_element(csf, orbitals_dict, bsp_data, Z, ns):
    """Diagonal: E = Σ_a q_a <a|h|a> + (1/2) Σ_{a,b} [q_a q_b F^0 - exchange]."""
    E_1e = 0.0
    for sh in csf.subshells:
        P, Q = _get_orb((sh.n, sh.kappa), orbitals_dict, ns)
        H0, S0 = build_dirac_matrices(ns, sh.kappa, Z, bsp_data)
        vec = np.concatenate([P, Q])
        norm_sq = vec @ S0 @ vec
        if norm_sq > 1e-20:
            E_1e += sh.occupation * (vec @ H0 @ vec) / norm_sq

    E_J, E_K = 0.0, 0.0
    for sh_a in csf.subshells:
        Pa, Qa = _get_orb((sh_a.n, sh_a.kappa), orbitals_dict, ns)
        tj_a = 2 * abs(sh_a.kappa) - 1
        for sh_b in csf.subshells:
            Pb, Qb = _get_orb((sh_b.n, sh_b.kappa), orbitals_dict, ns)
            tj_b = 2 * abs(sh_b.kappa) - 1

            f0 = compute_slater_fk(Pa, Qa, Pb, Qb, bsp_data, k=0)
            E_J += 0.5 * sh_a.occupation * sh_b.occupation * f0

            for k in exchange_k_range(sh_a.kappa, sh_b.kappa):
                lam = exchange_lambda(sh_a.kappa, sh_b.kappa, k)
                if lam < 1e-15:
                    continue
                gk = compute_slater_gk(Pa, Qa, Pb, Qb, bsp_data, k)
                w_k = (sh_a.occupation * sh_b.occupation
                       / ((tj_a + 1.0) * (tj_b + 1.0)) * lam)
                E_K += 0.5 * w_k * gk

    return E_1e + E_J - E_K


def _one_diff_element(csf_r, csf_s, orbitals_dict, bsp_data, Z, ns):
    """1-electron excitation a→b.

    H_rs = sqrt(q) * [<a|h|b> + Σ_c q_c (R^0(ac,bc) - Σ_k c_k R^k(ac,cb))]

    where c runs over common occupied orbitals, and q is the occupation factor.
    """
    diffs = csf_r.differs_from(csf_s)
    lost = [(key, qs, qo) for key, qs, qo in diffs if qs > qo]
    gained = [(key, qs, qo) for key, qs, qo in diffs if qs < qo]

    if len(lost) != 1 or len(gained) != 1:
        return 0.0

    key_a, qa_r, qa_s = lost[0]
    key_b, qb_r, qb_s = gained[0]
    Pa, Qa = _get_orb(key_a, orbitals_dict, ns)
    Pb, Qb = _get_orb(key_b, orbitals_dict, ns)
    kappa_a, kappa_b = key_a[1], key_b[1]

    # Occupation coupling factor
    sqrt_factor = np.sqrt(abs(qa_r - qa_s) * abs(qb_s - qb_r))

    # One-electron: <a|h|b> (nonzero only for same kappa)
    H_1e = 0.0
    if kappa_a == kappa_b:
        H0, S0 = build_dirac_matrices(ns, kappa_a, Z, bsp_data)
        vec_a = np.concatenate([Pa, Qa])
        vec_b = np.concatenate([Pb, Qb])
        H_1e = vec_a @ H0 @ vec_b

    # Two-electron: Σ_c q_c [direct - exchange] with common orbitals
    H_2e = 0.0
    # Common orbitals: those present in BOTH configurations
    occ_r = {(sh.n, sh.kappa): sh.occupation for sh in csf_r.subshells}
    occ_s = {(sh.n, sh.kappa): sh.occupation for sh in csf_s.subshells}

    for sh_c in csf_r.subshells:
        key_c = (sh_c.n, sh_c.kappa)
        # Use the minimum occupation (common electrons)
        q_common = min(occ_r.get(key_c, 0), occ_s.get(key_c, 0))
        if q_common <= 0:
            continue

        Pc, Qc = _get_orb(key_c, orbitals_dict, ns)
        kappa_c = key_c[1]
        tj_c = 2 * abs(kappa_c) - 1

        # Direct: R^0(ac, bc) — Coulomb between "a→b excitation" and orbital c
        rk_direct = compute_slater_rk(Pa, Qa, Pc, Qc, Pb, Qb, Pc, Qc,
                                       bsp_data, k=0)
        H_2e += q_common * rk_direct

        # Exchange: -Σ_k c_k R^k(ac, cb)
        for k in exchange_k_range(kappa_a, kappa_c):
            lam_ac = exchange_lambda(kappa_a, kappa_c, k)
            if lam_ac < 1e-15:
                continue
            # R^k(ac, cb) — exchange integral
            rk_exch = compute_slater_rk(Pa, Qa, Pc, Qc, Pc, Qc, Pb, Qb,
                                         bsp_data, k)
            c_k = q_common / (tj_c + 1.0) * lam_ac
            H_2e -= c_k * rk_exch

    return sqrt_factor * (H_1e + H_2e)


def _two_diff_element(csf_r, csf_s, orbitals_dict, bsp_data):
    """2-electron excitation (a,b→c,d).

    H_rs = Σ_k [direct R^k(ac,bd) - exchange R^k(ac,db)]
    weighted by angular coupling coefficients.
    """
    diffs = csf_r.differs_from(csf_s)
    lost = [(key, qs, qo) for key, qs, qo in diffs if qs > qo]
    gained = [(key, qs, qo) for key, qs, qo in diffs if qs < qo]

    if len(lost) != 2 or len(gained) != 2:
        return 0.0

    ns = len(list(orbitals_dict.values())[0][0]) if orbitals_dict else 50

    key_a, _, _ = lost[0]
    key_b, _, _ = lost[1]
    key_c, _, _ = gained[0]
    key_d, _, _ = gained[1]

    Pa, Qa = _get_orb(key_a, orbitals_dict, ns)
    Pb, Qb = _get_orb(key_b, orbitals_dict, ns)
    Pc, Qc = _get_orb(key_c, orbitals_dict, ns)
    Pd, Qd = _get_orb(key_d, orbitals_dict, ns)

    kappa_a, kappa_b = key_a[1], key_b[1]
    kappa_c, kappa_d = key_c[1], key_d[1]

    H_2e = 0.0

    # Direct contribution: R^0(ac, bd) (monopole)
    rk_direct = compute_slater_rk(Pa, Qa, Pc, Qc, Pb, Qb, Pd, Qd,
                                   bsp_data, k=0)
    H_2e += rk_direct

    # Exchange contribution: -Σ_k c_k R^k(ac, db)
    for k in exchange_k_range(kappa_a, kappa_d):
        lam = exchange_lambda(kappa_a, kappa_d, k)
        if lam < 1e-15:
            continue
        rk_exch = compute_slater_rk(Pa, Qa, Pc, Qc, Pd, Qd, Pb, Qb,
                                     bsp_data, k)
        tj_a = 2 * abs(kappa_a) - 1
        tj_d = 2 * abs(kappa_d) - 1
        c_k = lam / ((tj_a + 1.0) * (tj_d + 1.0))
        H_2e -= c_k * rk_exch

    return H_2e
