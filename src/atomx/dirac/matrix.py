"""Dirac Hamiltonian matrix assembly in B-spline basis.

Supports Restricted Kinetic Balance (RKB): P uses order ksp=ks-1, Q uses ksq=ks.
With RKB, four separate coupling matrices are needed (not transposes of each other):
  fpqbsd = <P_i | dQ_j/dr>   (P-bra, Q'-ket)
  fqpbsd = <Q_i | dP_j/dr>   (Q-bra, P'-ket)
  fpqbs  = <P_i | Q_j / r>   (P-bra, Q-ket)
  fqpbs  = <Q_i | P_j / r>   (Q-bra, P-ket)

Block assembly (DBSR_HF convention):
  H(P,Q) = -c * fpqbsd + κ*c * fpqbs
  H(Q,P) = +c * fqpbsd + κ*c * fqpbs
"""

import numpy as np
from atomx.constants import C, C_SQ
from atomx.bspline.integrals import (
    overlap_matrix,
    nuclear_matrix,
    coupling_matrix,
    coupling_deriv_matrix,
)


def _build_knot_sequence(t_master, ks_master, ns_master, k_target):
    """Build knot sequence for order k_target from master knot sequence.

    Same interior knots, different boundary multiplicity.
    """
    nv = ns_master - ks_master + 1
    n_target = nv + k_target - 1
    interior = t_master[ks_master:ns_master]  # interior knots

    t_new = np.zeros(n_target + k_target)
    t_new[:k_target] = t_master[0]         # boundary: k_target copies
    t_new[k_target:k_target + len(interior)] = interior
    t_new[n_target:] = t_master[ns_master]  # boundary: k_target copies of r_max
    return t_new, n_target


def build_dirac_matrices(ns, kappa, Z, bspline_data, nuclear_potential_func=None):
    """Assemble the Dirac Hamiltonian and overlap matrices.

    Handles both same-order (ksp=ksq) and RKB (ksp=ksq-1) cases.
    """
    nv = bspline_data["nv"]
    ks = bspline_data["ks"]
    gr = bspline_data["gr"]
    grw = bspline_data["grw"]

    # P and Q B-spline data
    bsp_p = bspline_data["bsp_p"]      # (nv, ng, ksp) values
    bsp_q = bspline_data["bsp_q"]      # (nv, ng, ksq) values
    bsp_q_d = bspline_data["bsp_q_d"]  # (nv, ng, ksq) Q derivatives
    bsp_p_d = bspline_data.get("bsp_p_d", None)  # (nv, ng, ksp) P derivatives (RKB)

    ksp = bspline_data.get("ksp", ks)
    ksq = bspline_data.get("ksq", ks)
    nsp = bspline_data.get("nsp", ns)
    nsq = bspline_data.get("nsq", ns)

    # Use ns_full = max(nsp, nsq) for matrix allocation (DBSR_HF convention)
    ns_full = max(nsp, nsq)
    ms = 2 * ns_full

    # --- Diagonal blocks ---
    S_PP = overlap_matrix(nv, ksp, ns_full, bsp_p, grw)
    S_QQ = overlap_matrix(nv, ksq, ns_full, bsp_q, grw)
    V_PP = nuclear_matrix(nv, ksp, ns_full, bsp_p, gr, grw, Z, nuclear_potential_func)
    V_QQ = nuclear_matrix(nv, ksq, ns_full, bsp_q, gr, grw, Z, nuclear_potential_func)

    # --- Coupling blocks: four separate matrices ---
    grw_over_r = grw / gr

    # P-Q block: <P_i | dQ_j/dr> and <P_i | Q_j/r>
    fpqbsd = coupling_deriv_matrix(nv, ksp, ksq, ns_full, bsp_p, bsp_q_d, grw)
    fpqbs = coupling_matrix(nv, ksp, ksq, ns_full, bsp_p, bsp_q, grw_over_r)

    # Q-P block: <Q_i | dP_j/dr> and <Q_i | P_j/r>
    if bsp_p_d is not None and ksp != ksq:
        # RKB: separate Q→P coupling (NOT transpose of P→Q)
        fqpbsd = coupling_deriv_matrix(nv, ksq, ksp, ns_full, bsp_q, bsp_p_d, grw)
        fqpbs = coupling_matrix(nv, ksq, ksp, ns_full, bsp_q, bsp_p, grw_over_r)
    else:
        # Same order: use same matrices (DBSR_HF: both blocks use fpqbsd/fpqbs)
        fqpbsd = fpqbsd
        fqpbs = fpqbs

    # --- Assemble ---
    H = np.zeros((ms, ms))
    S = np.zeros((ms, ms))

    H[:ns_full, :ns_full] = V_PP
    H[ns_full:, ns_full:] = V_QQ - 2.0 * C_SQ * S_QQ
    H[:ns_full, ns_full:] = -C * fpqbsd + kappa * C * fpqbs    # P-Q block
    H[ns_full:, :ns_full] = C * fqpbsd + kappa * C * fqpbs      # Q-P block

    S[:ns_full, :ns_full] = S_PP
    S[ns_full:, ns_full:] = S_QQ

    return H, S


def prepare_bspline_data(t, ns, ks, n_gauss=None, kinetic_balance=False):
    """Prepare all B-spline data for Dirac matrix assembly.

    Parameters
    ----------
    t : ndarray, shape (ns + ks,)
        Knot sequence (for the full ks order).
    ns : int
        Number of B-splines.
    ks : int
        B-spline order.
    n_gauss : int, optional
    kinetic_balance : bool
        If True, use ksp=ks-1 for P, ksq=ks for Q (eliminates spurious states).
    """
    from atomx.bspline.basis import setup_gauss_points, evaluate_bsplines_on_grid

    nv = ns - ks + 1
    gr, grw = setup_gauss_points(t, ns, ks, n_gauss)

    if kinetic_balance:
        ksp = ks - 1
        ksq = ks
    else:
        ksp = ks
        ksq = ks

    nsp = nv + ksp - 1
    nsq = nv + ksq - 1  # = ns when ksq = ks

    # Evaluate Q-component B-splines (full order ksq = ks)
    bsp_q_full = evaluate_bsplines_on_grid(t, nsq, ksq, gr, nderiv=2)
    bsp_q = bsp_q_full[:, :, :ksq, 0]
    bsp_q_d = bsp_q_full[:, :, :ksq, 1]

    if ksp == ksq:
        # Same order: share B-spline arrays
        bsp_p = bsp_q.copy()
        bsp_p_d = bsp_q_d.copy()
    else:
        # RKB: build separate P-component B-splines on dedicated knot sequence
        tp, nsp_check = _build_knot_sequence(t, ks, ns, ksp)
        assert nsp_check == nsp
        bsp_p_full = evaluate_bsplines_on_grid(tp, nsp, ksp, gr, nderiv=2)
        bsp_p = bsp_p_full[:, :, :ksp, 0]
        bsp_p_d = bsp_p_full[:, :, :ksp, 1]

    ns_full = max(nsp, nsq)

    return {
        "nv": nv,
        "ks": ks,
        "ksp": ksp,
        "ksq": ksq,
        "nsp": nsp,
        "nsq": nsq,
        "gr": gr,
        "grw": grw,
        "bsp_p": bsp_p,
        "bsp_q": bsp_q,
        "bsp_q_d": bsp_q_d,
        "bsp_p_d": bsp_p_d,
        "t": t,
        "ns": ns_full,  # max(nsp, nsq) for backward compatibility
    }
