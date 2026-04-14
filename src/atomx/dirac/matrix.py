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

    # --- Bloch boundary operator (restores variational principle with KB) ---
    if ksp != ksq:
        tmax = bspline_data["t"][bspline_data.get("nsq", ns)]  # boundary radius
        if tmax > 0:
            pnu = 1.0  # Q(R)/P(R) ratio at boundary (DBSR_HF default)
            RN = kappa / (2.0 * tmax * C)  # (RB+kappa)/(2*tmax*c), RB=0
            ip = nsp - 1       # last P-spline index (0-based)
            iq = ns_full + nsq - 1  # last Q-spline index (0-based)
            H[ip, ip] -= pnu * RN * C
            H[ip, iq] += pnu * C
            H[iq, ip] += (pnu - 1.0) * C  # = 0 when pnu=1
            H[iq, iq] -= (pnu - 1.0) / RN * C if abs(RN) > 1e-30 else 0.0

    return H, S


