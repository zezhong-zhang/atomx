"""B-spline assembly and eigenvalue solve for the Dirac equation.

Moved here from the old ``atomx.dirac.matrix`` and
``atomx.dirac.solver`` modules as part of the physics/numerics axis
refactor: the physics-level equation object lives in
:class:`atomx.physics.dirac.Dirac`; the B-spline-specific matrix
construction, Bloch boundary operator, and generalised-eigenvalue
solve live here.

Supports Restricted Kinetic Balance (RKB): P uses order ksp = ks − 1,
Q uses ksq = ks.  Under RKB four separate coupling matrices are
needed (not transposes of each other):

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
from atomx.numerics.bspline.integrals import (
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



# ---------------------------------------------------------------------------
# Generalised-eigenvalue solver
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field

from scipy import linalg


@dataclass
class DiracResult:
    """Result of a Dirac eigenvalue calculation for a given kappa."""
    kappa: int
    eigenvalues: np.ndarray       # physical eigenvalues (Dirac sea excluded)
    bound_energies: np.ndarray    # physical bound states (-2c^2 < E < 0)
    coefficients: np.ndarray      # eigenvectors, shape (ms, n_states)
    ns: int                       # nsq (Q-component dimension, for backward compat)
    nsp: int = 0                  # P-component dimension
    nsq: int = 0                  # Q-component dimension
    n_bound: int = field(init=False)

    def __post_init__(self):
        self.n_bound = len(self.bound_energies)
        if self.nsp == 0:
            self.nsp = self.ns
        if self.nsq == 0:
            self.nsq = self.ns

    def get_orbital(self, idx):
        """Get P and Q coefficients for the idx-th bound state.

        Layout: P in vec[0:nsp], Q in vec[ns_full:ns_full+nsq]
        where ns_full = max(nsp, nsq) = self.ns.
        Returns P, Q both zero-padded to length ns_full.
        """
        bound_mask = (self.eigenvalues > -2.0 * C_SQ) & (self.eigenvalues < 0)
        bound_indices = np.where(bound_mask)[0]
        sorted_order = np.argsort(self.eigenvalues[bound_indices])
        eigvec_idx = bound_indices[sorted_order[idx]]

        vec = self.coefficients[:, eigvec_idx]
        ns_full = self.ns  # = max(nsp, nsq)
        P = np.zeros(ns_full)
        Q = np.zeros(ns_full)
        P[:self.nsp] = vec[:self.nsp]
        Q[:self.nsq] = vec[ns_full:ns_full + self.nsq]
        return P, Q, self.bound_energies[idx]


def solve_dirac(H, S, ns, kappa, n_remove_boundary=1, nsp=None, nsq=None):
    """Solve the generalized Dirac eigenvalue problem.

    Handles both symmetric (nsp=nsq=ns) and RKB (nsp < nsq) cases.

    Matrix layout (DBSR_HF convention):
      ns_full = max(nsp, nsq) = ns (for standard grids)
      PP block: H[0:ns_full, 0:ns_full]  (only nsp rows/cols nonzero)
      QQ block: H[ns_full:2*ns_full, ns_full:2*ns_full]
      ms = 2 * ns_full

    Boundary conditions (DBSR_HF ilzero=1, ibzero=2):
      P: remove first n_remove_boundary splines + last ibzero splines + zero-padding
      Q: remove first n_remove_boundary splines + last ibzero splines
      ibzero defaults to 2 (DBSR_HF standard).
    """
    if nsp is None:
        nsp = ns
    if nsq is None:
        nsq = ns
    ns_full = max(nsp, nsq)
    ms = 2 * ns_full
    nb = n_remove_boundary
    ibzero = 2  # DBSR_HF default: remove 2 boundary B-splines at tail

    if nb > 0:
        # P: keep indices nb..nsp-ibzero-1 (remove nb at start, ibzero at end,
        # plus any zero-padded indices nsp..ns_full-1)
        keep_p = list(range(nb, nsp - ibzero))
        # Q: keep indices ns_full+nb..ns_full+nsq-ibzero-1
        keep_q = list(range(ns_full + nb, ns_full + nsq - ibzero))
        keep = keep_p + keep_q
        nk = len(keep)
        H_red = H[np.ix_(keep, keep)]
        S_red = S[np.ix_(keep, keep)]
    else:
        keep_p = list(range(nsp))
        keep_q = list(range(ns_full, ns_full + nsq))
        keep = keep_p + keep_q
        nk = len(keep)
        H_red = H[np.ix_(keep, keep)]
        S_red = S[np.ix_(keep, keep)]

    eigenvalues, eigenvectors = linalg.eigh(H_red, S_red)

    # Map back to full ms-dimensional vectors
    full_vecs = np.zeros((ms, nk))
    for col in range(nk):
        for i, k in enumerate(keep):
            full_vecs[k, col] = eigenvectors[i, col]

    sea_threshold = -2.0 * C_SQ
    physical_mask = eigenvalues > sea_threshold
    physical_eigenvalues = eigenvalues[physical_mask]
    physical_vecs = full_vecs[:, physical_mask]

    bound_mask = (eigenvalues > sea_threshold) & (eigenvalues < 0)
    bound_energies = np.sort(eigenvalues[bound_mask])

    return DiracResult(
        kappa=kappa,
        eigenvalues=physical_eigenvalues,
        bound_energies=bound_energies,
        coefficients=physical_vecs,
        ns=ns_full,
        nsp=nsp,
        nsq=nsq,
    )
