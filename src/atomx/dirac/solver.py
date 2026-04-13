"""Dirac eigenvalue solver using scipy.linalg.eigh (LAPACK dsygv)."""

import numpy as np
from scipy import linalg
from dataclasses import dataclass, field
from atomx.constants import C_SQ


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
      ns_full = max(nsp, nsq)
      PP block: H[0:ns_full, 0:ns_full]
      QQ block: H[ns_full:2*ns_full, ns_full:2*ns_full]
      PQ/QP coupling blocks accordingly.

    P coefficients use indices 0..nsp-1, Q uses ns_full..ns_full+nsq-1.
    Boundary removal excludes first/last B-splines from each component.
    """
    if nsp is None:
        nsp = ns
    if nsq is None:
        nsq = ns
    ns_full = max(nsp, nsq)
    ms = 2 * ns_full
    nb = n_remove_boundary

    if nb > 0:
        # P indices: nb..nsp-nb-1 (within [0, ns_full))
        keep_p = list(range(nb, nsp - nb))
        # Q indices: ns_full+nb..ns_full+nsq-nb-1 (within [ns_full, 2*ns_full))
        keep_q = list(range(ns_full + nb, ns_full + nsq - nb))
        keep = keep_p + keep_q
        nk = len(keep)
        H_red = H[np.ix_(keep, keep)]
        S_red = S[np.ix_(keep, keep)]
    else:
        # Use all P and Q indices (including zero-padded rows for nsp<ns_full)
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
