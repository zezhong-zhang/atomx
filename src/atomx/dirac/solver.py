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
        """Get P and Q coefficients for the idx-th bound state."""
        bound_mask = (self.eigenvalues > -2.0 * C_SQ) & (self.eigenvalues < 0)
        bound_indices = np.where(bound_mask)[0]
        sorted_order = np.argsort(self.eigenvalues[bound_indices])
        eigvec_idx = bound_indices[sorted_order[idx]]

        vec = self.coefficients[:, eigvec_idx]
        # P occupies first nsp entries, Q occupies next nsq entries
        P = np.zeros(max(self.nsp, self.nsq))  # pad to equal size for compatibility
        Q = np.zeros(max(self.nsp, self.nsq))
        P[:self.nsp] = vec[:self.nsp]
        Q[:self.nsq] = vec[self.nsp:self.nsp + self.nsq]
        return P, Q, self.bound_energies[idx]


def solve_dirac(H, S, ns, kappa, n_remove_boundary=1, nsp=None, nsq=None):
    """Solve the generalized Dirac eigenvalue problem.

    Handles both symmetric (nsp=nsq=ns) and asymmetric (RKB: nsp < nsq) cases.

    Parameters
    ----------
    H, S : ndarray, shape (ms, ms)
    ns : int
        Number of Q-component B-splines (backward compat; use nsq if given).
    kappa : int
    n_remove_boundary : int
    nsp, nsq : int, optional
        P and Q dimensions. Default: both = ns.
    """
    if nsp is None:
        nsp = ns
    if nsq is None:
        nsq = ns
    ms = nsp + nsq
    nb = n_remove_boundary

    if nb > 0:
        keep_p = list(range(nb, nsp - nb))
        keep_q = list(range(nsp + nb, ms - nb))
        keep = keep_p + keep_q
        nk = len(keep)
        H_red = H[np.ix_(keep, keep)]
        S_red = S[np.ix_(keep, keep)]
    else:
        H_red = H
        S_red = S
        keep = list(range(ms))
        nk = ms

    eigenvalues, eigenvectors = linalg.eigh(H_red, S_red)

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
        ns=nsq,
        nsp=nsp,
        nsq=nsq,
    )
