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


# Module-level cache for (nsp, nsq, nb, ibzero) -> (keep indices, np.ix_ grid).
# The boundary-removal mask is geometry-only: fixed across SCF iterations,
# so we build it once per basis shape and reuse.
_solve_dirac_basis_cache: dict = {}


def _basis_cache_key(nsp: int, nsq: int, nb: int, ibzero: int) -> tuple:
    return (int(nsp), int(nsq), int(nb), int(ibzero))


def _get_keep_grid(ns_full: int, nsp: int, nsq: int, nb: int, ibzero: int):
    """Return ``(keep_indices, np.ix_ grid)`` for the reduced submatrix.

    Cached by basis geometry — each SCF iteration calls ``solve_dirac`` once
    per target orbital (~24× on Pb, ~30 iters ⇒ ~720 calls), and rebuilding
    the index list + fresh ``np.ix_`` grid per call was measurable Python
    overhead. The cache key is intentionally coarse: two solvers with
    matching ``(nsp, nsq, nb, ibzero)`` share an entry.
    """
    key = _basis_cache_key(nsp, nsq, nb, ibzero)
    cached = _solve_dirac_basis_cache.get(key)
    if cached is not None:
        return cached
    if nb > 0:
        keep_p = np.arange(nb, nsp - ibzero, dtype=np.intp)
        keep_q = np.arange(ns_full + nb, ns_full + nsq - ibzero, dtype=np.intp)
    else:
        keep_p = np.arange(0, nsp, dtype=np.intp)
        keep_q = np.arange(ns_full, ns_full + nsq, dtype=np.intp)
    keep = np.concatenate([keep_p, keep_q])
    grid = np.ix_(keep, keep)
    _solve_dirac_basis_cache[key] = (keep, grid)
    return keep, grid


def clear_solve_dirac_caches() -> None:
    """Drop the reduction-mask cache. Benchmarks and tests that measure
    steady-state hot-loop cost should call this first so the initial
    cache-miss of the first iteration is absorbed."""
    _solve_dirac_basis_cache.clear()


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

    Performance notes
    -----------------
    Two micro-optimisations on the SCF hot path (bit-for-bit identical
    numerics to the naïve reference):

    1. The ``np.ix_`` reduction grid for the boundary-stripping submatrix
       depends only on basis geometry and is cached in
       ``_solve_dirac_basis_cache`` so it is built once per ``(nsp, nsq,
       nb, ibzero)`` instead of per call.
    2. ``check_finite=False`` is passed to ``scipy.linalg.eigh`` — scipy
       otherwise does a full-matrix NaN/Inf scan on every call, which is
       O(ms²) Python-level work that dominates for medium matrices.

    Subset-selection drivers (``subset_by_value`` / ``subset_by_index``)
    were investigated and rejected: for the ~200–500-dim matrices atomx
    produces, LAPACK's ``dsygv`` (full spectrum) beats ``dsygvx`` (subset
    via bisection + inverse iteration) by 2–3×. The subset drivers only
    pay off for banded or much larger problems.
    """
    if nsp is None:
        nsp = ns
    if nsq is None:
        nsq = ns
    ns_full = max(nsp, nsq)
    ms = 2 * ns_full
    nb = n_remove_boundary
    ibzero = 2  # DBSR_HF default: remove 2 boundary B-splines at tail

    keep, grid = _get_keep_grid(ns_full, nsp, nsq, nb, ibzero)
    H_red = H[grid]
    S_red = S[grid]

    # Solve the full generalised eigenproblem. Empirically for the matrix
    # sizes produced by atomx (~200–500), LAPACK ``dsygv`` /
    # ``dsygvd`` (divide-and-conquer) beats the subset-aware ``dsygvx``
    # (bisection + inverse iteration) by 2–3× — subset drivers only win
    # for banded or much larger problems. ``check_finite=False`` skips a
    # full-matrix scan that scipy otherwise does on every call.
    eigenvalues, eigenvectors = linalg.eigh(H_red, S_red, check_finite=False)

    sea_threshold = -2.0 * C_SQ
    physical_mask = eigenvalues > sea_threshold
    physical_eigenvalues = eigenvalues[physical_mask]

    # Embed the reduced eigenvectors back into the full ms-dim layout
    # (only physical branch; Dirac-sea columns are dropped).
    full_vecs = np.zeros((ms, physical_eigenvalues.size))
    full_vecs[keep, :] = eigenvectors[:, physical_mask]

    bound_mask = (eigenvalues > sea_threshold) & (eigenvalues < 0)
    bound_energies = np.sort(eigenvalues[bound_mask])

    return DiracResult(
        kappa=kappa,
        eigenvalues=physical_eigenvalues,
        bound_energies=bound_energies,
        coefficients=full_vecs,
        ns=ns_full,
        nsp=nsp,
        nsq=nsq,
    )
