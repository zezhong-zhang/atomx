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

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

from atomx.constants import C_SQ, C
from atomx.numerics.bspline.integrals import (
    coupling_deriv_matrix,
    coupling_matrix,
    nuclear_matrix,
    overlap_matrix,
)

#: Lower edge of the negative-energy continuum, ``-m_e c²``.
#:
#: Bound Dirac-Coulomb states satisfy ``-m_e c² < E < 0``; everything at
#: or below ``-m_e c²`` is the negative-energy sea, not an atomic orbital.
#: This used to be ``-2 m_e c²``, which let basis-set states that have
#: collapsed towards the sea (measured at ``-1.99996 m_e c²``) pass as
#: "bound": a requested d/f subshell with no genuine bound state then
#: silently returned a sea state, giving total energies off by ~10^4 Ha
#: with ``converged=True``.  See
#: ``docs/archive/2026-09/PHYSICS_NUMERICS_REVIEW_2026-09-19.md`` finding C4.
NEGATIVE_CONTINUUM_EDGE = -C_SQ


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
    _refine_orbital: object = field(default=None, repr=False)
    _refined_indices: set = field(default_factory=set, repr=False)

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
        bound_mask = (self.eigenvalues > NEGATIVE_CONTINUUM_EDGE) & (self.eigenvalues < 0)
        bound_indices = np.where(bound_mask)[0]
        sorted_order = np.argsort(self.eigenvalues[bound_indices])
        eigvec_idx = bound_indices[sorted_order[idx]]

        # Diffuse high roots can have eigenvector errors far above SCF tolerance
        # despite a backward-stable full-spectrum solve. Refine only requested
        # roots; refining all unused continuum states would dominate runtime.
        # Angular momentum also makes low-index roots diffuse: He 16i has
        # index 9 and l=6. Counting radial roots alone missed its inaccurate
        # eigenvector and prevented SCF convergence. For an unconstrained
        # channel, idx+l+1 is the principal quantum number.
        ell = self.kappa if self.kappa > 0 else -self.kappa - 1
        if idx + ell >= 10 and self._refine_orbital is not None and eigvec_idx not in self._refined_indices:
            energy, vector = self._refine_orbital(eigvec_idx)
            self.eigenvalues[eigvec_idx] = energy
            self.bound_energies[idx] = energy
            self.coefficients[:, eigvec_idx] = vector
            self._refined_indices.add(eigvec_idx)

        vec = self.coefficients[:, eigvec_idx]
        ns_full = self.ns  # = max(nsp, nsq)
        P = np.zeros(ns_full)
        Q = np.zeros(ns_full)
        P[:self.nsp] = vec[:self.nsp]
        Q[:self.nsq] = vec[ns_full:ns_full + self.nsq]
        P, Q = canonicalize_radial_phase(P, Q)
        return P, Q, self.bound_energies[idx]


def canonicalize_radial_phase(P, Q):
    """Return a deterministic sign for a radial spinor.

    Generalised eigensolvers may return either phase for an eigenvector.
    Densities and diagonal energies are unchanged, but CI, transition, and
    scattering amplitudes contain products of different orbitals and need a
    stable convention across basis sizes and solver backends. B-splines are
    non-negative locally, so the largest large-component coefficient is a
    robust proxy for the dominant radial lobe.
    """
    P_arr = np.asarray(P)
    if P_arr.size == 0:
        return P, Q
    i = int(np.argmax(np.abs(P_arr)))
    if float(P_arr[i]) < 0.0:
        return -P, -Q
    return P, Q


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




def _eigh_finite(H_red, S_red, max_retries=2):
    """``linalg.eigh`` with a transient-NaN tripwire (HYP-1590).

    Under extreme CPU oversubscription (box load > 100 with many
    concurrent runs) a BLAS/LAPACK call has been observed to return
    non-finite eigenvalues from finite inputs; an immediate re-issue of
    the same call recovers.  Gate the solver output on finiteness: retry
    the identical call a few times, and raise loudly if the corruption
    persists instead of letting a silent NaN poison the SCF.

    The finiteness scan is O(ms) on the eigenvalues only — negligible
    next to the O(ms^3) factorisation (the full-matrix input scan is
    what ``check_finite=False`` avoids, and stays off).
    """
    last_w = None
    for _attempt in range(max_retries + 1):
        # driver="gvd" (divide-and-conquer) on the advice of measurement
        # (critical review F2, 2026-09-27): at the 220–550-dim sizes atomx
        # produces it is tied with "gv" single-threaded and up to 1.7x faster
        # multi-threaded (549x549: 47 ms vs 80 ms at 4 threads), with
        # bit-identical eigenvalues (max |w_gvd - w_gv| = 0.0 on the real
        # Ne ns=113 matrices).  Subset drivers stay rejected, see solve_dirac.
        w, v = linalg.eigh(H_red, S_red, driver="gvd", check_finite=False)
        if np.all(np.isfinite(w)):
            return w, v
        last_w = w
    raise linalg.LinAlgError(
        "solve_dirac: eigh returned non-finite eigenvalues "
        f"({int(np.isnan(last_w).sum())} NaN of {last_w.size}) on "
        f"{max_retries + 1} consecutive attempts from finite input — "
        "transient numerical corruption did not recover (HYP-1590)."
    )


def _refine_eigenpair(H, S, energy, vector, gap):
    """Bordered Newton refinement in the actual constrained eigenbasis.

    The border enforces S-normalization and removes the eigenvector's null
    direction. No shift, potential approximation, or change of root is used.
    """
    h = np.tril(H) + np.tril(H, -1).T
    s = np.tril(S) + np.tril(S, -1).T
    original = vector.copy()
    x = original.copy()
    initial_energy = float(energy)
    best = (float(linalg.norm(h @ x - energy * (s @ x))), energy, x.copy())
    for _ in range(2):
        sx = s @ x
        residual = h @ x - energy * sx
        if not np.any(residual):
            break
        augmented = np.block([[h-energy*s, -sx[:, None]],
                              [-sx[None, :], np.zeros((1, 1))]])
        delta = linalg.solve(augmented, np.r_[-residual, 0.],
                             assume_a='sym', check_finite=True)
        x += delta[:-1]
        norm = float(x @ s @ x)
        if not np.isfinite(norm) or norm <= 0:
            raise linalg.LinAlgError('invalid refined orbital norm')
        x /= np.sqrt(norm)
        extended = x.astype(np.longdouble)
        energy = float(extended @ h.astype(np.longdouble) @ extended /
                       (extended @ s.astype(np.longdouble) @ extended))
        if (abs(energy-initial_energy) > 0.25 * gap
                or abs(original @ s @ x) < 0.99):
            raise linalg.LinAlgError('eigenvector refinement left the selected root')
        error = float(linalg.norm(h @ x - energy * (s @ x)))
        if error < best[0]:
            best = (error, energy, x.copy())
    return best[1], best[2]


def solve_dirac(H, S, ns, kappa, n_remove_boundary=1, nsp=None, nsq=None,
                orthogonal_to=None, refine_rydberg=True):
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
    pay off for banded or much larger problems.  (They are also
    structurally unusable here: callers take the *whole* physical branch
    above the Dirac-sea edge as the positive-energy projector, not the
    lowest k roots.)  Within full-spectrum drivers, ``gvd``
    (divide-and-conquer) is used on measured grounds — tied with ``gv``
    at one thread, up to 1.7× faster at four, bit-identical eigenvalues
    (critical review F2, 2026-09-27).
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
    complement = None

    # Solve the full generalised eigenproblem. Empirically for the matrix
    # sizes produced by atomx (~200–500), LAPACK ``dsygv`` /
    # ``dsygvd`` (divide-and-conquer) beats the subset-aware ``dsygvx``
    # (bisection + inverse iteration) by 2–3× — subset drivers only win
    # for banded or much larger problems. ``check_finite=False`` skips a
    # full-matrix scan that scipy otherwise does on every call.
    if orthogonal_to is not None and np.size(orthogonal_to):
        constraints = np.asarray(orthogonal_to)[keep, :]
        normals = S_red @ constraints
        basis, triangular = linalg.qr(normals, mode="full", check_finite=False)
        count = constraints.shape[1]
        if np.any(np.abs(np.diag(triangular[:count])) < 1e-13):
            raise ValueError("linearly dependent orbital orthogonality constraints")
        complement = basis[:, count:]
        # As in eigh(lower=True), the lower triangle defines H. Some
        # exchange builders intentionally omit the unused upper block.
        symmetric_h = np.tril(H_red) + np.tril(H_red, -1).T
        constrained_h = complement.T @ (symmetric_h @ complement)
        constrained_s = complement.T @ (S_red @ complement)
        eigenvalues, vectors = _eigh_finite(constrained_h, constrained_s)
        eigenvectors = complement @ vectors
    else:
        eigenvalues, eigenvectors = _eigh_finite(H_red, S_red)
        constrained_h, constrained_s = H_red, S_red
        vectors = eigenvectors

    sea_threshold = NEGATIVE_CONTINUUM_EDGE
    physical_mask = eigenvalues > sea_threshold
    physical_eigenvalues = eigenvalues[physical_mask]

    # Embed the reduced eigenvectors back into the full ms-dim layout
    # (only physical branch; Dirac-sea columns are dropped).
    full_vecs = np.zeros((ms, physical_eigenvalues.size))
    full_vecs[keep, :] = eigenvectors[:, physical_mask]

    bound_mask = (eigenvalues > sea_threshold) & (eigenvalues < 0)
    bound_energies = np.sort(eigenvalues[bound_mask])

    def refine(index):
        full_index = np.flatnonzero(physical_mask)[index]
        gaps = np.abs(eigenvalues - eigenvalues[full_index])
        gaps[full_index] = np.inf
        energy, vector = _refine_eigenpair(
            constrained_h, constrained_s, eigenvalues[full_index],
            vectors[:, full_index], float(np.min(gaps)))
        embedded = np.zeros(ms)
        embedded[keep] = complement @ vector if complement is not None else vector
        return energy, embedded

    return DiracResult(
        kappa=kappa,
        eigenvalues=physical_eigenvalues,
        bound_energies=bound_energies,
        coefficients=full_vecs,
        ns=ns_full,
        nsp=nsp,
        nsq=nsq,
        _refine_orbital=refine if refine_rydberg else None,
    )
