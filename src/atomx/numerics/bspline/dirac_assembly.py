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


def apply_dirac_hamiltonian(P, Q, kappa, bspline_data, *, potential):
    """Evaluate h_D on the native Gauss grid using exact spline derivatives.

    Uses the common physics Dirac action. ``potential`` is explicitly
    supplied on that same grid; no nuclear or model fallback is inferred.
    This differential action excludes the matrix solver's Bloch boundary
    operator. Bound spinors used for comparison must vanish at the box
    boundary. Coefficients obey the existing P/Q kinetic-balance layout.
    """
    from atomx.numerics.bspline.evaluate import eval_orbital_on_grid, eval_P, eval_Q
    from atomx.physics.dirac import Dirac

    data = bspline_data
    p, q = eval_P(P, data), eval_Q(Q, data)
    dp = eval_orbital_on_grid(
        P,
        data["bsp_p_d"],
        data["nv"],
        data.get("ksp", data["ks"]),
        data.get("nsp", data["ns"]),
    )
    dq = eval_orbital_on_grid(
        Q,
        data["bsp_q_d"],
        data["nv"],
        data.get("ksq", data["ks"]),
        data.get("nsq", data["ns"]),
    )
    return Dirac().radial_action(p, q, dp, dq, data["gr"], potential, kappa)


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

    def get_orbital(self, idx, *, tolerance=None):
        """Get P and Q coefficients for the idx-th bound state.

        Layout: P in vec[0:nsp], Q in vec[ns_full:ns_full+nsq]
        where ns_full = max(nsp, nsq) = self.ns.
        Returns P, Q both zero-padded to length ns_full.

        ``tolerance`` requests an inner eigenpair accuracy check. A spectral
        residual estimates the S-norm orbital error and relative energy
        error; only an inaccurate requested root is Newton-refined.
        This is an algebraic check, not a radial basis error estimate.
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
        if tolerance is not None and (not np.isfinite(tolerance) or tolerance <= 0):
            raise ValueError('eigenpair tolerance must be finite and positive')
        if tolerance is not None and self._refine_orbital is None:
            raise ValueError('eigenpair accuracy requested with refinement disabled')
        if (self._refine_orbital is not None and
                (tolerance is not None or (idx + ell >= 10 and eigvec_idx not in self._refined_indices))):
            energy, vector = self._refine_orbital(eigvec_idx, tolerance)
            self.eigenvalues[eigvec_idx] = energy
            self.bound_energies[idx] = energy
            self.coefficients[:, eigvec_idx] = vector

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


def _refine_eigenpair(H, S, energy, vector, gap, *, residual_norm=None):
    """Bordered Newton refinement in the actual constrained eigenbasis.

    The border enforces S-normalization and removes the eigenvector's null
    direction. No shift, potential approximation, or change of root is used.
    """
    h = np.tril(H) + np.tril(H, -1).T
    s = np.tril(S) + np.tril(S, -1).T
    original = vector.copy()
    x = original.copy()
    initial_energy = float(energy)
    measure = linalg.norm if residual_norm is None else residual_norm
    best = (float(measure(h @ x - energy * (s @ x))), energy, x.copy())
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
        error = float(measure(h @ x - energy * (s @ x)))
        if error < best[0]:
            best = (error, energy, x.copy())
    return best[1], best[2]


def refine_generalized_eigenblock(H, S, energies, coefficients, *, leading_count,
                                dimension_cap=256, column_cap=128,
                                memory_cap_bytes=128*1024**2):
    """Refine a leading block in the original full real generalized pencil.

    H's lower triangle defines the symmetric operator. S is its explicit
    positive metric. All supplied sorted columns are returned; only the
    declared leading block receives two-step bordered corrections. Corrections
    may leave the input numerical span: no restriction to that approximate
    invariant subspace is imposed. This does not select a physical branch.

    Ordered metric Cholesky QR jointly restores all columns. A shifted Ritz
    solve only in the leading block avoids mixing its small energies with
    the unselected high-energy spectrum. Wide contractions preserve the same
    supplied FP64 H/S. No eigenstate or coordinate is screened. Input/output
    Gram, ordering, root shifts and overlap are checked; full S-dual residuals
    are returned for the caller's unchanged scientific admission threshold.
    NumPy working-payload admission excludes Python/BLAS/SciPy RSS overhead.
    """
    arrays = tuple(np.asarray(a) for a in (H, S, energies, coefficients))
    if any(np.iscomplexobj(a) for a in arrays):
        raise ValueError('real generalized eigenblock required')
    h0, s0, e0, v0 = arrays
    n = h0.shape[0] if h0.ndim == 2 else 0
    m = e0.size if e0.ndim == 1 else 0
    for name, value, minimum in [('dimension_cap', dimension_cap, 1),
                                  ('column_cap', column_cap, 1),
                                  ('memory_cap_bytes', memory_cap_bytes, 1),
                                  ('leading_count', leading_count, 1)]:
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or value < minimum):
            raise ValueError(f'positive integer {name} required')
    if (not n or not m or h0.shape != (n, n) or s0.shape != (n, n)
            or v0.shape != (n, m) or leading_count > m or m > n):
        raise ValueError('inconsistent complete generalized eigenblock dimensions')
    estimate = 128*(2*n*n+n*m+m*m+2*n+1)
    if n > dimension_cap or m > column_cap or estimate > memory_cap_bytes:
        raise MemoryError('complete generalized eigenblock exceeds resource admission')
    if any(not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError('finite generalized eigenblock arrays required')
    if not np.array_equal(s0, s0.T):
        raise ValueError('exactly symmetric overlap metric required')
    if m < 2 or np.any(np.diff(e0) <= 0):
        raise ValueError('strictly ordered isolated input roots required')
    if np.finfo(np.longdouble).nmant <= np.finfo(float).nmant:
        raise RuntimeError('block refinement requires extended contractions')
    h = np.array(np.tril(h0)+np.tril(h0, -1).T, dtype=float)
    s = np.array(s0, dtype=float, copy=True)
    e = np.array(e0, dtype=float, copy=True)
    original = np.array(v0, dtype=float, copy=True)
    v = original.copy()
    gram0 = float(np.max(abs(v.T @ s @ v-np.eye(m))))
    if gram0 > 1e-10:
        raise ValueError('input generalized eigenblock failed Gram gate')
    lower = linalg.cholesky(s, lower=True, check_finite=False)

    def dual_norm(r):
        return float(linalg.norm(linalg.solve_triangular(lower, r, lower=True,
                                                        check_finite=False)))

    gaps = np.minimum(np.r_[np.inf, np.diff(e)], np.r_[np.diff(e), np.inf])
    for column in range(leading_count):
        e[column], v[:, column] = _refine_eigenpair(
            h, s, e[column], v[:, column], gaps[column], residual_norm=dual_norm)
    wide = np.longdouble
    sw, hw, vw = s.astype(wide), h.astype(wide), v.astype(wide)
    gram = np.asarray(vw.T @ (sw @ vw), float)
    upper = linalg.cholesky(gram, lower=False, check_finite=False)
    v = linalg.solve_triangular(upper.T, v.T, lower=True,
                                check_finite=False).T
    leading = v[:, :leading_count].astype(wide)
    center = np.mean(e[:leading_count].astype(wide))
    reduced = leading.T @ ((hw-center*sw) @ leading)
    reduced_s = leading.T @ (sw @ leading)
    shifted, rotation = linalg.eigh(np.asarray(reduced, float), np.asarray(reduced_s, float),
                                    driver='gvd', check_finite=False)
    v[:, :leading_count] = np.asarray(leading @ rotation.astype(wide), float)
    e[:leading_count] = np.asarray(center+shifted.astype(wide), float)
    overlap = np.sum(original*(s @ v), axis=0)
    v[:, overlap < 0] *= -1
    overlap = abs(overlap)
    change = abs(e-np.asarray(e0)) / gaps
    if np.any(np.diff(e) <= 0) or np.max(change) > .25 or np.min(overlap) < .99:
        raise linalg.LinAlgError('joint generalized block left an original root')
    gram_error = float(np.max(abs(v.T @ s @ v-np.eye(m))))
    if gram_error > 1e-10:
        raise linalg.LinAlgError('joint generalized eigenblock failed original Gram gate')
    residual = h @ v - (s @ v)*e
    whitened = linalg.solve_triangular(lower, residual, lower=True, check_finite=False)
    residuals = np.linalg.norm(whitened, axis=0)
    if not np.all(np.isfinite(residuals)):
        raise linalg.LinAlgError('nonfinite joint generalized block residual')
    info = dict(dimension=n, columns=m, leading_count=int(leading_count),
                estimated_working_bytes=estimate, bordered_solves_max=2*int(leading_count),
                cholesky_calls=2, leading_ritz_solves=1,
                original_gram_error=gram0, gram_error=gram_error,
                minimum_original_column_overlap=float(overlap.min()),
                maximum_root_shift_gap_ratio=float(change.max()),
                full_dual_residuals_Ha=residuals.tolist(),
                maximum_leading_dual_residual_Ha=float(residuals[:leading_count].max()),
                numerical_span_policy='full original coordinates, joint metric orthogonalization; no fixed-span restriction')
    return e, v, info


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

    refined_indices = set()

    def refine(index, tolerance=None):
        full_index = np.flatnonzero(physical_mask)[index]
        gaps = np.abs(eigenvalues - eigenvalues[full_index])
        gaps[full_index] = np.inf
        energy, vector = eigenvalues[full_index], vectors[:, full_index]
        needed = True
        if tolerance is not None:
            # In an S-orthonormal basis, v_j^T(H-eS)x/(e_j-e) estimates
            # the unwanted mode amplitudes. Retain negative-energy modes
            # as well: they also contribute to small-component errors.
            h = np.tril(constrained_h) + np.tril(constrained_h, -1).T
            s = np.tril(constrained_s) + np.tril(constrained_s, -1).T
            extended = vector.astype(np.longdouble)
            residual = np.asarray(h.astype(np.longdouble) @ extended
                - np.longdouble(energy) * (s.astype(np.longdouble) @ extended), dtype=float)
            projected = vectors.T @ residual
            orbital_error = linalg.norm(np.divide(projected, gaps,
                out=np.zeros_like(projected), where=gaps > 0))
            energy_error = abs(float(vector @ residual)) / max(abs(energy), np.finfo(float).tiny)
            needed = orbital_error > tolerance or energy_error > tolerance
            if not needed:
                # Keep the original GEMM embedding bit-for-bit. Recomputing
                # one constrained column with GEMV can change summation noise.
                return float(energy), full_vecs[:, index]
        if needed:
            energy, vector = _refine_eigenpair(
                constrained_h, constrained_s, energy, vector, float(np.min(gaps)))
            eigenvalues[full_index] = energy
            vectors[:, full_index] = vector
            refined_indices.add(index)
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
        _refined_indices=refined_indices,
    )
