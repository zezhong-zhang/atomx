"""CI Hamiltonian matrix construction for MCDHF.

Slater-Condon rules:
- 0-diff (diagonal): full 1e + 2e energy of the configuration
- 1-diff (a→b):   <a|h|b> + Σ_c q_c [R^0(ac,bc) - Σ_k c^k R^k(ac,cb)]
- 2-diff (ab→cd): Σ_k [direct R^k(ac,bd) - exchange R^k(ac,db)]
- 3+ diff:        0  (orthogonality)

For CSFs with explicit J-resolution (``total_J2`` set, intermediate
parentage inferable), a J-coupled angular correction is added on top of
the subshell-occupation kernel so that distinct (J, parity) blocks
acquire physically meaningful J-dependent splittings.  See
``_diagonal_J_correction`` and Grant, *Relativistic Quantum Theory of
Atoms and Molecules* (2007) §8.3.
"""

import numpy as np
from atomx.numerics.bspline.convol import (
    ConvolCache,
    compute_slater_fk_cell,
    compute_slater_gk_cell,
    compute_slater_rk_cell,
)
from atomx.numerics.shooting.constants import SL
from atomx.numerics.shooting.quadrature import slag6
from atomx.numerics.shooting.solver import _slater_integral_slag6


# Build-scoped memo for Slater integrals and Dirac H0/S0 blocks: every CSF
# in a CI build shares ONE orbital basis, so the same (orbitals, k)
# integrals recur across all diagonal/off-diagonal elements — a Co ground
# CI rebuilt them ~300k times per MCDHF outer iteration (7600 elements x
# ~40 integrals each) where only a few hundred are distinct.  Keys use
# id() of the orbital coefficient arrays, which orbitals_dict keeps alive
# for the duration of the build; build_ci_matrix clears the memo on entry
# so nothing leaks across outer iterations (whose orbitals differ).
#
# Two reserved keys carry the build context set by :func:`build_ci_matrix`:
#   "_convol_cache"      — ConvolCache with the cell-algorithm rkb tables.
#   "_nuclear_potential" — V(r) callable (finite nucleus) or None (point).
#
# Radial integrals use the DBSR_HF cell algorithm (``convol.*_cell``) — the
# same production integrator the DFSolver energy functional uses.  The
# previous Y^k-quadrature adapters (``slater_yk``) accumulate quadrature
# error where the density varies rapidly inside a knot cell; at Z=82 that
# put F^0(1s,1s) off by 0.97 Ha and the CI total off by ~74 Ha even on
# perfectly converged orbitals (Pb 6p² under-binding bug).
_SLATER_MEMO: dict = {}


def _context_cache(bsp_data) -> ConvolCache:
    """Cell-integral rkb cache for the current build (lazily created).

    Standalone helper calls (unit tests poking ``_diagonal_element`` etc.
    without going through :func:`build_ci_matrix`) get a fresh cache bound
    to their ``bsp_data``; the identity check evicts a stale cache when the
    grid changes between direct calls.
    """
    cache = _SLATER_MEMO.get("_convol_cache")
    if cache is None or cache.bsp_data is not bsp_data:
        cache = ConvolCache(bsp_data)
        _SLATER_MEMO["_convol_cache"] = cache
    return cache


def _fk(Pa, Qa, Pb, Qb, bsp_data, k):
    ia, ib = id(Pa), id(Pb)
    key = ("f", k) + ((ia, ib) if ia <= ib else (ib, ia))
    v = _SLATER_MEMO.get(key)
    if v is None:
        if isinstance(bsp_data, GridCIWorkspace):
            v = float(_slater_integral_slag6(
                _grid_pair_density(Pa, Qa, Pa, Qa),
                _grid_pair_density(Pb, Qb, Pb, Qb),
                k, bsp_data.r, bsp_data.difr, bsp_data.n))
        else:
            v = compute_slater_fk_cell(Pa, Qa, Pb, Qb, bsp_data, k,
                                       cache=_context_cache(bsp_data))
        _SLATER_MEMO[key] = v
    return v


def _gk(Pa, Qa, Pb, Qb, bsp_data, k):
    ia, ib = id(Pa), id(Pb)
    key = ("g", k) + ((ia, ib) if ia <= ib else (ib, ia))
    v = _SLATER_MEMO.get(key)
    if v is None:
        if isinstance(bsp_data, GridCIWorkspace):
            rho_ab = _grid_pair_density(Pa, Qa, Pb, Qb)
            v = float(_slater_integral_slag6(
                rho_ab, rho_ab, k, bsp_data.r, bsp_data.difr, bsp_data.n))
        else:
            v = compute_slater_gk_cell(Pa, Qa, Pb, Qb, bsp_data, k,
                                       cache=_context_cache(bsp_data))
        _SLATER_MEMO[key] = v
    return v


def _rk(Pa, Qa, Pb, Qb, Pc, Qc, Pd, Qd, bsp_data, k):
    key = ("r", k, id(Pa), id(Pb), id(Pc), id(Pd))
    v = _SLATER_MEMO.get(key)
    if v is None:
        if isinstance(bsp_data, GridCIWorkspace):
            # rho13 = a·c, rho24 = b·d — same pairing convention as
            # compute_slater_rk_cell.
            v = float(_slater_integral_slag6(
                _grid_pair_density(Pa, Qa, Pc, Qc),
                _grid_pair_density(Pb, Qb, Pd, Qd),
                k, bsp_data.r, bsp_data.difr, bsp_data.n))
        else:
            v = compute_slater_rk_cell(Pa, Qa, Pb, Qb, Pc, Qc, Pd, Qd,
                                       bsp_data, k,
                                       cache=_context_cache(bsp_data))
        _SLATER_MEMO[key] = v
    return v


def _dirac_blocks(ns, kappa, Z, bsp_data):
    key = ("h0", kappa)
    v = _SLATER_MEMO.get(key)
    if v is None:
        v = build_dirac_matrices(ns, kappa, Z, bsp_data,
                                 _SLATER_MEMO.get("_nuclear_potential"))
        _SLATER_MEMO[key] = v
    return v


# ---------------------------------------------------------------------------
# Grid (shooting-engine) flavour of the radial workspace
# ---------------------------------------------------------------------------
#
# Same seam idea as atomx.corrections.workspace: ONE physics implementation
# of the CI matrix elements serves both orbital representations.  Dispatch is
# on the workspace object (``isinstance(bsp_data, GridCIWorkspace)``), never
# on array shapes.  The consistency principle that fixed the +74 Ha Pb bug
# applies verbatim here: the grid flavour evaluates every matrix element with
# the shooting engine's OWN numerics — Slater integrals through
# ``_slater_integral_slag6`` (the same integrator
# ``ShootingHFSolver._two_electron_energy`` uses) and one-electron Dirac
# blocks with the solver's own nuclear potential array — so a single-CSF CI
# total on converged shooting orbitals reproduces ``ShootingResult
# .total_energy``.


class GridCIWorkspace:
    """Shooting-grid flavour of the CI radial workspace.

    Wraps the Salvat SGRID arrays so :func:`build_ci_matrix` /
    :func:`atomx.methods.ci.ci_solve` can consume orbitals tabulated on
    the shooting grid: ``orbitals_dict`` values are ``(P, Q)`` arrays of
    length ``len(r)`` (e.g. rows of ``ShootingResult._pa/_qa``), and the
    ``ns`` argument must equal ``len(r)``.

    Parameters
    ----------
    r : ndarray
        SGRID radial nodes (``r[0] == 0`` included — the SLAG6 kernels
        own the origin treatment).
    difr : ndarray
        Grid Jacobian ``dr/ds`` on the uniform-s index grid (``ds = 1``
        per index): integrals are ``slag6(1, f*difr)`` and derivatives
        are ``d/dr = (1/difr) d/ds``.
    rv_nuc : ndarray
        ``r·V_nuc(r)`` on the grid — the SAME array the shooting solver
        used (``ShootingHFSolver._nuclear_rv``), point or Fermi.  Kept
        in ``r·V`` form because ``V`` itself diverges at ``r = 0`` for
        a point nucleus.
    """

    __slots__ = ("r", "difr", "rv_nuc", "inv_r", "n")

    def __init__(self, r, difr, rv_nuc):
        r = np.asarray(r, dtype=float)
        difr = np.asarray(difr, dtype=float)
        rv_nuc = np.asarray(rv_nuc, dtype=float)
        if r.ndim != 1 or r.shape != difr.shape or r.shape != rv_nuc.shape:
            raise ValueError(
                f"GridCIWorkspace needs matching 1D r/difr/rv_nuc arrays "
                f"(got shapes {r.shape}, {difr.shape}, {rv_nuc.shape})")
        self.r = r
        self.difr = difr
        self.rv_nuc = rv_nuc
        self.n = r.shape[0]
        inv_r = np.zeros_like(r)
        mask = r > 1.0e-30
        inv_r[mask] = 1.0 / r[mask]
        self.inv_r = inv_r

    @classmethod
    def from_shooting(cls, hf_solver, result):
        """Workspace from a ShootingHFSolver + its ShootingResult.

        Rebuilds ``r·V_nuc`` through the solver's own
        ``_nuclear_rv`` so the CI one-electron blocks use the identical
        nuclear model (point/Fermi) the SCF solved with.
        """
        r = result._grid
        difr = result._difr
        return cls(r, difr, hf_solver._nuclear_rv(r, difr, len(r)))


# 8th-order first-derivative stencils on the uniform-s index grid.
# Central: antisymmetric weights for offsets ±1..±4; edge rows i=0..3 are
# the one-sided/offset 8th-order weights on nodes 0..8 (right edge uses the
# mirrored negated rows — first-derivative parity).
_FD_CENTRAL = (4.0 / 5.0, -1.0 / 5.0, 4.0 / 105.0, -1.0 / 280.0)


def _build_fd_edge_weights():
    W = np.empty((4, 9))
    nodes = np.arange(9, dtype=float)
    rhs = np.zeros(9)
    rhs[1] = 1.0  # Σ_j w_j x_j^m = m! δ_{m,1}
    for i in range(4):
        x = nodes - float(i)
        A = np.vander(x, 9, increasing=True).T
        W[i] = np.linalg.solve(A, rhs)
    return W


_FD_EDGE = _build_fd_edge_weights()


def _ds_derivative(y):
    """dy/ds on the uniform-s grid (unit index spacing), 8th order."""
    n = y.shape[0]
    if n < 9:
        raise ValueError(f"grid too small for 8th-order stencils (n={n})")
    c1, c2, c3, c4 = _FD_CENTRAL
    d = np.empty_like(y)
    d[4:n - 4] = (c1 * (y[5:n - 3] - y[3:n - 5])
                  + c2 * (y[6:n - 2] - y[2:n - 6])
                  + c3 * (y[7:n - 1] - y[1:n - 7])
                  + c4 * (y[8:n] - y[0:n - 8]))
    head = y[:9]
    tail = y[n - 9:][::-1]
    for i in range(4):
        d[i] = _FD_EDGE[i] @ head
        d[n - 1 - i] = -(_FD_EDGE[i] @ tail)
    return d


def _grid_ds(y):
    key = ("ds", id(y))
    v = _SLATER_MEMO.get(key)
    if v is None:
        v = _ds_derivative(y)
        _SLATER_MEMO[key] = (y, v)  # keep y alive: id-reuse safety
    else:
        v = v[1]
    return v


def _grid_pair_density(Pa, Qa, Pb, Qb):
    ia, ib = id(Pa), id(Pb)
    key = ("rho",) + ((ia, ib) if ia <= ib else (ib, ia))
    v = _SLATER_MEMO.get(key)
    if v is None:
        v = Pa * Pb + Qa * Qb
        _SLATER_MEMO[key] = v
    return v


def _grid_h_one_body(Pa, Qa, Pb, Qb, kappa, ws):
    """⟨a|h_D|b⟩ on the shooting grid (same-κ orbitals).

    Salvat convention (rest mass subtracted), matching the radial ODE the
    shooting engine integrates:

        h_D [P, Q] = [ V P + c(−Q' + κ Q/r),
                       c(P' + κ P/r) + (V − 2c²) Q ]

    The derivative terms integrate in s directly (``∫ f' g dr =
    ∫ (df/ds) g ds`` — the Jacobians cancel) and are symmetrised so the
    CI matrix stays exactly symmetric under a↔b; the local terms carry
    the ``difr`` Jacobian.  ``V`` enters as the workspace's ``rv_nuc/r``
    — never a rebuilt point-nucleus ``−Z/r``.
    """
    dPa, dQa = _grid_ds(Pa), _grid_ds(Qa)
    dPb, dQb = _grid_ds(Pb), _grid_ds(Qb)
    kin = 0.5 * SL * ((Qa * dPb + Qb * dPa) - (Pa * dQb + Pb * dQa))
    rho = _grid_pair_density(Pa, Qa, Pb, Qb)
    cross = Pa * Qb + Qa * Pb
    loc = ((SL * kappa * cross + ws.rv_nuc * rho) * ws.inv_r
           - 2.0 * SL * SL * (Qa * Qb))
    integrand = kin + loc * ws.difr
    return float(slag6(1.0, integrand, ws.n)[ws.n - 1])


def _h_one_body(Pa, Qa, Pb, Qb, kappa, Z, bsp_data, ns):
    """One-electron Dirac matrix element ⟨a|h_D|b⟩ — flavour dispatched."""
    if isinstance(bsp_data, GridCIWorkspace):
        ia, ib = id(Pa), id(Pb)
        key = ("h1", kappa) + ((ia, ib) if ia <= ib else (ib, ia))
        v = _SLATER_MEMO.get(key)
        if v is None:
            v = _grid_h_one_body(Pa, Qa, Pb, Qb, kappa, bsp_data)
            _SLATER_MEMO[key] = v
        return v
    H0, _S0 = _dirac_blocks(ns, kappa, Z, bsp_data)
    vec_a = np.concatenate([Pa, Qa])
    vec_b = np.concatenate([Pb, Qb])
    return vec_a @ H0 @ vec_b


def _norm_sq(P, Q, kappa, Z, bsp_data, ns):
    """⟨a|a⟩ — flavour dispatched (grid: SLAG6, bspline: overlap matrix)."""
    if isinstance(bsp_data, GridCIWorkspace):
        key = ("n2", id(P))
        v = _SLATER_MEMO.get(key)
        if v is None:
            rho = _grid_pair_density(P, Q, P, Q)
            v = float(slag6(1.0, rho * bsp_data.difr,
                            bsp_data.n)[bsp_data.n - 1])
            _SLATER_MEMO[key] = v
        return v
    _H0, S0 = _dirac_blocks(ns, kappa, Z, bsp_data)
    vec = np.concatenate([P, Q])
    return vec @ S0 @ vec
from atomx.numerics.bspline.evaluate import eval_orbital_on_grid, compute_yk
from atomx.angular.coefficients import exchange_lambda, exchange_k_range, reduced_cl_j2
from atomx.angular.wigner import wigner_6j
from atomx.angular.clebsch import clebsch_gordan
from atomx.angular.rcfp import ReducedSubshellState
from atomx.angular.csf_recoupling import make_chain_coupled_state, chain_projection_amplitudes, single_state_for_subshell
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices


def build_ci_matrix(csfs, orbitals_dict, orbital_basis, bsp_data, Z, ns,
                    nuclear_potential=None, convol_cache=None):
    """Build the CI Hamiltonian matrix H_{rs} in the CSF basis.

    Parameters
    ----------
    bsp_data : dict or GridCIWorkspace
        B-spline workspace (``orbitals_dict`` values are coefficient
        vectors), or a :class:`GridCIWorkspace` (values are ``(P, Q)``
        arrays on the shooting grid; ``ns`` must equal ``len(r)``).  The
        grid flavour evaluates every element with the shooting engine's
        own numerics (SLAG6 Slater integrals, workspace nuclear
        potential) — the same consistency principle as the bspline cell
        integrator below.
    nuclear_potential : callable or None
        ``V(r)`` for the one-electron Dirac blocks.  Pass the solver's
        potential (e.g. ``DFSolver._nuclear_potential``) so the CI
        one-electron energies use the SAME finite-nucleus model as the
        SCF that produced the orbitals; ``None`` = point nucleus.
        B-spline flavour only — the grid workspace carries its own
        ``rv_nuc`` array.
    convol_cache : ConvolCache or None
        Reuse an existing cell-integral rkb cache built on ``bsp_data``
        (e.g. ``DFSolver._convol_cache``) — the tables depend only on the
        B-spline geometry, so sharing them avoids an O(seconds) rebuild
        per CI construction.  A cache built on different ``bsp_data`` is
        ignored and a fresh one created.  B-spline flavour only.
    """
    _SLATER_MEMO.clear()
    if isinstance(bsp_data, GridCIWorkspace):
        if nuclear_potential is not None:
            raise ValueError(
                "grid-flavour CI takes its nuclear potential from "
                "GridCIWorkspace.rv_nuc; nuclear_potential= is "
                "bspline-only")
        if convol_cache is not None:
            raise ValueError("convol_cache= is bspline-only")
        if ns != bsp_data.n:
            raise ValueError(
                f"grid-flavour CI needs ns == len(r) "
                f"(got ns={ns}, grid size {bsp_data.n})")
    else:
        if convol_cache is not None and convol_cache.bsp_data is bsp_data:
            _SLATER_MEMO["_convol_cache"] = convol_cache
        _SLATER_MEMO["_nuclear_potential"] = nuclear_potential
    n_csf = len(csfs)
    H = np.zeros((n_csf, n_csf))
    for r in range(n_csf):
        for s in range(r, n_csf):
            h_rs = _ci_matrix_element(csfs[r], csfs[s], orbitals_dict,
                                       orbital_basis, bsp_data, Z, ns,
                                       is_diagonal=(r == s))
            H[r, s] = h_rs
            if r != s:
                H[s, r] = h_rs
    return H


def _ci_matrix_element(csf_r, csf_s, orbitals_dict, orbital_basis,
                        bsp_data, Z, ns, *, is_diagonal=False):
    n_diff = csf_r.n_electron_diff(csf_s)
    if n_diff == 0:
        # Subshell occupations match.  When ``is_diagonal`` we are on
        # the matrix diagonal (r == s) and apply the J-coupled
        # correction (Grant §8.3) so distinct (J, parity) CSFs receive
        # different physically meaningful diagonal energies.
        #
        # Two distinct CSFs with identical subshell lists but different
        # ``total_J2`` (parentage-only difference) reuse the
        # configuration-averaged kernel here; their cross-coupling is
        # then zeroed by the block filter in ``methods/ci.py`` when
        # ``(J, parity)`` differs.
        if is_diagonal:
            return _diagonal_with_J(csf_r, orbitals_dict, bsp_data, Z, ns)
        if (csf_r.total_J2 == csf_s.total_J2
                and csf_r.parity == csf_s.parity):
            # Same occupations AND same (J, parity): repeated-parentage
            # rows (e.g. 1s 2s 2p with (1s,2s) coupled to 0 or 1).  The
            # exact off-diagonal needs coefficients of fractional
            # parentage; returning the configuration-average energy here
            # (the old behaviour) made the duplicate block rank-1 —
            # [[E, E], [E, E]] has eigenvalues 2E and 0, which collapsed
            # a 3-open-shell Li-like solve 6.9 keV BELOW its own ground
            # state.  Orthogonal-parentage approximation: 0 keeps every
            # such state at the configuration-average energy and
            # preserves the multiplet centroid.
            return 0.0
        return _diagonal_element(csf_r, orbitals_dict, bsp_data, Z, ns)
    elif n_diff == 1:
        base = _one_diff_element(csf_r, csf_s, orbitals_dict, bsp_data, Z, ns)
        return base + _off_diagonal_J_correction(
            csf_r, csf_s, orbitals_dict, bsp_data, ns, n_diff=1,
        )
    elif n_diff == 2:
        base = _two_diff_element(csf_r, csf_s, orbitals_dict, bsp_data)
        return base + _off_diagonal_J_correction(
            csf_r, csf_s, orbitals_dict, bsp_data, ns, n_diff=2,
        )
    return 0.0


def _diagonal_with_J(csf, orbitals_dict, bsp_data, Z, ns):
    """Diagonal CI element with J-coupled angular correction.

    For CSFs with explicit ``total_J2`` and inferable intermediate
    coupling — specifically those with at most two open subshells where
    each open subshell carries a unique parentage (single electron or
    single hole) — adds a J-dependent correction on top of the
    configuration-averaged kernel so different (J, parity) members of
    the same configuration acquire physically meaningful splittings.

    Closed-shell or non-J-resolved CSFs are returned unchanged
    (correction is zero), preserving backwards compatibility.
    """
    base = _diagonal_element(csf, orbitals_dict, bsp_data, Z, ns)
    return base + _diagonal_J_correction(csf, orbitals_dict, bsp_data, ns)


def _get_orb(key, orbitals_dict, ns):
    """Get orbital (P, Q) or zeros if not found."""
    if key in orbitals_dict:
        return orbitals_dict[key]
    return np.zeros(ns), np.zeros(ns)


def _diagonal_element(csf, orbitals_dict, bsp_data, Z, ns):
    """Configuration-average diagonal CI element.

    Same-subshell two-electron terms must use q(q-1)/2, not q^2/2;
    otherwise the CI diagonal contains self-interaction for closed subshells.
    The coefficients match DFSolver._compute_total_energy so MCDHF CI
    diagonals and the SCF energy functional stay consistent.
    """
    E_1e = 0.0
    for sh in csf.subshells:
        P, Q = _get_orb((sh.n, sh.kappa), orbitals_dict, ns)
        norm_sq = _norm_sq(P, Q, sh.kappa, Z, bsp_data, ns)
        if norm_sq > 1e-20:
            E_1e += (sh.occupation
                     * _h_one_body(P, Q, P, Q, sh.kappa, Z, bsp_data, ns)
                     / norm_sq)

    E_J, E_K = 0.0, 0.0
    subshells = list(csf.subshells)
    for i, sh_a in enumerate(subshells):
        Pa, Qa = _get_orb((sh_a.n, sh_a.kappa), orbitals_dict, ns)
        tj_a = 2 * abs(sh_a.kappa) - 1
        l_a = sh_a.kappa if sh_a.kappa > 0 else -sh_a.kappa - 1
        for j, sh_b in enumerate(subshells[: i + 1]):
            Pb, Qb = _get_orb((sh_b.n, sh_b.kappa), orbitals_dict, ns)
            tj_b = 2 * abs(sh_b.kappa) - 1

            if i == j:
                coeff = sh_a.occupation * (sh_a.occupation - 1.0) / 2.0
                if abs(coeff) < 1e-15:
                    continue
                f0 = _fk(Pa, Qa, Pa, Qa, bsp_data, 0)
                E_J += coeff * f0
                for k in range(1, tj_a + 1):
                    if (k + 2 * l_a) % 2 != 0:
                        continue
                    lam = exchange_lambda(sh_a.kappa, sh_a.kappa, k)
                    if lam < 1e-15:
                        continue
                    fk = _fk(Pa, Qa, Pa, Qa, bsp_data, k)
                    E_K += coeff * lam / (tj_a * (tj_a + 1.0)) * fk
                continue

            coeff = sh_a.occupation * sh_b.occupation
            f0 = _fk(Pa, Qa, Pb, Qb, bsp_data, 0)
            E_J += coeff * f0
            for k in exchange_k_range(sh_a.kappa, sh_b.kappa):
                lam = exchange_lambda(sh_a.kappa, sh_b.kappa, k)
                if lam < 1e-15:
                    continue
                gk = _gk(Pa, Qa, Pb, Qb, bsp_data, k)
                E_K += coeff * lam / ((tj_a + 1.0) * (tj_b + 1.0)) * gk

    return E_1e + E_J - E_K


def _one_diff_element(csf_r, csf_s, orbitals_dict, bsp_data, Z, ns):
    """1-electron promotion (b → a) — ``csf_s`` (ket) → ``csf_r`` (bra).

    Variable convention:

    * ``a`` (= ``lost[0]``: bra > ket) is the **destination** orbital where an
      electron arrived in the bra.
    * ``b`` (= ``gained[0]``: bra < ket) is the **source** orbital that lost
      one electron going from ket to bra.

    The configuration-averaged jj-coupled Slater-Condon expression is

        H_{rs} = √(q_b^{ket} (g_a − q_a^{ket})) ·
                 [⟨a|h|b⟩ δ_{κ_a,κ_b}
                  + Σ_{c ∈ common} q_c (X^direct − X^exchange)]

    where ``q_b^{ket}`` is the source occupation in the ket and
    ``g_a − q_a^{ket}`` are the destination vacancies in the ket.  The
    direct integral ``R^0(ac, bc)`` is non-zero only when ``κ_a = κ_b``
    (angular monopole orthogonality); for ``κ_a ≠ κ_b`` the only
    contribution comes from the exchange multipoles ``k`` satisfying
    triangle conditions on both ``(j_a, j_c)`` and ``(j_c, j_b)``.

    For different-κ promotions the exchange angular factor is the
    geometric mean ``√(λ^k(a,c) λ^k(c,b))`` (which reduces to ``λ^k(a,c)``
    when ``κ_a = κ_b``).
    """
    diffs = csf_r.differs_from(csf_s)
    lost = [(key, qs, qo) for key, qs, qo in diffs if qs > qo]
    gained = [(key, qs, qo) for key, qs, qo in diffs if qs < qo]

    if len(lost) != 1 or len(gained) != 1:
        return 0.0

    key_a, qa_r, qa_s = lost[0]    # destination (bra > ket)
    key_b, qb_r, qb_s = gained[0]  # source (bra < ket)
    Pa, Qa = _get_orb(key_a, orbitals_dict, ns)
    Pb, Qb = _get_orb(key_b, orbitals_dict, ns)
    kappa_a, kappa_b = key_a[1], key_b[1]

    # jj-coupling 1-diff occupation weight: q_source^{ket} · h_dest^{ket}.
    # ``qb_s`` = ket occupation of source subshell ``b`` (must be > 0).
    # ``g_a − qa_s`` = ket vacancies in destination subshell ``a`` (> 0).
    g_a = 2 * abs(kappa_a)
    sqrt_factor = np.sqrt(qb_s * (g_a - qa_s))

    # One-electron: ⟨a|h|b⟩ (non-zero only for same κ; h is a scalar tensor)
    H_1e = 0.0
    if kappa_a == kappa_b:
        H_1e = _h_one_body(Pa, Qa, Pb, Qb, kappa_a, Z, bsp_data, ns)

    # Two-electron: Σ_c q_c [direct − exchange] over common spectator orbitals
    H_2e = 0.0
    occ_r = {(sh.n, sh.kappa): sh.occupation for sh in csf_r.subshells}
    occ_s = {(sh.n, sh.kappa): sh.occupation for sh in csf_s.subshells}
    seen_c = set()
    for sh_c in csf_r.subshells:
        key_c = (sh_c.n, sh_c.kappa)
        if key_c in seen_c or key_c in (key_a, key_b):
            continue
        seen_c.add(key_c)
        q_common = min(occ_r.get(key_c, 0), occ_s.get(key_c, 0))
        if q_common <= 0:
            continue

        Pc, Qc = _get_orb(key_c, orbitals_dict, ns)
        kappa_c = key_c[1]
        tj_c = 2 * abs(kappa_c) - 1

        # Direct R^0(ac, bc) — angular factor at k=0 is δ_{κ_a, κ_b}.
        # Skip entirely for different-κ promotions to avoid the spurious
        # ⟨ψ_3s|·|ψ_3d⟩-type cross-coupling that was inflating the
        # off-diagonal CI matrix for Fe XVII 2p⁵3l J=1 states.
        if kappa_a == kappa_b:
            rk_direct = _rk(Pa, Qa, Pc, Qc, Pb, Qb, Pc, Qc, bsp_data, 0)
            H_2e += q_common * rk_direct

        # Exchange: Σ_k c_k R^k(ac, cb) with k allowed on BOTH (a,c) and (c,b).
        k_range_ab = sorted(set(exchange_k_range(kappa_a, kappa_c))
                            & set(exchange_k_range(kappa_c, kappa_b)))
        for k in k_range_ab:
            lam_ac = exchange_lambda(kappa_a, kappa_c, k)
            lam_cb = exchange_lambda(kappa_c, kappa_b, k)
            if lam_ac < 1e-15 or lam_cb < 1e-15:
                continue
            # A closed spherical core contributes a scalar mean field. It
            # cannot mix different κ orbitals in a one-electron promotion;
            # FAC's recoupled Hamiltonian has the same selection behavior.
            # Leaving the configuration-average geometric exchange term here
            # over-couples Fe XVII 2p-hole/3d J=1 CSFs by O(100 eV).
            cap_c = 2 * abs(kappa_c)
            if kappa_a != kappa_b and int(round(q_common)) == cap_c:
                continue
            rk_exch = _rk(Pa, Qa, Pc, Qc, Pc, Qc, Pb, Qb,
                                         bsp_data, k)
            # Angular factor: √(λ^k(a,c) λ^k(c,b)) reduces to λ^k(a,c)
            # when κ_a = κ_b (the standard same-orbital HF exchange).
            ang_k = lam_ac if kappa_a == kappa_b else np.sqrt(lam_ac * lam_cb)
            c_k = q_common / (tj_c + 1.0) * ang_k
            H_2e -= c_k * rk_exch

    return sqrt_factor * (H_1e + H_2e)


def _two_diff_element(csf_r, csf_s, orbitals_dict, bsp_data):
    """2-electron promotion (c → a, d → b).

    ``lost`` (bra > ket): orbitals where the bra gained electrons → destinations.
    ``gained`` (bra < ket): orbitals where the bra lost electrons → sources.

    The configuration-averaged two-body Slater–Condon expression is

        H_{rs} = δ_{κ_a,κ_c} δ_{κ_b,κ_d} R^0(ac, bd)
                 − Σ_k λ̂^k_{ad,cb} R^k(ac, db)

    where the direct ``k=0`` integral is non-zero only when the two
    "left-orbital" pairs are angularly compatible (same κ), and the
    exchange ``k`` must satisfy the triangle conditions on both
    ``(κ_a, κ_d)`` and ``(κ_c, κ_b)`` simultaneously.  For different-κ
    exchange pairs the magnitude is the geometric mean
    ``√(λ^k(a,d) λ^k(c,b))`` (reduces to ``λ^k(a,d)`` in the same-κ
    case).
    """
    diffs = csf_r.differs_from(csf_s)
    lost = [(key, qs, qo) for key, qs, qo in diffs if qs > qo]
    gained = [(key, qs, qo) for key, qs, qo in diffs if qs < qo]

    # Same-shell pair excitation a² → b² (single source AND single
    # destination key, Δq = 2) falls through to 0 here.  The J=0 pairing
    # element is well-defined —
    #   ⟨(j_b)² 0|1/r₁₂|(j_a)² 0⟩ = Σ_k (−1)^k ⟨j_b‖C^k‖j_a⟩²/√(g_a g_b) G^k(a,b)
    # (≈ +0.04 Ha for Pb (6p₁/₂)² ↔ (6p₃/₂)², ~5 % ground admixture) —
    # but wiring it in is deferred: feeding the resulting pair-correlation
    # weight back into the Guerra c²-driven occupations is bistable (the
    # admixture deoptimises the very orbital that sustains it), sending
    # the Pb outer loop into a period-~6 limit cycle instead of the
    # 4-iteration convergence without it.  Enabling this channel needs
    # fixed-weight (statistical/equal) radial optimisation for pair
    # channels or a simultaneous orbital+coefficient Newton — see the
    # MCDHF solver notes.
    if len(lost) != 2 or len(gained) != 2:
        return 0.0

    ns = len(list(orbitals_dict.values())[0][0]) if orbitals_dict else 50

    key_a, _, _ = lost[0]    # destination 1
    key_b, _, _ = lost[1]    # destination 2
    key_c, _, _ = gained[0]  # source 1
    key_d, _, _ = gained[1]  # source 2

    Pa, Qa = _get_orb(key_a, orbitals_dict, ns)
    Pb, Qb = _get_orb(key_b, orbitals_dict, ns)
    Pc, Qc = _get_orb(key_c, orbitals_dict, ns)
    Pd, Qd = _get_orb(key_d, orbitals_dict, ns)

    kappa_a, kappa_b = key_a[1], key_b[1]
    kappa_c, kappa_d = key_c[1], key_d[1]

    H_2e = 0.0

    # Direct ⟨ac|g|bd⟩^{k=0}: angular factor at k=0 requires both
    # ⟨ψ_a|·|ψ_b⟩_{k=0} ∝ δ_{κ_a, κ_b} and ⟨ψ_c|·|ψ_d⟩_{k=0} ∝ δ_{κ_c, κ_d}
    # to be simultaneously non-zero.  In the (a→c, b→d) two-body
    # promotion this means κ_a = κ_c AND κ_b = κ_d.
    if kappa_a == kappa_c and kappa_b == kappa_d:
        rk_direct = _rk(Pa, Qa, Pc, Qc, Pb, Qb, Pd, Qd, bsp_data, 0)
        H_2e += rk_direct

    # Exchange ⟨ac|g|db⟩.  We keep the original (κ_a, κ_d)-based form for
    # backwards compatibility (it correctly handles the same-κ case used by
    # existing benchmarks); for different-κ exchange pairs the proper
    # angular factor is a geometric mean √(λ_ad · λ_cb), but the off-diagonal
    # phase requires a full 6j treatment that is deferred to a follow-up.
    for k in exchange_k_range(kappa_a, kappa_d):
        lam = exchange_lambda(kappa_a, kappa_d, k)
        if lam < 1e-15:
            continue
        rk_exch = _rk(Pa, Qa, Pc, Qc, Pd, Qd, Pb, Qb,
                                     bsp_data, k)
        tj_a = 2 * abs(kappa_a) - 1
        tj_d = 2 * abs(kappa_d) - 1
        c_k = lam / ((tj_a + 1.0) * (tj_d + 1.0))
        H_2e -= c_k * rk_exch

    return H_2e


def _diagonal_J_correction(csf, orbitals_dict, bsp_data, ns):
    """J-dependent diagonal correction for two-open-subshell CSFs.

    The configuration-averaged exchange coefficient between two open
    subshells (a, b) in :func:`_diagonal_element` is
    ``q_a q_b /(g_a g_b) * λ^K``.  The proper Grant §8.3 J-coupled
    coefficient for two electrons (or, by particle-hole symmetry, two
    holes / one of each) with unique parentage ``J_a = j_a`` and
    ``J_b = j_b`` coupled to total ``J`` is

        g^K(J) = (-1)^{j_a + j_b + J + K}
                 * λ^K_{ab} * {j_a j_b J; j_b j_a K}

    where ``λ^K_{ab} = (2j_a+1)(2j_b+1) (j_a K j_b; -1/2 0 1/2)^2``.
    The correction returned here is the difference times the Slater
    integral ``G^K(a,b)``, summed over allowed multipoles ``K``, with
    a sign that matches the existing kernel's ``-E_K`` convention.

    Particle-hole conjugation changes the sign of the J-coupled pair
    coefficient for one-hole/one-electron pairs.  The average coefficient
    being replaced must still be the coefficient actually subtracted by
    :func:`_diagonal_element`, namely ``q_a q_b λ^K/(g_a g_b)``.

    Returns ``0.0`` for closed-shell-only CSFs (no open subshells),
    for CSFs without explicit ``total_J2`` metadata, and for CSFs with
    more than two open subshells (the general case requires a full
    parentage chain that is not inferable from a bare ``CSF`` object
    today — those continue to use the configuration-averaged kernel
    until parentage labels are threaded through the CSF generator).
    """
    if not _has_explicit_J(csf):
        return 0.0

    open_subs = _open_subshells(csf)
    if len(open_subs) != 2:
        return 0.0

    sh_a, sh_b = open_subs
    qa = int(round(sh_a.occupation))
    qb = int(round(sh_b.occupation))
    cap_a = 2 * abs(sh_a.kappa)   # 2j_a + 1 (subshell capacity)
    cap_b = 2 * abs(sh_b.kappa)
    tj_a = cap_a - 1              # 2 j_a
    tj_b = cap_b - 1              # 2 j_b

    shell_totals: dict[tuple[int, int], tuple[int, int]] = {}
    for sh in csf.subshells:
        l = int(sh.kappa) if sh.kappa > 0 else -int(sh.kappa) - 1
        key = (int(sh.n), l)
        total_q, total_cap = shell_totals.get(key, (0, 4 * l + 2))
        shell_totals[key] = (total_q + int(round(sh.occupation)), total_cap)

    def _electron_hole_flags(sh, q, cap):
        is_elec = q == 1
        is_hole = q == cap - 1
        if is_elec and is_hole:
            # j=1/2 with q=1 is both a one-electron and one-hole subshell.
            # Resolve it from the parent nonrelativistic nl occupancy: in a
            # more-than-half-filled shell such as Fe XVII 2p^5, p1/2^1 is a
            # core hole; in 1s^1 or 2p^1 it remains a valence electron.
            l = int(sh.kappa) if sh.kappa > 0 else -int(sh.kappa) - 1
            total_q, total_cap = shell_totals.get((int(sh.n), l), (q, cap))
            is_hole = total_q > total_cap / 2.0
            is_elec = not is_hole
        return is_elec, is_hole

    # Each open subshell must be a single electron (q=1) or a single
    # hole (q=2j) so that its parentage J = j_subshell is unique.
    is_elec_a, is_hole_a = _electron_hole_flags(sh_a, qa, cap_a)
    is_elec_b, is_hole_b = _electron_hole_flags(sh_b, qb, cap_b)
    if not ((is_elec_a or is_hole_a) and (is_elec_b or is_hole_b)):
        return 0.0

    Pa, Qa_orb = _get_orb((sh_a.n, sh_a.kappa), orbitals_dict, ns)
    Pb, Qb_orb = _get_orb((sh_b.n, sh_b.kappa), orbitals_dict, ns)

    total_J2 = int(csf.total_J2)
    delta = 0.0
    for k in exchange_k_range(sh_a.kappa, sh_b.kappa):
        lam = exchange_lambda(sh_a.kappa, sh_b.kappa, k)
        if lam < 1e-15:
            continue
        sixj = wigner_6j(tj_a, tj_b, total_J2, tj_b, tj_a, 2 * k)
        if abs(sixj) < 1e-15:
            g_K_J = 0.0
        else:
            exp_int = (tj_a + tj_b + total_J2) // 2 + k
            phase = -1.0 if exp_int % 2 else 1.0
            g_K_J = phase * sixj * lam
        if is_hole_a ^ is_hole_b:
            g_K_J = -g_K_J

        avg_coef = qa * qb * lam / (cap_a * cap_b)

        # Final energy uses (- exchange).  The existing kernel
        # subtracts ``avg_coef * G^K`` from the diagonal; the proper
        # J-coupled subtraction is ``g_K_J * G^K``.  Net correction to
        # ADD to the existing diagonal energy is therefore
        # ``(avg_coef - g_K_J) * G^K``.
        gk_val = _gk(Pa, Qa_orb, Pb, Qb_orb, bsp_data, k)
        delta += (avg_coef - g_K_J) * gk_val

    return float(delta)


def _aligned_unique_chain_from_occ(keys, occ, total_J2, parity):
    subshells = []
    for _n, kappa in keys:
        j2 = 2 * abs(int(kappa)) - 1
        q = int(round(occ.get((_n, kappa), 0)))
        state = single_state_for_subshell(j2, q)
        if state is None:
            return None
        subshells.append(state)

    cumulative = []
    current = 0
    nonzero = 0
    for state in subshells:
        if not cumulative:
            current = state.J2
        elif state.J2 == 0:
            pass
        elif current == 0:
            current = state.J2
            nonzero += 1
        else:
            nonzero += 1
            if nonzero > 2:
                return None
            current = int(total_J2)
        if state.J2 != 0 and len(cumulative) == 0:
            nonzero = 1
        cumulative.append(current)

    if current != int(total_J2):
        return None
    try:
        return make_chain_coupled_state(
            tuple(subshells), tuple(cumulative), parity=int(parity),
        )
    except ValueError:
        return None


def _simple_subshell_projection(state: ReducedSubshellState, M2: int):
    j2, q = int(state.j2), int(state.q)
    cap = j2 + 1
    m_values = list(range(-j2, j2 + 1, 2))
    if q == 0:
        return {0: 1.0} if M2 == 0 else {}
    if q == cap:
        return {(1 << cap) - 1: 1.0} if M2 == 0 else {}
    if q == 1:
        return {1 << m_values.index(M2): 1.0} if M2 in m_values else {}
    if q == cap - 1:
        hole_m = -M2
        if hole_m not in m_values:
            return {}
        return {((1 << cap) - 1) ^ (1 << m_values.index(hole_m)): 1.0}
    return {}


def _chain_global_projection(chain):
    offsets = []
    total = 0
    for state in chain.subshells:
        offsets.append(total)
        total += int(state.j2) + 1

    out = {}
    for projections, chain_coeff in chain_projection_amplitudes(chain, chain.total_J2):
        partial = {0: float(chain_coeff)}
        for idx, (state, M2) in enumerate(zip(chain.subshells, projections)):
            next_partial = {}
            for mask, coeff in partial.items():
                for local_mask, local_coeff in _simple_subshell_projection(state, M2).items():
                    global_mask = mask | (local_mask << offsets[idx])
                    next_partial[global_mask] = next_partial.get(global_mask, 0.0) + coeff * local_coeff
            partial = next_partial
        for mask, coeff in partial.items():
            if abs(coeff) > 1.0e-14:
                out[mask] = out.get(mask, 0.0) + coeff
    return out, offsets


def _annihilate_global(mask, orbital):
    if not (mask & (1 << orbital)):
        return None
    phase = -1.0 if (mask & ((1 << orbital) - 1)).bit_count() % 2 else 1.0
    return phase, mask ^ (1 << orbital)


def _create_global(mask, orbital):
    if mask & (1 << orbital):
        return None
    phase = -1.0 if (mask & ((1 << orbital) - 1)).bit_count() % 2 else 1.0
    return phase, mask | (1 << orbital)


def _branch9_one_body_apply(mask, chain, offsets, source, dest, K2, Q2):
    out = {}
    source_j2 = int(chain.subshells[source].j2)
    dest_j2 = int(chain.subshells[dest].j2)
    source_m = list(range(-source_j2, source_j2 + 1, 2))
    dest_m = list(range(-dest_j2, dest_j2 + 1, 2))
    for m_create in dest_m:
        mu = Q2 - m_create
        if mu not in source_m:
            continue
        cg = clebsch_gordan(dest_j2, m_create, source_j2, mu, K2, Q2)
        if abs(cg) < 1.0e-14:
            continue
        remove_m = -mu
        ann = _annihilate_global(mask, offsets[source] + source_m.index(remove_m))
        if ann is None:
            continue
        phase_a, intermediate = ann
        cre = _create_global(intermediate, offsets[dest] + dest_m.index(m_create))
        if cre is None:
            continue
        phase_c, final = cre
        tilde_phase = -1.0 if ((source_j2 - mu) // 2) % 2 else 1.0
        out[final] = out.get(final, 0.0) + phase_a * phase_c * tilde_phase * cg
    return out


def _branch9_double_excitation_scalar(csf_r, csf_s, source1, dest1, source2, dest2, K2):
    occ_r = {(int(sh.n), int(sh.kappa)): int(round(sh.occupation)) for sh in csf_r.subshells}
    occ_s = {(int(sh.n), int(sh.kappa)): int(round(sh.occupation)) for sh in csf_s.subshells}
    keys = tuple(sorted(
        set(occ_r) | set(occ_s) | {tuple(source1), tuple(dest1), tuple(source2), tuple(dest2)},
        key=lambda key: (key[0], abs(key[1]), key[1]),
    ))
    index = {key: i for i, key in enumerate(keys)}
    bra = _aligned_unique_chain_from_occ(keys, occ_r, int(csf_r.total_J2), int(getattr(csf_r, "parity", 1)))
    ket = _aligned_unique_chain_from_occ(keys, occ_s, int(csf_s.total_J2), int(getattr(csf_s, "parity", 1)))
    if bra is None or ket is None:
        return None

    bra_vec, offsets = _chain_global_projection(bra)
    ket_vec, _ = _chain_global_projection(ket)
    s1, d1 = index[tuple(source1)], index[tuple(dest1)]
    s2, d2 = index[tuple(source2)], index[tuple(dest2)]

    total = 0.0
    for ket_mask, ket_coeff in ket_vec.items():
        for q in range(-K2, K2 + 1, 2):
            scalar_cg = clebsch_gordan(K2, q, K2, -q, 0, 0)
            if abs(scalar_cg) < 1.0e-14:
                continue
            mid_terms = _branch9_one_body_apply(ket_mask, ket, offsets, s2, d2, K2, -q)
            for mid_mask, coeff2 in mid_terms.items():
                final_terms = _branch9_one_body_apply(mid_mask, ket, offsets, s1, d1, K2, q)
                for final_mask, coeff1 in final_terms.items():
                    total += bra_vec.get(final_mask, 0.0) * scalar_cg * coeff1 * coeff2 * ket_coeff
    return float(total)


def _off_diagonal_two_diff_J_correction(csf_r, csf_s, orbitals_dict, bsp_data, ns):
    """J-coupled replacement correction for two-diff unique-parentage CSFs."""
    diffs = csf_r.differs_from(csf_s)
    lost = [(key, qr, qs) for key, qr, qs in diffs if qr > qs]
    gained = [(key, qr, qs) for key, qr, qs in diffs if qr < qs]
    if len(lost) != 2 or len(gained) != 2:
        return 0.0

    key_a, _, _ = lost[0]
    key_b, _, _ = lost[1]
    key_c, _, _ = gained[0]
    key_d, _, _ = gained[1]

    # Conservative scope: both endpoint CSFs carry only unique-parentage open
    # subshells.  Do not require the raw source occupation in the opposite
    # endpoint to be open: a p1/2 closed shell in the ket can be the source of
    # a single p1/2 hole in the bra, and that particle-hole CSF still has
    # unique parentage.  Requiring the closed source itself to be "unique"
    # incorrectly skips the J-coupled replacement for Fe XVII p1/2-hole
    # two-diff couplings and leaves the configuration-average term behind.
    occ_r = {(int(sh.n), int(sh.kappa)): int(round(sh.occupation)) for sh in csf_r.subshells}
    occ_s = {(int(sh.n), int(sh.kappa)): int(round(sh.occupation)) for sh in csf_s.subshells}
    for csf, occ in ((csf_r, occ_r), (csf_s, occ_s)):
        open_keys = []
        for sh in csf.subshells:
            key = (int(sh.n), int(sh.kappa))
            q = occ[key]
            cap = 2 * abs(int(sh.kappa))
            if 0 < q < cap:
                if q not in (1, cap - 1):
                    return 0.0
                open_keys.append(key)
        if len(open_keys) != 2:
            return 0.0

    total_J2 = int(csf_r.total_J2)
    tj_a = 2 * abs(int(key_a[1])) - 1
    tj_b = 2 * abs(int(key_b[1])) - 1
    tj_c = 2 * abs(int(key_c[1])) - 1
    tj_d = 2 * abs(int(key_d[1])) - 1
    Pa, Qa = _get_orb(key_a, orbitals_dict, ns)
    Pb, Qb = _get_orb(key_b, orbitals_dict, ns)
    Pc, Qc = _get_orb(key_c, orbitals_dict, ns)
    Pd, Qd = _get_orb(key_d, orbitals_dict, ns)

    def _nl_totals(csf):
        totals: dict[tuple[int, int], tuple[int, int]] = {}
        for sh in csf.subshells:
            l = int(sh.kappa) if sh.kappa > 0 else -int(sh.kappa) - 1
            key = (int(sh.n), l)
            q0, cap0 = totals.get(key, (0, 4 * l + 2))
            totals[key] = (q0 + int(round(sh.occupation)), cap0)
        return totals

    occ_r_all = {(int(sh.n), int(sh.kappa)): int(round(sh.occupation)) for sh in csf_r.subshells}
    occ_s_all = {(int(sh.n), int(sh.kappa)): int(round(sh.occupation)) for sh in csf_s.subshells}
    nl_r = _nl_totals(csf_r)
    nl_s = _nl_totals(csf_s)

    def _is_ambiguous_core_hole(key, *, side):
        n, kappa = int(key[0]), int(key[1])
        cap = 2 * abs(kappa)
        if cap != 2:
            return False
        q = occ_r_all.get((n, kappa), 0) if side == "bra" else occ_s_all.get((n, kappa), 0)
        if q != 1:
            return False
        l = kappa if kappa > 0 else -kappa - 1
        total_q, total_cap = (nl_r if side == "bra" else nl_s).get((n, l), (q, cap))
        return total_q > total_cap / 2.0

    # FAC's all-distinct AngularZxZ0 branch carries an extra particle-hole
    # phase for ambiguous j=1/2 holes.  The determinant projection alone cannot
    # distinguish a p1/2 electron from a p1/2 hole because both have the same
    # one-particle occupancy mask.  Use the compact FAC-phase coefficient for
    # those core-hole transfers; keep the determinant branch for ordinary
    # all-distinct electron-electron cases.
    def _is_s_p12_hole_transfer(dest_key, source_key):
        dest = (int(dest_key[0]), int(dest_key[1]))
        source = (int(source_key[0]), int(source_key[1]))
        if dest[0] != source[0]:
            return False
        if {dest[1], source[1]} != {-1, 1}:
            return False
        # The p1/2 member is the ambiguous cap=2 hole.  Restrict the compact
        # FAC phase to s1/2 <-> p1/2 hole transfer; p1/2 <-> p3/2 hole
        # couplings are already handled correctly by the determinant branch.
        p12 = dest if dest[1] == 1 else source
        return _is_ambiguous_core_hole(p12, side="bra") or _is_ambiguous_core_hole(p12, side="ket")

    ambiguous_hole_transfer = (
        _is_s_p12_hole_transfer(key_a, key_c)
        or _is_s_p12_hole_transfer(key_b, key_d)
    )

    proper = 0.0
    all_distinct = len({tuple(key_a), tuple(key_b), tuple(key_c), tuple(key_d)}) == 4
    # Direct-coupled tensor product <a||C^K||c><b||C^K||d> R^K(ac,bd).
    direct_ks = sorted(set(exchange_k_range(key_a[1], key_c[1]))
                       & set(exchange_k_range(key_b[1], key_d[1])))
    for k in direct_ks:
        lam_ac = exchange_lambda(key_a[1], key_c[1], k)
        lam_bd = exchange_lambda(key_b[1], key_d[1], k)
        if lam_ac < 1.0e-15 or lam_bd < 1.0e-15:
            continue
        if all_distinct and not ambiguous_hole_transfer:
            coeff = _branch9_double_excitation_scalar(
                csf_r, csf_s, key_c, key_a, key_d, key_b, 2 * k,
            )
            if coeff is None or abs(coeff) < 1.0e-15:
                continue
        else:
            sixj = wigner_6j(tj_a, tj_b, total_J2, tj_d, tj_c, 2 * k)
            if abs(sixj) < 1.0e-15:
                continue
            phase_exp = (tj_a + tj_b + tj_c + tj_d + total_J2) // 2 + k
            if all_distinct and ambiguous_hole_transfer:
                phase_exp += 1
            phase = -1.0 if phase_exp % 2 else 1.0
            coeff = phase * sixj * float(np.sqrt(lam_ac * lam_bd))
        if bsp_data:
            radial = _rk(Pa, Qa, Pc, Qc, Pb, Qb, Pd, Qd, bsp_data, k)
        else:
            radial = 1.0
        proper += coeff * radial

    # Antisymmetrized exchange partner <a||C^K||d><b||C^K||c> R^K(ad,bc).
    exchange_ks = sorted(set(exchange_k_range(key_a[1], key_d[1]))
                         & set(exchange_k_range(key_b[1], key_c[1])))
    for k in exchange_ks:
        lam_ad = exchange_lambda(key_a[1], key_d[1], k)
        lam_bc = exchange_lambda(key_b[1], key_c[1], k)
        if lam_ad < 1.0e-15 or lam_bc < 1.0e-15:
            continue
        sixj = wigner_6j(tj_a, tj_b, total_J2, tj_c, tj_d, 2 * k)
        if abs(sixj) < 1.0e-15:
            continue
        phase_exp = (tj_a + tj_b + tj_c + tj_d + total_J2) // 2 + k + 1
        phase = -1.0 if phase_exp % 2 else 1.0
        coeff = phase * sixj * float(np.sqrt(lam_ad * lam_bc))
        if bsp_data:
            radial = _rk(Pa, Qa, Pd, Qd, Pb, Qb, Pc, Qc, bsp_data, k)
        else:
            radial = 1.0
        proper += coeff * radial

    base = _two_diff_element(csf_r, csf_s, orbitals_dict, bsp_data) if bsp_data else 0.0
    return float(proper - base)


def _has_explicit_J(csf):
    """Whether a CSF carries explicit J-resolution metadata.

    Accepts a non-default ``total_J2`` attribute *or* a label containing
    ``J=`` in either the legacy form ``... J=2/2`` or the bracketed form
    ``... [J=2/2]`` that :func:`enumerate_jj_csfs` emits.  The bracketed
    form is required for closed-shell J=0 CSFs whose ``total_J2`` is
    zero — without it those silently bypass the J-coupled correction.
    """
    if int(getattr(csf, "total_J2", 0)) > 0:
        return True
    label = str(getattr(csf, "label", ""))
    return " J=" in label or "[J=" in label


def _open_subshells(csf):
    """Return open subshells (0 < occupation < capacity) of a CSF."""
    open_subs = []
    for sh in csf.subshells:
        cap = 2 * abs(sh.kappa)
        q = int(round(sh.occupation))
        if 0 < q < cap:
            open_subs.append(sh)
    return open_subs


def _off_diagonal_J_correction(csf_r, csf_s, orbitals_dict, bsp_data, ns,
                                 *, n_diff):
    """J-coupled off-diagonal CI correction (Grant §8.3) — scaffold.

    HYP-184 follow-up.  The configuration-averaged 1-diff/2-diff
    matrix elements computed by :func:`_one_diff_element` and
    :func:`_two_diff_element` use the orbital-occupation-weighted
    Slater-Condon expression appropriate for single-open-subshell
    CSFs.  For two CSFs sharing the *same* (J, parity) block where
    each CSF has two-or-more open subshells with **unique parentage**
    (every open subshell is a single electron ``q=1`` or single hole
    ``q=2j``), the proper Grant §8.3 6j-style J-coupled exchange
    coefficient differs from the average-of-shell weight and must be
    added as an off-diagonal correction analogous to
    :func:`_diagonal_J_correction`.

    The two relevant cases for the Fe XVII 2p⁵3l J=1⁻ block are:

    * **1-diff**: bra and ket share one open subshell ``c`` (the
      spectator) and differ in a single-electron promotion ``a → b``
      (e.g. ``2p_{1/2}^{1} 3d_{3/2}^{1} ↔ 2p_{3/2}^{-1} 3d_{3/2}^{1}``).
      The proper correction is

          ΔH = Σ_K v^K(j_a, j_b, j_c; J) · R^K(ac, cb)

      where ``v^K`` is the J-coupled exchange coefficient with a Wigner
      6j weight ``{j_a j_b J; j_c j_c K}`` and the standard
      ``(j_a, j_b)`` reduced matrix element.  The difference between
      this and the geometric-mean configuration-averaged factor
      ``√(λ^K(a,c) λ^K(c,b))`` used in
      :func:`_one_diff_element` is what spreads the three J=1⁻
      3d-electron CSFs into the proper ¹P₁ / ³D₁ / ³P₁ intensity
      pattern that Bernitt 2012 sees.

    * **2-diff**: bra and ket differ in two open subshells (e.g.
      ``2p_{1/2}^{1} 3d_{3/2}^{1} ↔ 2p_{3/2}^{-1} 3d_{5/2}^{1}``).
      The J-coupled formula involves a single 6j coupling on the
      (j_a, j_b) → (j_c, j_d) → J chain analogous to the 1-diff
      case.

    Implementation notes (Grant §8.3 / Cowan §12):

      g^K_offdiag(J) = (-1)^{j_a + j_b + j_c + j_d + J + K}
                       * {6j(j_a j_b J; j_d j_c K)}
                       * <j_a||C^K||j_b> <j_c||C^K||j_d>

    The reduced matrix elements ``<j||C^K||j'>`` are the standard
    angular tensor coefficients available via
    :func:`atomx.angular.coefficients.exchange_lambda` (square root,
    with sign managed by the 6j phase).

    **Current status — scaffold only.**  The off-diagonal correction
    is *not* applied yet (returns ``0.0``).  A correct implementation
    requires:

    1. Identifying the unique-parentage subset (each open subshell is
       a single electron or single hole) on **both** ``csf_r`` and
       ``csf_s`` and the matching same-(J,parity) constraint.
    2. Selecting the spectator/source/destination shells and reading
       off the ``(j_a, j_b, j_c, j_d)`` labels and total ``J``.
    3. Evaluating the Slater integrals ``R^K`` over the radial part
       using the module-level ``_rk`` (cell-algorithm integrator).
    4. Composing the 6j-weighted angular factor and accumulating the
       additive correction.
    5. Verifying against a He-like 1s2p ¹P₁/³P₁ analytic test (the
       two J=1 CSFs ``|1s 2p_{1/2}; J=1⟩`` and ``|1s 2p_{3/2}; J=1⟩``
       should diagonalise to the singlet/triplet states with the
       2·G¹(1s,2p) splitting).

    Parameters
    ----------
    csf_r, csf_s : CSF
        The two CSFs whose off-diagonal element is being corrected.
        ``r != s`` (we never call this on the diagonal).
    orbitals_dict : dict
        ``(n, kappa) -> (P, Q)`` orbital map.
    bsp_data : dict
        B-spline basis data for Slater integrals.
    ns : int
        Number of B-spline coefficients per orbital (for orbital lookup).
    n_diff : int
        Number of subshell differences (1 or 2).  Used to dispatch on
        the 1-diff vs 2-diff sub-formula; not used in the no-op
        scaffold.

    Returns
    -------
    float
        Additive correction to be added to the base 1-diff / 2-diff
        matrix element.  Always ``0.0`` in the current scaffold.
    """
    # Only apply when both CSFs carry explicit J/parity metadata and
    # sit in the same (J, parity) block.  The block filter in
    # ``methods/ci.py`` already drops cross-block elements, so this
    # check is mostly a clarity gate for the no-op path.
    if not (_has_explicit_J(csf_r) and _has_explicit_J(csf_s)):
        return 0.0
    if int(csf_r.total_J2) != int(csf_s.total_J2):
        return 0.0
    if int(getattr(csf_r, "parity", 1)) != int(getattr(csf_s, "parity", 1)):
        return 0.0

    if n_diff not in (1, 2):
        return 0.0

    if n_diff == 1:
        return _off_diagonal_one_diff_J_correction(
            csf_r, csf_s, orbitals_dict, bsp_data, ns,
        )

    return _off_diagonal_two_diff_J_correction(
        csf_r, csf_s, orbitals_dict, bsp_data, ns,
    )


def _one_diff_single_spectator_exact_correction(
    csf_r, csf_s, key_a, key_b, key_c, sqrt_factor, orbitals_dict, bsp_data, ns
):
    """Exact chain-coupled replacement for a one-electron spectator.

    The compact 6j shortcut is reliable for the FAC-like single-hole
    spectator branch used by the same-hole 3d spin-orbit pair, but it
    over-corrects the Fe XVII p-hole move with a single 3d spectator.  In
    that case evaluate the scalar product of the two one-body tensors in the
    same m-scheme machinery used for branch-9 two-diff terms, then subtract
    the configuration-average contribution already present in
    ``_one_diff_element``.
    """
    Pa, Qa = _get_orb(key_a, orbitals_dict, ns)
    Pb, Qb = _get_orb(key_b, orbitals_dict, ns)
    Pc, Qc = _get_orb(key_c, orbitals_dict, ns)

    proper = 0.0
    base = 0.0
    kappa_a, kappa_b, kappa_c = int(key_a[1]), int(key_b[1]), int(key_c[1])
    tj_c = 2 * abs(kappa_c) - 1

    if kappa_a == kappa_b:
        rk_direct = _rk(Pa, Qa, Pc, Qc, Pb, Qb, Pc, Qc, bsp_data, 0)
        base += rk_direct

    base_ks = sorted(set(exchange_k_range(kappa_a, kappa_c))
                     & set(exchange_k_range(kappa_c, kappa_b)))
    for k in base_ks:
        lam_ac = exchange_lambda(kappa_a, kappa_c, k)
        lam_cb = exchange_lambda(kappa_c, kappa_b, k)
        if lam_ac < 1.0e-15 or lam_cb < 1.0e-15:
            continue
        rk_exch = _rk(Pa, Qa, Pc, Qc, Pc, Qc, Pb, Qb, bsp_data, k)
        ang_k = lam_ac if kappa_a == kappa_b else float(np.sqrt(lam_ac * lam_cb))
        base -= ang_k / (tj_c + 1.0) * rk_exch

    direct_ks = sorted(set(exchange_k_range(kappa_a, kappa_b))
                       & set(exchange_k_range(kappa_c, kappa_c)))
    for k in direct_ks:
        coeff = _branch9_double_excitation_scalar(
            csf_r, csf_s, key_b, key_a, key_c, key_c, 2 * k,
        )
        if coeff is None or abs(coeff) < 1.0e-15:
            continue
        radial = _rk(Pa, Qa, Pb, Qb, Pc, Qc, Pc, Qc, bsp_data, k)
        proper += coeff * radial

    exchange_ks = sorted(set(exchange_k_range(kappa_a, kappa_c))
                         & set(exchange_k_range(kappa_c, kappa_b)))
    for k in exchange_ks:
        coeff = _branch9_double_excitation_scalar(
            csf_r, csf_s, key_c, key_a, key_b, key_c, 2 * k,
        )
        if coeff is None or abs(coeff) < 1.0e-15:
            continue
        radial = _rk(Pa, Qa, Pc, Qc, Pc, Qc, Pb, Qb, bsp_data, k)
        proper += coeff * radial

    return float(sqrt_factor * (proper - base))


def _off_diagonal_one_diff_J_correction(csf_r, csf_s, orbitals_dict, bsp_data, ns):
    """Grant-style J-coupled correction for one-diff unique-parentage CSFs."""
    diffs = csf_r.differs_from(csf_s)
    lost = [(key, qr, qs) for key, qr, qs in diffs if qr > qs]
    gained = [(key, qr, qs) for key, qr, qs in diffs if qr < qs]
    if len(lost) != 1 or len(gained) != 1:
        return 0.0

    key_a, qa_r, qa_s = lost[0]    # destination in the bra
    key_b, qb_r, qb_s = gained[0]  # source in the ket
    if int(round(qa_r - qa_s)) != 1 or int(round(qb_s - qb_r)) != 1:
        return 0.0

    tj_a = 2 * abs(int(key_a[1])) - 1
    tj_b = 2 * abs(int(key_b[1])) - 1
    sqrt_factor = float(np.sqrt(qb_s * (2 * abs(int(key_a[1])) - qa_s)))
    total_J2 = int(csf_r.total_J2)
    occ_r = {(int(sh.n), int(sh.kappa)): float(sh.occupation) for sh in csf_r.subshells}
    occ_s = {(int(sh.n), int(sh.kappa)): float(sh.occupation) for sh in csf_s.subshells}

    delta = 0.0
    seen: set[tuple[int, int]] = set()
    for sh_c in csf_r.subshells:
        key_c = (int(sh_c.n), int(sh_c.kappa))
        if key_c in seen or key_c in (key_a, key_b):
            continue
        seen.add(key_c)
        q_common = min(occ_r.get(key_c, 0.0), occ_s.get(key_c, 0.0))
        if q_common <= 0.0:
            continue

        cap_c = 2 * abs(int(key_c[1]))
        q_common_i = int(round(q_common))
        # The compact 6j expression is valid when the spectator parentage is
        # unique: a single electron or a single hole.  Closed spectators remain
        # covered by the configuration-averaged base kernel.
        if q_common_i not in (1, cap_c - 1):
            continue

        if q_common_i == 1 and bsp_data:
            delta += _one_diff_single_spectator_exact_correction(
                csf_r, csf_s, key_a, key_b, key_c, sqrt_factor,
                orbitals_dict, bsp_data, ns,
            ) / sqrt_factor
            continue

        tj_c = cap_c - 1
        Pa, Qa = _get_orb(key_a, orbitals_dict, ns)
        Pb, Qb = _get_orb(key_b, orbitals_dict, ns)
        Pc, Qc = _get_orb(key_c, orbitals_dict, ns)

        k_values = sorted(
            set(exchange_k_range(key_a[1], key_c[1]))
            & set(exchange_k_range(key_c[1], key_b[1]))
        )
        for k in k_values:
            lam_ac = exchange_lambda(key_a[1], key_c[1], k)
            lam_cb = exchange_lambda(key_c[1], key_b[1], k)
            if lam_ac < 1.0e-15 or lam_cb < 1.0e-15:
                continue
            angular_mag = float(np.sqrt(lam_ac * lam_cb))
            avg_coef = q_common / (tj_c + 1.0) * angular_mag
            sixj = wigner_6j(tj_a, tj_b, total_J2, tj_c, tj_c, 2 * k)
            if abs(sixj) < 1.0e-15:
                j_coef = 0.0
            else:
                # FAC keeps the signed ReducedCL products separate from the
                # radial Slater integral.  Using sqrt(lambda) here loses the
                # relative p-/p+ and d-/d+ phases needed for the Fe XVII
                # particle-hole J=1 off-diagonal signs.
                r_ac = reduced_cl_j2(tj_a, 2 * k, tj_c)
                r_cb = reduced_cl_j2(tj_c, 2 * k, tj_b)
                phase_exp = (tj_a + tj_b + total_J2) // 2 + k
                phase = -1.0 if phase_exp % 2 else 1.0
                if q_common_i == cap_c - 1 and cap_c > 2:
                    # Particle-hole conjugation for a single-hole spectator.
                    # FAC's recoupled Hamiltonian keeps the k=1 and k=3
                    # same-hole 2p3/2^-1 3d J=1 terms in phase; only the
                    # global hole sign differs from the electron spectator.
                    phase *= -1.0
                j_coef = phase * sixj * r_ac * r_cb

            if bsp_data:
                rk_exch = _rk(Pa, Qa, Pc, Qc, Pc, Qc, Pb, Qb, bsp_data, k)
            else:
                # Unit-radial fallback for analytic angular tests that only
                # assert the correction is wired and non-zero.
                rk_exch = 1.0
            delta += (avg_coef - j_coef) * rk_exch

    # ``_one_diff_element`` multiplies the configuration-average spectator
    # bracket by the jj occupation square-root factor.  The J-coupled
    # replacement correction lives in the same bracket, so it must carry the
    # same factor before being added to the base matrix element.
    return float(sqrt_factor * delta)


def _fac_inner_shell_index(chain_keys, key):
    """Return FAC-style one-based shell index for an aligned key chain."""
    try:
        return tuple(chain_keys).index(tuple(key)) + 1
    except ValueError:
        return None


def _fac_one_diff_open_spectator_exchange_coef(csf_r, csf_s, key_a, key_b, key_c, k):
    """Conservative reachability probe for the FAC one-diff open-spectator path."""
    occ_r = {(int(sh.n), int(sh.kappa)): int(round(sh.occupation)) for sh in csf_r.subshells}
    occ_s = {(int(sh.n), int(sh.kappa)): int(round(sh.occupation)) for sh in csf_s.subshells}
    q_common = min(occ_r.get(tuple(key_c), 0), occ_s.get(tuple(key_c), 0))
    cap_c = 2 * abs(int(key_c[1]))
    if q_common not in (1, cap_c - 1):
        return None
    if exchange_lambda(key_a[1], key_c[1], k) <= 1.0e-15:
        return None
    if exchange_lambda(key_c[1], key_b[1], k) <= 1.0e-15:
        return None
    return 0.0


def _chain_one_diff_spectator_exact_scalar(*_args, **_kwargs):
    """Placeholder for optional branch-inventory diagnostics."""
    return None


def _fac_two_diff_branch9_element(
    csf_r, csf_s, key_a, key_b, key_c, key_d, orbitals_dict, bsp_data, ns,
    *, unit_radials=False,
):
    """Reachability probe for FAC's all-four-distinct AngularZxZ0 branch.

    Production still uses :func:`_off_diagonal_two_diff_J_correction`; this
    helper exists so the inventory can distinguish branch-9 coverage from
    completely unsupported CSF shapes while the full FAC SumCoeff recoupling is
    being ported.
    """
    keys = {tuple(key_a), tuple(key_b), tuple(key_c), tuple(key_d)}
    if len(keys) != 4:
        return None
    if int(getattr(csf_r, "total_J2", 0)) != int(getattr(csf_s, "total_J2", 0)):
        return None
    if int(getattr(csf_r, "parity", 1)) != int(getattr(csf_s, "parity", 1)):
        return None
    if unit_radials:
        return 0.0
    return _off_diagonal_two_diff_J_correction(csf_r, csf_s, orbitals_dict, bsp_data, ns)
