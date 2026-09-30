"""CI Hamiltonian construction from exact jj-coupled CSFs.

One- and two-electron operators connect determinants differing by at most
two electron replacements. The angular kernel applies the Slater–Condon
rules before projection onto the coupled CSF basis.

The production builder contracts exact angular coefficients in each
J/parity block with one-body Dirac and two-body radial integrals. CSF
parentage and fermion phases belong to ``angular.csf_hamiltonian``;
radial integration uses the supplied grid or B-spline representation.
"""

import math
import os
from functools import lru_cache

import numpy as np
from scipy.sparse.linalg import LinearOperator

from atomx.angular.clebsch import clebsch_gordan
from atomx.angular.coefficients import (
    cg,
    ckm,
    exchange_k_range,
    exchange_lambda,
    reduced_cl_j2,
)
from atomx.angular.csf_recoupling import (
    chain_projection_amplitudes,
    make_chain_coupled_state,
    single_state_for_subshell,
)
from atomx.angular.rcfp import ReducedSubshellState
from atomx.angular.wigner import wigner_6j
from atomx.angular.csf_hamiltonian import (
    MAX_BLOCK_DETERMINANTS,
    csf_signature,
    exact_block_matrix,
)
from atomx.numerics.bspline.convol import (
    ConvolCache,
    _convol_case1_t,
    _density_sym,
    _sum_amb_banded,
    compute_slater_fk_cell,
    compute_slater_gk_cell,
    compute_slater_rk_cell,
)
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices
from atomx.numerics.shooting.constants import SL
from atomx.numerics.shooting.quadrature import slag6, slag6_weights
from atomx.numerics.shooting.coulomb import _slater_integral_slag6, _yk_over_r

# Build-scoped memo for Slater integrals and Dirac H0/S0 blocks: every CSF
# in a CI build shares ONE orbital basis, so the same (orbitals, k)
# integrals recur across all diagonal/off-diagonal elements — a Co ground
# CI rebuilt them ~300k times per MCDHF outer iteration (7600 elements x
# ~40 integrals each) where only a few hundred are distinct.  Keys use
# id() of the orbital coefficient arrays, which orbitals_dict keeps alive
# for the duration of the build; build_ci_matrix clears the id()-keyed
# entries on entry so nothing leaks across builds whose orbitals differ.
# The one exception is the ("h0", ns, kappa, Z) one-body Dirac blocks,
# which carry no id() and are kept across builds of the same workspace
# under the same nuclear model — see _reset_memo_for_ci_build.
#
# An id()-keyed cache is only sound while the keyed arrays are alive:
# CPython recycles an id() as soon as the object is freed, so a later array
# can inherit an id and read a stale entry.  That is enforced here rather
# than assumed — every array whose id() enters a key is appended to
# ``_MEMO_KEEPALIVE`` when its entry is *stored*, so no memoised id can be
# recycled for the memo's lifetime, and ``_memo_context`` drops the whole
# memo if a different radial workspace is used (a key does not carry
# bsp_data, so a stale entry from another grid must not survive).
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

# Strong references to every array whose id() appears in a _SLATER_MEMO key.
_MEMO_KEEPALIVE: list = []

# The radial workspace the current memo contents belong to (identity, not id()).
_MEMO_CONTEXT = None


def _reset_memo() -> None:
    """Drop all memoised integrals and the references pinning their keys."""
    global _MEMO_CONTEXT
    _SLATER_MEMO.clear()
    _MEMO_KEEPALIVE.clear()
    _MEMO_CONTEXT = None


def _reset_memo_for_ci_build(bsp_data, nuclear_potential):
    """Entry-point reset for the CI builders: wipe orbital-dependent
    integrals, carry what is orbital-independent.

    Every id()-keyed Slater integral is wiped, as before — orbital
    coefficients change between builds and the id-recycling argument in
    the module docstring requires it.  But the ``("h0", ns, kappa, Z)``
    one-body Dirac blocks from :func:`_dirac_blocks` are rebuilt from
    (basis geometry, symmetry, Z, nuclear model) alone; they contain no
    id() and are bit-identical on every rebuild.  Rebuilding them once
    per ``build_ci_matrix`` call cost ~16 % of a variational MCDHF solve
    (measured on Fe XVII j-resolved; docs/HPC_NUMERICS_RESEARCH_2026-09-27.md
    §2.2), with ~3800 redundant builds per solve.

    The carry is sound only when BOTH guards hold, and is applied only
    then:

    - the radial workspace is the identical object (``_memo_context``
      resets on any workspace change — the MCDHF driver rebuilds
      ``bsp_data`` fresh every outer iteration, so a new basis still
      triggers a full rebuild once per iteration), and
    - the nuclear-potential object is identical (``build_dirac_matrices``
      takes the potential as an argument; a different finite-nucleus
      model must not read blocks built for the previous one).

    The stashed ``_convol_cache`` (rkb tables, grid-only) is carried
    exactly as before.
    """
    stashed_cache = _SLATER_MEMO.get("_convol_cache")
    prev_potential = _SLATER_MEMO.get("_nuclear_potential")
    carry_h0 = (
        not isinstance(bsp_data, GridCIWorkspace)
        and _MEMO_CONTEXT is bsp_data
        and prev_potential is nuclear_potential
        and "_nuclear_potential" in _SLATER_MEMO
    )
    carried = (
        {k: v for k, v in _SLATER_MEMO.items()
         if isinstance(k, tuple) and k[0] == "h0"}
        if carry_h0 else None
    )
    _reset_memo()
    _memo_context(bsp_data)
    if stashed_cache is not None:
        _SLATER_MEMO["_convol_cache"] = stashed_cache
    if carried:
        _SLATER_MEMO.update(carried)


def _memo_context(bsp_data):
    """Bind the memo to one radial workspace, discarding it on a change."""
    global _MEMO_CONTEXT
    if _MEMO_CONTEXT is not bsp_data:
        _reset_memo()
        _MEMO_CONTEXT = bsp_data
    return bsp_data


def _pin(*arrays) -> None:
    """Keep ``arrays`` alive so their id()s cannot be recycled while memoised.

    Called only when an entry is *stored*, which is what makes the id() keys
    sound: every id in the memo then belongs to a live array, so no lookup
    can hit an entry left behind by a freed one.
    """
    _MEMO_KEEPALIVE.extend(arrays)


def _context_cache(bsp_data) -> ConvolCache:
    """Cell-integral rkb cache for the current build (lazily created).

    Standalone helper calls (unit tests poking ``_diagonal_element`` etc.
    without going through :func:`build_ci_matrix`) get a fresh cache bound
    to their ``bsp_data``; the *content* check evicts a stale cache when the
    grid changes between direct calls (an identity check would needlessly
    rebuild the banks for a rebuilt — but bit-identical — workspace dict;
    with the process-wide bank registry the rebuilt cache is free either
    way).
    """
    cache = _SLATER_MEMO.get("_convol_cache")
    if cache is None or not cache.same_grid(bsp_data):
        cache = ConvolCache(bsp_data)
        _SLATER_MEMO["_convol_cache"] = cache
    return cache


def _fk(Pa, Qa, Pb, Qb, bsp_data, k):
    _memo_context(bsp_data)
    ia, ib = id(Pa), id(Pb)
    key = ("f", k) + ((ia, ib) if ia <= ib else (ib, ia))
    v = _SLATER_MEMO.get(key)
    if v is None:
        _pin(Pa, Pb)
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
    _memo_context(bsp_data)
    ia, ib = id(Pa), id(Pb)
    key = ("g", k) + ((ia, ib) if ia <= ib else (ib, ia))
    v = _SLATER_MEMO.get(key)
    if v is None:
        _pin(Pa, Pb)
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
    _memo_context(bsp_data)
    key = ("r", k, id(Pa), id(Pb), id(Pc), id(Pd))
    v = _SLATER_MEMO.get(key)
    if v is None:
        _pin(Pa, Pb, Pc, Pd)
        if isinstance(bsp_data, GridCIWorkspace):
            # Convention: the first two orbitals form the first pair and the
            # last two the second, i.e. ``R^k(ab, cd)`` — the same reading as
            # the cell integrator below.  Every caller passes the orbitals in
            # the role order it wants, e.g. ``_rk(Pa, Pc, Pb, Pd)`` for
            # ``R^k(ac, bd)``.  This branch used to pair (a, c) and (b, d)
            # instead, so the two engines evaluated *different* integrals for
            # the same call and the grid-flavour CI was wrong wherever an
            # off-diagonal element went through here.
            v = float(_slater_integral_slag6(
                _grid_pair_density(Pa, Qa, Pb, Qb),
                _grid_pair_density(Pc, Qc, Pd, Qd),
                k, bsp_data.r, bsp_data.difr, bsp_data.n))
        else:
            v = compute_slater_rk_cell(Pa, Qa, Pb, Qb, Pc, Qc, Pd, Qd,
                                       bsp_data, k,
                                       cache=_context_cache(bsp_data))
        _SLATER_MEMO[key] = v
    return v


def _dirac_blocks(ns, kappa, Z, bsp_data):
    _memo_context(bsp_data)
    key = ("h0", ns, kappa, Z)
    v = _SLATER_MEMO.get(key)
    if v is None:
        v = build_dirac_matrices(ns, kappa, Z, bsp_data,
                                 _SLATER_MEMO.get("_nuclear_potential"))
        _SLATER_MEMO[key] = v
    return v


def _breit_rk(f1, f2, f3, f4, comps, bsp_data, L):
    """``int int f1 f3 (r_<^L/r_>^{L+1}) f2 f4`` — flavour dispatched.

    The Breit (Gaunt) operator needs *component-resolved* Slater
    integrals: the density on electron 1 is ``f1*f3`` (one Dirac component
    of the bra orbital times one of the ket orbital) and on electron 2
    ``f2*f4``.  The Coulomb path never needs this — it always uses the
    summed density ``P_aP_c + Q_aQ_c`` inside :func:`_rk` — so this is a
    separate integrator rather than a mode of ``_rk``.

    ``f1..f4`` are expansion coefficient vectors (B-spline flavour) or
    tabulated arrays (grid flavour) and ``comps`` says which Dirac
    component each one is: the B-spline large and small components live
    in *different* bases, so evaluating a small-component vector through
    the large-component table would silently mix them.

    The B-spline path uses interpolatory within-cell partial integrals
    and Gauss sums across cells. The grid path retains SLAG6. Symmetric
    electron interchange is enforced at quadrature level.
    """
    _memo_context(bsp_data)
    key = ("br", L, id(f1), id(f2), id(f3), id(f4)) + tuple(comps)
    v = _SLATER_MEMO.get(key)
    if v is None:
        _pin(f1, f2, f3, f4, bsp_data)
        if isinstance(bsp_data, GridCIWorkspace):
            v = float(_slater_integral_slag6(
                f1 * f3, f2 * f4, L, bsp_data.r, bsp_data.difr, bsp_data.n))
        else:
            from atomx.numerics.radial_workspace import (
                eval_orbital_pair,
                workspace_arrays,
            )
            from atomx.numerics.yk import compute_yk

            gr, grw = workspace_arrays(bsp_data)
            r_flat = np.ravel(gr)
            w_flat = np.ravel(grw)

            def _eval(coeff, comp):
                zeros = np.zeros_like(coeff)
                pair = ((coeff, zeros) if comp == "P" else (zeros, coeff))
                return np.ravel(eval_orbital_pair(*pair, bsp_data)[
                    0 if comp == "P" else 1])

            rho1 = _eval(f1, comps[0]) * _eval(f3, comps[2])
            rho2 = _eval(f2, comps[1]) * _eval(f4, comps[3])
            v = float(0.5 * np.sum(grw * (rho1.reshape(gr.shape) *
                compute_yk(rho2.reshape(gr.shape), gr, grw, *gr.shape, L) +
                rho2.reshape(gr.shape) *
                compute_yk(rho1.reshape(gr.shape), gr, grw, *gr.shape, L))))
        _SLATER_MEMO[key] = v
    return v


# ---------------------------------------------------------------------------
# Grid (shooting-engine) flavour of the radial workspace
# ---------------------------------------------------------------------------
#
# Same seam idea as atomx.numerics.radial_workspace: ONE physics implementation
# of the CI matrix elements serves both orbital representations.  Dispatch is
# on the workspace object (``isinstance(bsp_data, GridCIWorkspace)``), never
# on array shapes.  The consistency principle that fixed the +74 Ha Pb bug
# applies verbatim here: the grid flavour evaluates every matrix element with
# the shooting engine's OWN numerics — Slater integrals through
# ``_slater_integral_slag6`` (the same integrator
# ``RadialHartreeFock.two_electron_energy`` uses) and one-electron Dirac
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
        nuclear model (point/Fermi) — and the identical plasma screening
        (``nuclear_rv_screened``) — the SCF solved with.
        """
        r = result._grid
        difr = result._difr
        return cls(r, difr, hf_solver.nuclear_rv_screened(r, difr, len(r)))


# 8th-order first-derivative stencils on the uniform-s index grid.
# Central: antisymmetric weights for offsets ±1..±4; edge rows i=0..3 are
# the one-sided/offset 8th-order weights on nodes 0..8 (right edge uses the
# mirrored negated rows — first-derivative parity).
from atomx.numerics.shooting.operators import (
    _FD_CENTRAL, _FD_EDGE, _ds_derivative,
)


def _grid_ds(y):
    key = ("ds", id(y))
    v = _SLATER_MEMO.get(key)
    if v is None:
        _pin(y)
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
        _pin(Pa, Pb)
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
        _memo_context(bsp_data)
        ia, ib = id(Pa), id(Pb)
        key = ("h1", kappa) + ((ia, ib) if ia <= ib else (ib, ia))
        v = _SLATER_MEMO.get(key)
        if v is None:
            _pin(Pa, Pb)
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
        _memo_context(bsp_data)
        key = ("n2", id(P))
        v = _SLATER_MEMO.get(key)
        if v is None:
            _pin(P)
            rho = _grid_pair_density(P, Q, P, Q)
            v = float(slag6(1.0, rho * bsp_data.difr,
                            bsp_data.n)[bsp_data.n - 1])
            _SLATER_MEMO[key] = v
        return v
    _H0, S0 = _dirac_blocks(ns, kappa, Z, bsp_data)
    vec = np.concatenate([P, Q])
    return vec @ S0 @ vec


def build_ci_matrix(csfs, orbitals_dict, orbital_basis, bsp_data, Z, ns,
                    nuclear_potential=None, convol_cache=None,
                    max_determinants=None, include_breit=False, as_operator=False,
                    *, breit_orbital_energies=None, one_body_shifts=None):
    """Build the exact CI Hamiltonian matrix H_{rs} in the CSF basis.

    ``H[r, s]`` is the true expectation value ``<CSF_r|H|CSF_s>`` of the
    Dirac-Coulomb Hamiltonian in the jj-coupled CSF basis: each CSF is
    expanded as a vector over the Slater determinants of its own
    configuration at ``2M = 2J``, the determinant Hamiltonian is built by
    explicit Slater-Condon rules, and the CSF matrix is the projection
    ``H = C^T H_det C`` (see :mod:`atomx.angular.csf_hamiltonian` and
    ``benchmarks/mscheme_ci_reference.py``, the independent oracle it is
    gated against).  The result is variational and symmetric, and elements
    between different ``(2J, parity)`` are exactly ``0.0``.

    The previous kernel — a configuration-average diagonal plus two
    "J-correction" patches — is still reachable as
    :func:`_legacy_build_ci_matrix_approximate` for comparison (explicit
    ``allow_legacy_incorrect_physics=True`` opt-in), but is *not* on
    the production path: it is not a representation of the Hamiltonian
    (Be ``correlation="sd"`` came out 0.39 Ha below the exact ground
    state).  There is no fallback to it.

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
        per CI construction.  A cache built on a *different* grid is
        ignored and a fresh one created.  B-spline flavour only.
    max_determinants : int or None
        Cap on the raw determinant dimension of one ``(2J, parity)``
        block (default :data:`atomx.angular.csf_hamiltonian.MAX_BLOCK_DETERMINANTS`).  The determinant space at ``2M = 2J`` also
        contains the lower-M members of higher-J multiplets, so it can be
        much larger than the CSF space; exceeding the cap raises
        ``ExactCISpaceError`` naming the configuration and both sizes
        rather than hanging.
    include_breit : bool
        Add the Breit interaction to each ``(2J, parity)`` block as an
        operator (Gaunt plus static retardation, see
        :mod:`atomx.physics.csf.breit_ci`), so it splits levels instead
        of shifting them uniformly.  Off by default: the extra integrals
        are only paid for when asked. Static unless explicit orbital energies
        are supplied below.
    breit_orbital_energies : mapping or None
        Explicit ``{(n, kappa): energy_in_Hartree}`` for finite-frequency
        transverse exchange. Requires ``include_breit=True``. Uses the
        symmetric average at the two electron-pair energy transfers, with
        energies held fixed during CI. Replaces static Breit; does not add
        a second Breit interaction or imply full bound-state QED.
    """
    if breit_orbital_energies is not None and not include_breit:
        raise ValueError("breit_orbital_energies requires include_breit=True")
    # ``_reset_memo`` drops every memoised integral, including the CI's
    # stashed rkb cache and the orbital-independent h0 blocks.  The
    # entry-point reset carries the rkb cache (as before) and, when the
    # workspace and nuclear model are unchanged, the h0 blocks too.
    _reset_memo_for_ci_build(bsp_data, nuclear_potential)
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
        if convol_cache is not None and convol_cache.same_grid(bsp_data):
            _SLATER_MEMO["_convol_cache"] = convol_cache
        _SLATER_MEMO["_nuclear_potential"] = nuclear_potential
    _check_orbitals_present(csfs, orbitals_dict)
    if _metric_check_due(bsp_data):
        _check_orbital_metric(csfs, orbitals_dict, bsp_data, Z, ns)
    n_csf = len(csfs)
    H = None if as_operator else np.zeros((n_csf, n_csf))
    if n_csf == 0:
        return np.zeros((0, 0))
    if max_determinants is None:
        max_determinants = MAX_BLOCK_DETERMINANTS
    evaluate = _radial_evaluator(orbitals_dict, bsp_data, Z, ns,
                                 nuclear_potential=nuclear_potential)
    if one_body_shifts:
        # CI+MBPT Σ⁽¹⁾ injection: add a diagonal effective one-body shift
        # <a|h_D + Σ₁|a> = <a|h_D|a> + Δε_a for each listed orbital.  The
        # determinant/CSF projection machinery then places the shift on
        # every occupied occurrence of the orbital with the correct
        # one-body density weights, including between-CSF elements.
        _base_evaluate = evaluate

        def evaluate(key):  # noqa: F811 — deliberate narrow shadow
            value = _base_evaluate(key)
            if key[0] == "h":
                _, n_a, kappa, n_c = key
                if n_a == n_c and (n_a, kappa) in one_body_shifts:
                    value = value + one_body_shifts[(n_a, kappa)]
            return value

    breit_evaluate = (_breit_radial_evaluator(orbitals_dict, bsp_data, Z, ns,
                                           orbital_energies=breit_orbital_energies)
                      if include_breit else None)
    operator_blocks = []
    for _key, idx in _group_csf_indices(csfs):
        signatures = [csf_signature(csfs[i]) for i in idx]
        h_block = exact_block_matrix(signatures, evaluate,
                                     max_determinants=max_determinants,
                                     breit_evaluator=breit_evaluate,
                                     as_operator=as_operator)
        if as_operator:
            operator_blocks.append((np.asarray(idx), h_block))
        else:
            H[np.ix_(idx, idx)] = h_block
    if as_operator:
        def apply(v):
            out = np.zeros_like(v, dtype=float)
            for idx, block in operator_blocks:
                out[idx] = block @ v[idx]
            return out

        diagonal = np.zeros(n_csf)
        for idx, block in operator_blocks:
            diagonal[idx] = block.diagonal()
        H = LinearOperator((n_csf, n_csf), matvec=apply, matmat=apply,
                           rmatvec=apply, dtype=float)
        H.diagonal = lambda: diagonal.copy()
        H.blocks = operator_blocks
    # Elements between different (2J, parity) blocks were never written and
    # are exactly 0.0 — J and parity are exact symmetries, and the old
    # kernel's non-zero cross-J elements (Be 1s2 2s1 2p1: -14.479 Ha between
    # J=0 and J=1) are gone rather than merely masked by ci._solve_blocked.
    return H


def _group_csf_indices(csfs):
    """Group CSF indices by ``(2J, parity)``, lowest J first.

    Stable and deterministic; the order only fixes the order in which
    blocks are built (and therefore which caches warm first), never the
    matrix values.
    """
    groups: dict = {}
    for i, csf in enumerate(csfs):
        key = (int(getattr(csf, "total_J2", 0)), int(getattr(csf, "parity", 1)))
        groups.setdefault(key, []).append(i)
    return sorted(groups.items(), key=lambda kv: (kv[0][0], -kv[0][1]))


def _radial_evaluator(orbitals_dict, bsp_data, Z, ns, *, nuclear_potential=None):
    """Map an exact-CI radial key to its numeric value.

    Two key shapes, evaluated with the same radial kernels as the SCF
    energy functional. Densities and pair potentials are shared within
    this build; orbitals must remain unchanged for the evaluator's lifetime.

    * ``("h", n_a, kappa, n_c)``  — ``<a|h_D|c> / <c|c>``, the
      normalisation convention the oracle uses;
    * ``("r", k, (n_a,kappa_a), (n_c,kappa_c), (n_b,kappa_b),
      (n_d,kappa_d))`` — ``R^k(ac, bd)``, i.e. ``_rk`` called with its
      four orbitals in the role order ``Pa, Pc, Pb, Pd``.
    """
    is_grid = isinstance(bsp_data, GridCIWorkspace)
    if is_grid:
        weights = slag6_weights(bsp_data.n) * bsp_data.difr
    else:
        # Capture the one-body blocks with this operator's physical model.
        # Evaluators can outlive/interleave CI builds with another nucleus;
        # consulting the process memo later would silently change h_D.
        _memo_context(bsp_data)
        if _SLATER_MEMO.get('_nuclear_potential') is not nuclear_potential:
            _reset_memo_for_ci_build(bsp_data, nuclear_potential)
        _SLATER_MEMO['_nuclear_potential'] = nuclear_potential
        one_body_blocks = {kappa: _dirac_blocks(ns, kappa, Z, bsp_data)
                           for _, kappa in orbitals_dict}
        orders = (bsp_data.get("ksp", bsp_data["ks"]),
                  bsp_data.get("ksq", bsp_data["ks"]))
        cache = _context_cache(bsp_data)

    @lru_cache(maxsize=512)
    def density(a, b):
        if is_grid:
            pa, qa = _get_orb(a, orbitals_dict, ns)
            pb, qb = _get_orb(b, orbitals_dict, ns)
            return pa*pb+qa*qb
        return tuple(_density_sym(x, y, ns, order)
                     for x, y, order in zip(orbitals_dict[a], orbitals_dict[b], orders, strict=True))

    @lru_cache(maxsize=256 if is_grid else 1024)
    def pair_potential(rank, a, b):
        source = density(a, b)
        if is_grid:
            return weights * _yk_over_r(source, rank, bsp_data.r,
                                        bsp_data.difr, bsp_data.n)
        potentials = []
        for left, label_left in enumerate(("p", "q")):
            value = np.zeros((ns, orders[left]))
            for right, label_right in enumerate(("p", "q")):
                bank, extent, _ = cache.get_rkb_t(rank, label_left + label_right + label_left + label_right)
                value += _convol_case1_t(bank, source[right], extent, orders[left], orders[right])
            potentials.append(value)
        return tuple(potentials)

    @lru_cache(maxsize=8192)
    def evaluate(key):
        if key[0] == "h":
            _, n_a, kappa, n_c = key
            Pa, Qa = _get_orb((n_a, kappa), orbitals_dict, ns)
            Pc, Qc = _get_orb((n_c, kappa), orbitals_dict, ns)
            if is_grid:
                h = _h_one_body(Pa, Qa, Pc, Qc, kappa, Z, bsp_data, ns)
                norm_sq = _norm_sq(Pc, Qc, kappa, Z, bsp_data, ns)
            else:
                h0, overlap = one_body_blocks[kappa]
                bra, ket = np.concatenate((Pa, Qa)), np.concatenate((Pc, Qc))
                h = bra @ h0 @ ket
                norm_sq = ket @ overlap @ ket
            if norm_sq > 0.0:
                return h / norm_sq
            # A zero-norm "orbital" is not a wavefunction and h/0 is
            # undefined.  Report a non-finite element instead of dropping
            # it: the old kernel's ``norm_sq > 1e-20`` guard silently
            # returned 0 here, which is one of the silent-zero failure
            # modes MissingOrbitalError exists to stop.
            return float("nan")
        _, k, key_a, key_c, key_b, key_d = key
        if not is_grid:
            # Many R^k integrals share a pair density. Contract its cell
            # kernel once; subsequent integrals are banded dot products.
            # Both caches belong to this immutable CI build and are bounded.
            source = tuple(sorted((key_b, key_d)))
            target = tuple(sorted((key_a, key_c)))
            value = sum(_sum_amb_banded(v, d, ns, order)
                        for v, d, order in zip(pair_potential(k, *source),
                                               density(*target), orders, strict=True))
            if not np.isfinite(value):
                raise FloatingPointError(f"non-finite Slater integral {key}")
            return value
        # Keep the original SLAG6 source/target order: its discrete kernel
        # is not exactly symmetric. Reuse only identical pair potentials;
        # exchanging the two electron densities would change quadrature.
        source = tuple(sorted((key_a, key_c)))
        target = tuple(sorted((key_b, key_d)))
        return float(np.dot(pair_potential(k, *source), density(*target)))

    return evaluate


def _breit_radial_evaluator(orbitals_dict, bsp_data, Z, ns, *, orbital_energies=None):
    """Map a Breit (Gaunt or retardation) radial key to its numeric value.

    Explicit ``orbital_energies`` selects finite-frequency transverse
    exchange. With no mapping the existing static integrators are used.

    Key shape (see :func:`atomx.physics.csf.breit_ci.gaunt_terms`)::

        ("g", L, (X_a, n_a, kappa_a), (X_c, n_c, kappa_c),
                 (X_b, n_b, kappa_b), (X_d, n_d, kappa_d))

    with ``X`` in ``{"P", "Q"}`` selecting the large/small Dirac
    component of that orbital.  The value is

        ``int int f_a(r1) f_c(r1) [r_<^L / r_>^{L+1}] f_b(r2) f_d(r2)``

    evaluated with the *same* radial workspace and integrator the
    Coulomb elements use (:func:`_rk`'s conventions), so the two
    operators are built on identical numerics.
    """
    if orbital_energies is not None:
        from atomx.physics.csf.transverse import radial_evaluator
        return radial_evaluator(orbitals_dict, bsp_data, orbital_energies)
    from atomx.physics.csf.retardation import radial_evaluator
    evaluate_retardation = radial_evaluator(orbitals_dict, bsp_data)
    def evaluate(key):
        if key[0] == "t":
            return evaluate_retardation(key)
        _, L, key_a, key_c, key_b, key_d = key
        coeffs = []
        comps = []
        for (comp, n, kappa) in (key_a, key_c, key_b, key_d):
            P, Q = _get_orb((n, kappa), orbitals_dict, ns)
            coeffs.append(P if comp == "P" else Q)
            comps.append(comp)
        return _breit_rk(coeffs[0], coeffs[2], coeffs[1], coeffs[3],
                         (comps[0], comps[2], comps[1], comps[3]),
                         bsp_data, L)

    return evaluate


def _legacy_build_ci_matrix_approximate(csfs, orbitals_dict, orbital_basis,
                                        bsp_data, Z, ns,
                                        nuclear_potential=None,
                                        convol_cache=None, *,
                                        allow_legacy_incorrect_physics=False):
    """LEGACY approximate CI kernel — physically wrong, opt-in only.

    This is the pre-fix assembly: ``_diagonal_element`` (a
    configuration-average diagonal) plus the ``_diagonal_J_correction`` /
    ``_off_diagonal_J_correction`` patches for the one- and two-electron
    differences.  It is *not* a representation of the Hamiltonian — Be
    ``correlation="sd"`` came out 0.39 Ha below the exact ground state, and
    it returns non-zero elements between different total J — so
    :func:`build_ci_matrix` does not fall back to it.  It survives because
    the diagnostics in `docs/archive/2026-09/CORRELATION_SD_RESEARCH_2026-09-20.md`
    (and anyone re-measuring the old numbers) need the old assembly
    reachable.

    The ``_legacy_`` name and the explicit ``allow_legacy_incorrect_physics``
    flag are the guard (critical review F4): an agent or grepping user must
    read this docstring — which documents the measured defects — before the
    kernel will run.  Production and test paths never pass the flag.
    """
    if not allow_legacy_incorrect_physics:
        raise RuntimeError(
            "_legacy_build_ci_matrix_approximate is NOT a Hamiltonian: "
            "Be correlation='sd' came out 0.39 Ha below the exact ground "
            "state and it produces non-zero inter-J elements. Pass "
            "allow_legacy_incorrect_physics=True only to re-measure those "
            "legacy diagnostics (docs/archive/2026-09/CORRELATION_SD_RESEARCH_2026-09-20.md)."
        )
    # ``_reset_memo`` drops every memoised integral, including the CI's
    # stashed rkb cache and the orbital-independent h0 blocks.  The
    # entry-point reset carries the rkb cache (as before) and, when the
    # workspace and nuclear model are unchanged, the h0 blocks too.
    _reset_memo_for_ci_build(bsp_data, nuclear_potential)
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
        if convol_cache is not None and convol_cache.same_grid(bsp_data):
            _SLATER_MEMO["_convol_cache"] = convol_cache
        _SLATER_MEMO["_nuclear_potential"] = nuclear_potential
    _check_orbitals_present(csfs, orbitals_dict)
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


def _subshell_tm(key):
    """Doubled-m values of a ``(n, kappa)`` subshell, ascending."""
    tj = 2 * abs(key[1]) - 1
    return list(range(-tj, tj + 1, 2))


def _two_electron_state(pattern, tJ, order):
    """m-resolved J-coupled two-electron state as ``{(i, j): coefficient}``.

    ``pattern`` is ``[(key, 2)]`` (both electrons in one subshell) or
    ``[(key_c, 1), (key_d, 1)]`` (one electron in each of two subshells);
    ``order`` maps ``(key, tm)`` to a global spin-orbital index that fixes the
    ascending-order sign convention (shared by both states of an element, so
    relative signs are consistent).  States are built at ``M = 0``, which is
    sufficient because a scalar two-body element is M-independent.

    The one-subshell case uses the antisymmetrised combination
    ``cg(m_i, m_j) - cg(m_j, m_i)`` over ascending pairs; the two-subshell case
    uses ``cg`` in the ``(c, d)`` slot order with the sign flipped whenever the
    ascending pair reverses that order.
    """
    vec: dict = {}
    if len(pattern) == 1:
        key = pattern[0][0]
        tj = 2 * abs(key[1]) - 1
        tms = _subshell_tm(key)
        for x, tm_i in enumerate(tms):
            for tm_j in tms[x + 1:]:
                if tm_i + tm_j != 0:
                    continue
                c = (cg(tj, tm_i, tj, tm_j, tJ, 0)
                     - cg(tj, tm_j, tj, tm_i, tJ, 0))
                if c != 0.0:
                    vec[(order[(key, tm_i)], order[(key, tm_j)])] = (
                        vec.get((order[(key, tm_i)], order[(key, tm_j)]), 0.0) + c)
    else:
        key_c, key_d = pattern[0][0], pattern[1][0]
        tjc, tjd = 2 * abs(key_c[1]) - 1, 2 * abs(key_d[1]) - 1
        for tm_i in _subshell_tm(key_c):
            for tm_j in _subshell_tm(key_d):
                if tm_i + tm_j != 0:
                    continue
                c = cg(tjc, tm_i, tjd, tm_j, tJ, 0)
                if c == 0.0:
                    continue
                i, j = order[(key_c, tm_i)], order[(key_d, tm_j)]
                pair = (i, j) if i < j else (j, i)
                vec[pair] = vec.get(pair, 0.0) + (c if i < j else -c)
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm < 1e-14:
        return {}                      # J not allowed for this pattern
    return {k: v / norm for k, v in vec.items()}


def _pattern_occupation_ok(occupations, pattern):
    """True when every pattern subshell holds exactly the pattern's electrons.

    ``occupations`` maps ``(n, kappa) -> q`` for the CSF on one side of the
    element and ``pattern`` is that side's ``[(key, count)]`` active-pair
    pattern.  The m-scheme two-electron state accounts for *every* electron
    in the pattern subshells, so a subshell holding more than ``count``
    electrons would silently drop the extras (HYP-1773: e.g. bra c³d¹ vs ket
    c²b² built a c¹d¹ pair state and lost the third c electron).

    There is deliberately no closed-subshell exception.  A fully occupied
    subshell *is* a pure J = 0 pair condensate (pair-removal cfp of unit
    magnitude, so the extra pair factorises as a scalar spectator), but the
    element then still depends on the *opposite* side's residual-subshell
    parentage: e.g. c⁴ ↔ c²b² at J = 0 has the two rows [(c²)₀(b²)₀] and
    [(c²)₂(b²)₂] with genuinely different elements, and occupation-only
    CSFs carry no parentage label to say which row is meant (review of
    PR #81: oracle gives +0.120679 Ha for the first row and −0.141562 Ha
    for the second on the same probe data).  The channel therefore declines
    every over-occupied pattern conservatively.
    """
    for key, count in pattern:
        q = occupations.get(key, 0.0)
        if abs(q - count) > 1e-9:
            return False
    return True


def _two_electron_replacement_element(csf_r, csf_s, pattern_r, pattern_s,
                                      orbitals_dict, bsp_data):
    """⟨pattern_r| 1/r₁₂ |pattern_s⟩ for two electrons outside closed spectators.

    Both patterns hold exactly two electrons, in one subshell (``[(key, 2)]``,
    the a² ↔ b² pair channel) or in two different subshells (``[(c,1),(d,1)]``,
    the a² ↔ c¹d¹ channel and its transpose).  With every other subshell closed
    the spectators couple to J = 0, so the two electrons are a pure J-coupled
    two-electron state with ``J`` equal to the CSF's total J, and the element
    is exact — no configuration average is involved.

    Implementation: the two J-coupled states are expanded in the m-scheme
    (Clebsch-Gordan coefficients, ascending-order signs) and contracted with
    the antisymmetrised two-body elements

        ⟨i j|g|k l⟩ = Σ_t R^t(i k, j l) · ckm·ckm · phase
                      − (k ↔ l)

    exactly as the FAC-ported determinant code does.  The radial integrals
    ``R^t`` depend only on the subshells (not on m), so each distinct
    (i,k,j,l,t) combination is evaluated once through the existing cell
    integrator and reused for every m pair.

    Guards (conservative): the CSF pair must be in the same (J, parity) block
    (enforced upstream), every spectator subshell must be closed, and each
    pattern subshell must hold exactly the pattern's electrons in its side's
    CSF (see :func:`_pattern_occupation_ok`) — otherwise the two active
    electrons are not in a definite J state and the configuration-average
    machinery would have to supply the parentage.  Spectator occupations
    are identical on both sides (only pattern subshells differ), so
    checking them on ``csf_r`` covers ``csf_s`` as well.
    """
    occ_r = {(sh.n, sh.kappa): sh.occupation for sh in csf_r.subshells}
    occ_s = {(sh.n, sh.kappa): sh.occupation for sh in csf_s.subshells}
    if not (_pattern_occupation_ok(occ_r, pattern_r)
            and _pattern_occupation_ok(occ_s, pattern_s)):
        return 0.0

    if any(abs(sh.occupation - (2 * abs(sh.kappa))) > 1e-9
           for sh in csf_r.subshells
           if (sh.n, sh.kappa) not in {p[0] for p in pattern_r + pattern_s}):
        return 0.0

    keys = sorted({p[0] for p in pattern_r + pattern_s})
    order = {(key, tm): idx
             for idx, (key, tm) in enumerate(
                 (key, tm) for key in keys for tm in _subshell_tm(key))}

    tJ = int(getattr(csf_r, "total_J2", 0))
    state_r = _two_electron_state(pattern_r, tJ, order)
    state_s = _two_electron_state(pattern_s, tJ, order)
    if not state_r or not state_s:
        return 0.0

    ns = len(list(orbitals_dict.values())[0][0]) if orbitals_dict else 50
    pq = {key: _get_orb(key, orbitals_dict, ns) for key in keys}
    so = {idx: (key, tm) for (key, tm), idx in order.items()}

    cache: dict = {}

    def r_table(key_i, key_j, key_k, key_l, t):
        """R^t(i k, j l) — subshell-resolved, so m-independent."""
        ck = (key_i, key_j, key_k, key_l, t)
        val = cache.get(ck)
        if val is None:
            Pi, Qi = pq[key_i]
            Pj, Qj = pq[key_j]
            Pk, Qk = pq[key_k]
            Pl, Ql = pq[key_l]
            # _rk pairs its first two orbitals and its last two, so pass
            # (i, k, j, l) to obtain R^t(i k, j l).
            val = _rk(Pi, Qi, Pk, Qk, Pj, Qj, Pl, Ql, bsp_data, t)
            cache[ck] = val
        return val

    def v_element(idx_i, idx_j, idx_k, idx_l):
        """⟨i(1) j(2)| 1/r₁₂ |k(1) l(2)⟩ (m-resolved)."""
        key_i, tm_i = so[idx_i]
        key_j, tm_j = so[idx_j]
        key_k, tm_k = so[idx_k]
        key_l, tm_l = so[idx_l]
        tq = tm_i - tm_k
        if tq != tm_l - tm_j:
            return 0.0
        total = 0.0
        for t in exchange_k_range(key_i[1], key_k[1]):
            if t not in exchange_k_range(key_j[1], key_l[1]):
                continue
            if abs(tq) > 2 * t:
                continue
            # slot order matches atomx.scattering...mscheme_ci._v exactly:
            # ckm(kappa_a, tm_a, kappa_c, tm_c, k, tq) for the (a -> c) pair
            e1 = ckm(key_i[1], tm_i, key_k[1], tm_k, t, tq)
            if e1 == 0.0:
                continue
            e2 = ckm(key_j[1], tm_j, key_l[1], tm_l, t, -tq)
            if e2 == 0.0:
                continue
            phase = 1.0 if (tq // 2) % 2 == 0 else -1.0
            total += (r_table(key_i, key_j, key_k, key_l, t) * phase * e1 * e2)
        return total

    total = 0.0
    for (i, j), c_r in state_r.items():
        for (k, l), c_s in state_s.items():
            total += c_r * c_s * (v_element(i, j, k, l) - v_element(i, j, l, k))
    return total


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


class MissingOrbitalError(ValueError):
    """A CI element needed a radial orbital the SCF never solved.

    The old behaviour zero-filled the absent orbital, which silently
    deleted every matrix element involving it: a multi-CSF run whose
    correlation orbital was not part of the radial set came back as the
    single-configuration energy (the review's Be ``["1s2", "1s2 2p2"]``
    measurement: E(2 CSF) == E(1 CSF)) with no diagnostic at all.  A CI
    spanning orbitals that were never solved is a broken calculation, so
    it now raises instead.
    """


def _get_orb(key, orbitals_dict, ns):
    """Get orbital ``(P, Q)``; raise if the CI needs an unsolved orbital.

    Two kinds of caller reach this helper:

    * **production** — everything under :func:`build_ci_matrix`, where the
      radial workspace is populated by the SCF and a missing key means an
      orbital the CSF list needs was never solved.  Zero-filling it (the
      previous behaviour) turned every element involving it into a silent
      zero — see :class:`MissingOrbitalError` — so it raises.
    * **angular-only unit-radial probes** — the J-coupling scaffolding is
      exercised directly with no radial data at all, e.g.
      ``_off_diagonal_J_correction(alpha, beta, orbitals_dict={},
      bsp_data={}, ns=1)`` in
      ``tests/test_off_diagonal_j_ci_helike_1s2p.py``, which substitutes
      ``radial = 1.0`` for every Slater integral.  An *entirely empty*
      map therefore means "no radial data supplied" and yields zeros, so
      the angular body stays independently testable.  Production never
      sees an empty map: :func:`_check_orbitals_present` rejects it at
      the entry point.
    """
    if key in orbitals_dict:
        return orbitals_dict[key]
    if not orbitals_dict:
        return np.zeros(ns), np.zeros(ns)
    raise MissingOrbitalError(
        f"CI matrix needs orbital (n={key[0]}, kappa={key[1]}), which was "
        f"not solved; available orbitals: {sorted(orbitals_dict)}. "
        "Include every orbital used by the CSF list in the radial "
        "optimisation (the MCDHF driver solves all of them when they "
        "appear in `configs`).")


def _check_orbitals_present(csfs, orbitals_dict) -> None:
    """Fail up front if any CSF needs an orbital absent from the workspace.

    Cheaper and far clearer than discovering it element-by-element inside
    :func:`_get_orb`: reports every missing ``(n, kappa)`` at once.  This
    is the *production* gate — an empty ``orbitals_dict`` (which
    :func:`_get_orb` tolerates for the angular-only unit-radial probes)
    is rejected here, because a CI build with no radial data at all is
    exactly the silent-zero failure this guard exists to stop.
    """
    missing = sorted({
        (sh.n, sh.kappa)
        for csf in csfs
        for sh in csf.subshells
        if (sh.n, sh.kappa) not in orbitals_dict
    })
    if missing:
        raise MissingOrbitalError(
            f"CI build needs {len(missing)} orbital(s) that were never "
            f"solved: {missing}; available: {sorted(orbitals_dict)}. "
            "The CSF list must not reference orbitals outside the radial "
            "set — otherwise their matrix elements silently vanish and the "
            "CI returns the uncorrelated energy.")


# Cadence of the orbital-metric tripwire (see _metric_check_due).
# 1 = check on every CI build (the historical behaviour); N > 1 = check on
# the first build of each radial workspace and then every N builds of the
# same workspace; 0 = first build of each workspace only.
_METRIC_CHECK_EVERY = int(os.environ.get("ATOMX_CI_METRIC_CHECK_EVERY", "20"))

# Per-workspace state for _metric_check_due: the last workspace object the
# cadence was evaluated against and how many builds of it have passed.
_METRIC_CHECK_STATE = {"context": None, "count": 0}


def _metric_check_due(workspace) -> bool:
    """Decide whether this CI build pays the orbital-metric audit.

    The audit is a tripwire, not a calculation input: it guards against a
    solver handing the CI a non-orthonormal orbital set, and a persistent
    violation is caught by the *next* audit of the same workspace.  Yet it
    cost ~27 ms per call and ran on every build_ci_matrix call — ~20 % of
    a variational MCDHF solve is the line-search rebuilding the same Gram
    matrices for near-identical trial orbitals (measured; see
    docs/HPC_NUMERICS_RESEARCH_2026-09-27.md §2.3).  The first build of
    every new radial workspace is always audited (a freshly solved orbital
    set is exactly where a violation can appear); subsequent builds of the
    same workspace are audited every ``_METRIC_CHECK_EVERY`` builds.
    Set ``ATOMX_CI_METRIC_CHECK_EVERY=1`` to restore per-build auditing.
    """
    state = _METRIC_CHECK_STATE
    if state["context"] is not workspace:
        state["context"] = workspace
        state["count"] = 0
        return True
    state["count"] += 1
    if _METRIC_CHECK_EVERY <= 0:
        return False
    if _METRIC_CHECK_EVERY == 1:
        return True
    return state["count"] % _METRIC_CHECK_EVERY == 0


def _check_orbital_metric(csfs, orbitals, workspace, Z, ns, tolerance=1e-7):
    """Slater–Condon projection requires orthonormal radial spinors.

    Validate the actual integration metric rather than assuming that
    individually normalized eigenvectors of different Fock operators are
    mutually orthogonal. Nonorthogonal CI requires an overlap-aware
    determinant formulation and cannot use this kernel.
    """
    groups = {}
    for n, kappa in sorted({(sh.n, sh.kappa) for csf in csfs for sh in csf.subshells}):
        groups.setdefault(kappa, []).append((n, kappa))
    for kappa, keys in groups.items():
        if isinstance(workspace, GridCIWorkspace):
            gram = np.empty((len(keys), len(keys)))
            for i, a in enumerate(keys):
                pa, qa = orbitals[a]
                for j, b in enumerate(keys):
                    pb, qb = orbitals[b]
                    gram[i, j] = slag6(1.0, (pa * pb + qa * qb) * workspace.difr, ns)[-1]
        else:
            _, metric = _dirac_blocks(ns, kappa, Z, workspace)
            coefficients = np.column_stack([np.concatenate(orbitals[key]) for key in keys])
            gram = coefficients.T @ metric @ coefficients
        defect = float(np.max(np.abs(gram - np.eye(len(keys)))))
        if not np.isfinite(defect) or defect > tolerance:
            raise ValueError(
                f"CI requires orthonormal orbitals; kappa={kappa}, orbitals={keys}, "
                f"max |overlap - identity|={defect:.3e} exceeds {tolerance:.0e}. "
                "Solve with orbital constraints before constructing the CI Hamiltonian.")


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


def _pair_pattern(side):
    """Occupancy pattern of the two active electrons, or None if not a pair.

    ``side`` is the ``(key, q_bra, q_ket)`` list of subshells that gained or
    lost electrons between the two CSFs.  Returns ``[(key, 2)]`` when a single
    subshell changed by ±2 and ``[(key_c, 1), (key_d, 1)]`` when two subshells
    changed by ±1 each (i.e. the two electrons arrived/left together); returns
    None for any other pattern, which the caller treats as "not this channel".
    """
    if len(side) == 1:
        key, q_bra, q_ket = side[0]
        if abs(q_bra - q_ket) == 2:
            return [(key, 2)]
        return None
    if len(side) == 2:
        for _key, q_bra, q_ket in side:
            if abs(q_bra - q_ket) != 1:
                return None
        return [(side[0][0], 1), (side[1][0], 1)]
    return None


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

    # Two-electron replacements where one subshell changes by two electrons
    # leave the two active electrons in a *definite* J-coupled state (with
    # closed spectators the pair's J is the CSF's total J), so the element is
    # exact rather than a configuration average.  Three patterns reach here:
    #
    #   a² ↔ b²        one key loses 2, one gains 2   (2s² ↔ 2p², the
    #                                                  6p½² ↔ 6p3/2² jj pair)
    #   a² ↔ c¹ d¹     one key loses 2, two gain 1    (double excitation into
    #                                                  two different orbitals)
    #   a¹ b¹ ↔ c²     two lose 1, one gains 2        (the transpose)
    #
    # All three go through the general two-electron element, which handles any
    # total J (the J = 0 case was implemented first; J ≠ 0 needs the same
    # machinery, not a special case).
    pair_r = _pair_pattern(lost)
    pair_s = _pair_pattern(gained)
    if pair_r is not None and pair_s is not None:
        return _two_electron_replacement_element(
            csf_r, csf_s, pair_r, pair_s, orbitals_dict, bsp_data)

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
        for idx, (state, M2) in enumerate(zip(chain.subshells, projections, strict=False)):
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
        radial = _rk(Pa, Qa, Pc, Qc, Pb, Qb, Pd, Qd, bsp_data, k) if bsp_data else 1.0
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
        radial = _rk(Pa, Qa, Pd, Qd, Pb, Qb, Pc, Qc, bsp_data, k) if bsp_data else 1.0
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
    """J-coupled off-diagonal CI correction (Grant §8.3) — LANDED, UNVALIDATED.

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

    **Current status — body landed, correctness not established.**  This
    docstring previously said the function "returns ``0.0``", which is
    no longer true and was actively misleading: the body was landed
    without updating the contract text.  What the body does today:

    * it returns a populated correction for the unique-parentage,
      same-``(J, parity)`` envelope described above;
    * that correction is **quantitatively wrong**.  For the He-like
      1s2p J=1 anchor it contributes ``+0.431282`` on a base
      :func:`_one_diff_element` of ``-0.193857``, moving the J=1 level
      to -14.711459 Ha where an independent m-scheme Slater-Condon CI
      on the *same orbitals* gives -14.490933 Ha.  The "verify against
      a He-like 1s2p analytic test" step below was therefore never
      satisfied;
    * outside the envelope nothing is returned, so the diagonal and
      off-diagonal elements are drawn from two different physical
      approximations depending on the CSF shape.

    The structural reason an exact element is not available here is
    that :class:`~atomx.config.csf.CSF` carries only
    ``total_J2`` and an occupation list — no coupling tree (parentage) —
    so the Grant §8.3 recoupling chain cannot be inferred from the
    objects this function receives.

    Measured consequences, the independent oracle used to measure them,
    and the fix options are in
    ``docs/archive/2026-09/CORRELATION_SD_RESEARCH_2026-09-20.md``; the regression gates
    are in ``tests/test_ci_matrix_vs_mscheme_reference.py``.

    Any future body must satisfy, at minimum:

    1. Identifying the unique-parentage subset (each open subshell is
       a single electron or single hole) on **both** ``csf_r`` and
       ``csf_s`` and the matching same-(J,parity) constraint.
    2. Selecting the spectator/source/destination shells and reading
       off the ``(j_a, j_b, j_c, j_d)`` labels and total ``J``.
    3. Evaluating the Slater integrals ``R^K`` over the radial part
       using the module-level ``_rk`` (cell-algorithm integrator).
    4. Composing the 6j-weighted angular factor and accumulating the
       additive correction.
    5. Verifying against the He-like 1s2p J=1 anchor: the two CSFs
       ``|1s 2p_{1/2}; J=1⟩`` and ``|1s 2p_{3/2}; J=1⟩`` must
       diagonalise to the singlet/triplet states with the
       ``2·G¹(1s,2p)`` splitting **and** agree with
       ``benchmarks/mscheme_ci_reference.py`` on the same orbitals.

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
        the 1-diff vs 2-diff sub-formula; not used when the envelope
        check above rejects the pair.

    Returns
    -------
    float
        Additive correction to be added to the base 1-diff / 2-diff
        matrix element, or ``0.0`` when the pair is outside the
        unique-parentage same-``(J, parity)`` envelope.  The non-zero
        branch is **not yet correct** — see the status note above.
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
