"""CI Hamiltonian construction from exact jj-coupled CSFs.

One- and two-electron operators connect determinants differing by at most
two electron replacements. The angular kernel applies the Slater–Condon
rules before projection onto the coupled CSF basis.

The production builder contracts exact angular coefficients in each
J/parity block with one-body Dirac and two-body radial integrals. CSF
parentage and fermion phases belong to ``angular.csf_hamiltonian``;
radial integration uses the supplied grid or B-spline representation.
"""

import hashlib
from functools import lru_cache

import numpy as np
from scipy.sparse.linalg import LinearOperator

from atomx.angular.csf_hamiltonian import (
    MAX_BLOCK_DETERMINANTS,
    csf_signature,
    exact_block_matrix,
)
from atomx.numerics.bspline.convol import (
    SEMISEPARABLE_DIRECT_NS,
    ConvolCache,
    _convol_case1_semiseparable_t,
    _convol_case1_t,
    _density_sym,
    _sum_amb_banded,
    compute_slater_fk_cell,
    compute_slater_gk_cell,
    compute_slater_rk_cell,
)
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices
from atomx.numerics.shooting.constants import SL
from atomx.numerics.shooting.coulomb import _slater_integral_slag6, _yk_over_r
from atomx.numerics.shooting.operators import _ds_derivative
from atomx.numerics.shooting.quadrature import slag6, slag6_weights

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
_MEMO_GEOMETRY = None


def _reset_memo() -> None:
    """Drop all memoised integrals and the references pinning their keys."""
    global _MEMO_CONTEXT, _MEMO_GEOMETRY
    _SLATER_MEMO.clear()
    _MEMO_KEEPALIVE.clear()
    _MEMO_CONTEXT = None
    _MEMO_GEOMETRY = None


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
    global _MEMO_GEOMETRY
    geometry = _workspace_fingerprint(bsp_data)
    if nuclear_potential is not None and not isinstance(bsp_data, GridCIWorkspace):
        # A caller can update parameters captured by the same callable.
        # H0 depends on its values on the actual integration grid, not
        # callable identity alone. These are the nuclear_matrix inputs.
        values = np.fromiter(
            (nuclear_potential(float(radius)) for radius in bsp_data["gr"].ravel()),
            dtype=np.float64, count=bsp_data["gr"].size)
        geometry += hashlib.sha256(
            repr((values.shape, values.dtype.str)).encode() + values.tobytes()
        ).digest()
    stashed_cache = _SLATER_MEMO.get("_convol_cache")
    prev_potential = _SLATER_MEMO.get("_nuclear_potential")
    carry_h0 = (
        not isinstance(bsp_data, GridCIWorkspace)
        and _MEMO_CONTEXT is bsp_data
        and geometry == _MEMO_GEOMETRY
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
    _MEMO_GEOMETRY = geometry
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

    Standalone helper calls (unit tests poking the radial helpers without
    going through :func:`build_ci_matrix`) get a fresh cache bound
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
            np.ravel(gr)
            np.ravel(grw)

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
                    *, breit_orbital_energies=None, one_body_shifts=None,
                    radial_snapshot=None, one_body_operator=None,
                    angular_storage='cached'):
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
    radial_snapshot : CIRadialSnapshot or None
        Reuse Dirac-Coulomb radial values across CI spaces on exactly the same
        B-spline orbital/Hamiltonian snapshot. Mismatches raise. One-body
        shifts and Breit remain independently applied per matrix build.
    one_body_operator : CIEffectiveOneBody or None
        Bound signed scalar matrix added to the one-body radial keys. Requires
        the producer's orbital/core/nuclear reference and filled correlated
        core in every CSF. Cannot be combined with legacy one_body_shifts.
    angular_storage : {'cached', 'streamed'}
        Reuse the cached angular coefficient table (default), or contract
        identical signed terms directly into numeric sparse determinant H.
        Streamed storage keeps a build-local radial-value cache and the
        complete physical CSF space; orbital-density tables remain separate.
    """
    if breit_orbital_energies is not None and not include_breit:
        raise ValueError("breit_orbital_energies requires include_breit=True")
    effective_evaluate = None
    if angular_storage not in ('cached', 'streamed'):
        raise ValueError("angular_storage must be 'cached' or 'streamed'")
    if one_body_operator is not None:
        from atomx.physics.csf.effective_one_body import CIEffectiveOneBody
        if not isinstance(one_body_operator, CIEffectiveOneBody):
            raise TypeError("one_body_operator must be CIEffectiveOneBody")
        if one_body_shifts is not None:
            raise ValueError("cannot combine effective operator and legacy one-body shifts")
        effective_evaluate = one_body_operator.evaluator_for(
            csfs, orbitals_dict, bsp_data, Z, ns, nuclear_potential=nuclear_potential)
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
    metric_fp = _orbital_metric_fingerprint(
        orbitals_dict, _csf_orbital_keys(csfs), Z, ns, bsp_data)
    if _metric_check_due(bsp_data, metric_fp):
        _check_orbital_metric(csfs, orbitals_dict, bsp_data, Z, ns)
        _metric_check_record(bsp_data, metric_fp)
    n_csf = len(csfs)
    if n_csf == 0:
        if not as_operator:
            return np.zeros((0, 0))
        # Honour the as_operator contract: same LinearOperator shape and
        # block-structure attributes as the non-empty path below.
        def apply_empty(v):
            return np.zeros_like(v, dtype=float)

        H = LinearOperator((0, 0), matvec=apply_empty, matmat=apply_empty,
                           rmatvec=apply_empty, dtype=float)
        H.diagonal = lambda: np.zeros(0)
        H.blocks = []
        return H
    H = None if as_operator else np.zeros((n_csf, n_csf))
    if max_determinants is None:
        max_determinants = MAX_BLOCK_DETERMINANTS
    if radial_snapshot is None:
        evaluate = _radial_evaluator(orbitals_dict, bsp_data, Z, ns,
                                    nuclear_potential=nuclear_potential)
    else:
        from atomx.physics.csf.radial_snapshot import CIRadialSnapshot
        if not isinstance(radial_snapshot, CIRadialSnapshot):
            raise TypeError("radial_snapshot must be CIRadialSnapshot")
        evaluate = radial_snapshot.evaluator_for(orbitals_dict, bsp_data, Z, ns,
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

    if effective_evaluate is not None:
        base_with_effective = evaluate

        def evaluate(key):
            return base_with_effective(key) + effective_evaluate(key)

    breit_evaluate = (_breit_radial_evaluator(orbitals_dict, bsp_data, Z, ns,
                                           orbital_energies=breit_orbital_energies)
                      if include_breit else None)
    operator_blocks = []
    for _key, idx in _group_csf_indices(csfs):
        signatures = [csf_signature(csfs[i]) for i in idx]
        h_block = exact_block_matrix(signatures, evaluate,
                                     max_determinants=max_determinants,
                                     breit_evaluator=breit_evaluate,
                                     as_operator=as_operator,
                                     angular_storage=angular_storage)
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
    matrix values.  Every CSF must carry explicit ``total_J2`` and
    ``parity`` attributes (both are declared fields of
    :class:`atomx.config.csf.CSF`); a CSF lacking either raises rather
    than being silently misassigned to the ``J = 0``, even-parity block.
    """
    groups: dict = {}
    for i, csf in enumerate(csfs):
        key = (int(csf.total_J2), int(csf.parity))
        groups.setdefault(key, []).append(i)
    return sorted(groups.items(), key=lambda kv: (kv[0][0], -kv[0][1]))


def _bspline_pair_potential(cache, ns, orders, rank, source):
    """The ordinary CI cell operator on one explicit pair-density tuple."""
    potentials = []
    for left, label_left in enumerate(("p", "q")):
        value = np.zeros((ns, orders[left]))
        for right, label_right in enumerate(("p", "q")):
            components = label_left + label_right + label_left + label_right
            if ns >= SEMISEPARABLE_DIRECT_NS:
                factors = cache.get_coulomb_factors(rank, components)
                contracted = _convol_case1_semiseparable_t(
                    *factors, source[right], ns, orders[left], orders[right])
            else:
                bank, extent, _ = cache.get_rkb_t(rank, components)
                contracted = _convol_case1_t(
                    bank, source[right], extent, orders[left], orders[right])
            value += contracted
        potentials.append(value)
    return tuple(potentials)


def _radial_evaluator(orbitals_dict, bsp_data, Z, ns, *, nuclear_potential=None,
                      pair_potential_cache_size=None):
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
    pair_capacity = (256 if is_grid else 1024) if pair_potential_cache_size is None else pair_potential_cache_size
    if not isinstance(pair_capacity, int) or isinstance(pair_capacity, bool) or pair_capacity < 1:
        raise ValueError('pair potential cache capacity must be a positive integer')
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
            pa, qa = _get_orb(a, orbitals_dict)
            pb, qb = _get_orb(b, orbitals_dict)
            return pa*pb+qa*qb
        return tuple(_density_sym(x, y, ns, order)
                     for x, y, order in zip(orbitals_dict[a], orbitals_dict[b], orders, strict=True))

    @lru_cache(maxsize=pair_capacity)
    def pair_potential(rank, a, b):
        source = density(a, b)
        if is_grid:
            return weights * _yk_over_r(source, rank, bsp_data.r,
                                        bsp_data.difr, bsp_data.n)
        return _bspline_pair_potential(cache, ns, orders, rank, source)

    @lru_cache(maxsize=8192)
    def evaluate(key):
        if key[0] == "h":
            _, n_a, kappa, n_c = key
            Pa, Qa = _get_orb((n_a, kappa), orbitals_dict)
            Pc, Qc = _get_orb((n_c, kappa), orbitals_dict)
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

    evaluate.pair_cache_info = pair_potential.cache_info
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
            P, Q = _get_orb((n, kappa), orbitals_dict)
            coeffs.append(P if comp == "P" else Q)
            comps.append(comp)
        return _breit_rk(coeffs[0], coeffs[2], coeffs[1], coeffs[3],
                         (comps[0], comps[2], comps[1], comps[3]),
                         bsp_data, L)

    return evaluate


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


def _get_orb(key, orbitals_dict):
    """Get orbital ``(P, Q)``; raise if the CI needs an unsolved orbital.

    Everything under :func:`build_ci_matrix` reaches this helper with the
    radial workspace populated by the SCF, so a missing key means an
    orbital the CSF list needs was never solved.  Zero-filling it (the
    pre-:class:`MissingOrbitalError` behaviour) turned every element
    involving it into a silent zero, so it raises.
    """
    if key in orbitals_dict:
        return orbitals_dict[key]
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
    is the *production* gate — an empty ``orbitals_dict`` is rejected here,
    because a CI build with no radial data at all is exactly the
    silent-zero failure this guard exists to stop.
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


# Per-workspace validation memo for the orbital-metric tripwire.  The audit
# runs on the first build of every radial workspace and again whenever the
# CONTENT of the audited orbitals changes; it is skipped only when the
# byte-identical orbital content was already validated in the same workspace
# context.  There is no cadence and no environment knob: whether the audit
# ran is a function of the inputs, not of build history.  The memo is
# written only after a *passed* audit, so a rejected orbital set is
# re-audited on every subsequent build attempt.
_METRIC_CHECK_STATE = {"context": None, "fingerprint": None}


def _csf_orbital_keys(csfs):
    """Sorted ``(n, kappa)`` subshell keys the CSF list references."""
    return sorted({(sh.n, sh.kappa) for csf in csfs for sh in csf.subshells})


def _workspace_fingerprint(workspace) -> bytes:
    """Bind one-body reuse and metric validation to actual radial inputs."""
    if isinstance(workspace, GridCIWorkspace):
        values = ((key, getattr(workspace, key))
                  for key in ("r", "difr", "rv_nuc", "n"))
    else:
        # Complete arrays and layout consumed by build_dirac_matrices.
        values = ((key, workspace.get(key)) for key in (
            "nv", "ks", "ksp", "ksq", "ns", "nsp", "nsq", "gr", "grw",
            "bsp_p", "bsp_q", "bsp_p_d", "bsp_q_d"))
    digest = hashlib.sha256()
    for key, value in values:
        digest.update(key.encode())
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(repr((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.digest()


def _orbital_metric_fingerprint(orbitals, keys, Z, ns, workspace=None) -> bytes:
    """Content hash of everything the orbital-metric audit reads.

    The id()-keyed integral memo cannot serve here: it answers "same
    object", and the audit must re-run when a solver writes new
    coefficients into the same arrays (the MCDHF line search does exactly
    that).  The digest covers ``(Z, ns)`` — which the Gram blocks depend
    on — and, per audited ``(n, kappa)`` key, the byte content of the
    orbital's ``(P, Q)`` arrays and the radial representation. Reuse requires
    the same input content; no timing claim is attached to this audit gate.
    """
    h = hashlib.sha256()
    h.update(repr((Z, ns)).encode())
    if workspace is not None:
        h.update(_workspace_fingerprint(workspace))
    for key in keys:
        h.update(repr(key).encode())
        for component in orbitals[key]:
            arr = np.ascontiguousarray(component, dtype=np.float64)
            h.update(repr(arr.shape).encode())
            h.update(arr.tobytes())
    return h.digest()


def _metric_check_due(workspace, fingerprint) -> bool:
    """Whether this build's orbital content still needs the metric audit.

    Pure query: ``True`` unless the identical workspace object and the
    identical content fingerprint were recorded by
    :func:`_metric_check_record` after a passed audit.  The first build of
    a workspace and any content change are always due.  The audit is a
    tripwire, not a calculation input: it guards against a solver handing
    the CI a non-orthonormal orbital set.
    """
    state = _METRIC_CHECK_STATE
    return not (state["context"] is workspace
                and state["fingerprint"] == fingerprint)


def _metric_check_record(workspace, fingerprint) -> None:
    """Record a passed audit; call only after ``_check_orbital_metric``."""
    _METRIC_CHECK_STATE["context"] = workspace
    _METRIC_CHECK_STATE["fingerprint"] = fingerprint


def _check_orbital_metric(csfs, orbitals, workspace, Z, ns, tolerance=1e-7):
    """Slater–Condon projection requires orthonormal radial spinors.

    Validate the actual integration metric rather than assuming that
    individually normalized eigenvectors of different Fock operators are
    mutually orthogonal. Nonorthogonal CI requires an overlap-aware
    determinant formulation and cannot use this kernel.
    """
    groups = {}
    for n, kappa in _csf_orbital_keys(csfs):
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
