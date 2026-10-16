"""Exact jj-coupled CI matrix elements by m-scheme Slater-Condon projection.

Why this module exists
----------------------
:func:`atomx.physics.csf.matrix.build_ci_matrix` used to evaluate a
*configuration-average* diagonal plus two hand-rolled "J-correction"
patches.  That matrix is not a representation of the Hamiltonian, so the
CI was not variational: for Be ``correlation="sd"`` it returned
-15.05 Ha, 0.39 Ha below the exact non-relativistic ground state
(-14.66736 Ha).  See ``docs/archive/2026-09/CORRELATION_SD_RESEARCH_2026-09-20.md``.

A CSF in jj-coupling is a *coupling tree*, not just
``(subshells, total_J2)``, so the exact matrix element cannot be written
down from the old :class:`~atomx.config.csf.CSF` object.  This
module sidesteps parentage entirely, the way DIRAC's ``krci`` and Cowan's
code do: every CSF is expanded as an explicit vector over Slater
determinants, the Hamiltonian is built in that determinant basis by
Slater-Condon rules, and the CSF matrix is the projection

    H_CSF = C^T H_det C

with column ``r`` of ``C`` the determinant-space vector of CSF ``r``.
The expansion needs no coefficients of fractional parentage and is exact
for any CSF list: it is defined by the CSF vectors, whatever coupling
tree produced them.

Independence
------------
``benchmarks/mscheme_ci_reference.py`` implements the same construction
from scratch (sympy 3j angular algebra, brute-force second quantisation).
It is the oracle this module is gated against in
``tests/test_ci_matrix_vs_mscheme_reference.py``; the two share the
``_h_one_body`` / ``_norm_sq`` / ``_rk`` radial integrals and nothing
else.  The angular kernel here (:func:`two_body_angular`) reproduces the
oracle's formula with the repo's numeric
:func:`atomx.angular.wigner.wigner_3j` and
:func:`atomx.angular.coefficients.reduced_C_jj`.

Cost model
----------
Everything orbital-independent — determinant spaces, coupling trees, CSF
vectors, and the sparse Slater-Condon coefficient table — is a pure
function of integer quantum numbers and is cached with
:func:`functools.lru_cache`.  Rebuilding the numeric matrix for a new
orbital set (the MCDHF outer loop) is then a sparse mat-vec over the
distinct radial integrals of the block.

Determinism and the degenerate-parentage caveat
-----------------------------------------------
A fixed CSF list produces bit-identical matrices across processes.  One
caveat is *physics* basis-dependence rather than nondeterminism: the
highest-weight subshell states come from
:func:`atomx.angular.mscheme.highest_weight_states`, which returns an
arbitrary (SVD-derived) orthonormal basis inside a *degenerate* subshell
J — e.g. ``(7/2)^4`` has J = 2 and J = 4 twice each.  For such a subshell
the ``index`` field of a state is a basis index, not a seniority label,
so individual ``H_rs`` elements between CSFs that differ only in that
index depend on the basis choice; only the block eigenvalues and the
trace are basis-independent.  The basis is still deterministic for a
fixed input and the CSFs are orthonormal, so the energies are well
defined, but a state label such as "#2" for ``(7/2)^4`` must not be read
as a seniority quantum number.
"""

from __future__ import annotations

import itertools
import math
from array import array
from functools import lru_cache

import numpy as np
from scipy.sparse import block_diag, csr_matrix, eye
from scipy.sparse.linalg import LinearOperator

from atomx.angular.clebsch import clebsch_gordan
from atomx.angular.coefficients import reduced_C_jj
from atomx.angular.csf_recoupling import allowed_coupled_J2
from atomx.angular.mscheme import (
    _annihilate,
    _create,
    _jplus_apply,
    determinants_by_m,
    highest_weight_states,
    jplus_matrix,
    subshell_j_counts_mscheme,
)
from atomx.angular.recoupling import twoj_of_kappa
from atomx.angular.wigner import wigner_3j

__all__ = [
    "MAX_BLOCK_DETERMINANTS",
    "BreitBlockStructure",
    "CSFSignature",
    "ExactCISpaceError",
    "block_terms",
    "breit_block_terms",
    "canonical_coupling_tree",
    "coupling_paths",
    "csf_signature",
    "csf_magnetic_vector",
    "continuum_hamiltonian_terms",
    "exact_block_matrix",
    "two_body_angular",
]

#: Hard cap on the *raw* determinant dimension of one (2J, parity) block.
#:
#: The determinant space at ``2M = 2J`` contains the lower-M members of
#: every higher-J multiplet of the same configuration, so it is larger than
#: the CSF space (Be ``correlation="sd"``: 936 determinants over 399 CSFs,
#: the largest single block being 166 determinants for 62 CSFs; Co 3d7 4s2:
#: 55 determinants over 19 CSFs in 6 blocks; Fe XVII 2p5 3l: 37 over 37).
#: Common N-2 spectators enumerate connected pairs without an all-pairs scan.
#: Connected work and the cached sparse coefficient table can still approach
#: quadratic size, so the cap keeps a pathological configuration list from
#: hanging: :func:`exact_block_matrix` raises
#: :class:`ExactCISpaceError` naming the configuration and both sizes
#: instead.  2000 determinants is ~4e6 pairs — a few seconds and a few tens
#: of MB of cached table; callers with a justified larger case may override
#: it with ``max_determinants=``.
MAX_BLOCK_DETERMINANTS = 2000


class ExactCISpaceError(ValueError):
    """The exact CI route cannot build a requested (2J, parity) block.

    Raised when the raw determinant space of a block exceeds
    ``max_determinants``, when a CSF's ``total_J2`` is not reachable by any
    sequential coupling of its subshell angular momenta, or when the CSF
    list does not span the configuration's J-block (which would silently
    truncate the variational space).
    """


# ---------------------------------------------------------------------------
# Spinor conventions.  ``2m`` is stored as an integer, so all half-integer
# arithmetic stays exact.
# ---------------------------------------------------------------------------

def twoj(kappa: int) -> int:
    """``2j`` for a Dirac quantum number ``kappa``.

    Thin alias of the canonical :func:`atomx.angular.recoupling.twoj_of_kappa`;
    kept under this short name because the CI assembler and
    ``atomx.physics.csf.matrix`` call it throughout.
    """
    return twoj_of_kappa(int(kappa))


def _m2_values(tj: int) -> tuple[int, ...]:
    return tuple(range(-tj, tj + 1, 2))


def _mask_m2(mask: int, m2_of_bit: tuple[int, ...]) -> int:
    """Total ``2M`` of a bitmask whose bit ``i`` carries ``m2_of_bit[i]``."""
    total = 0
    while mask:
        low = mask & -mask
        total += m2_of_bit[low.bit_length() - 1]
        mask ^= low
    return total


def _bits(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


# ---------------------------------------------------------------------------
# Configuration layouts
# ---------------------------------------------------------------------------

class PatternLayout:
    """Spinor basis and determinant enumeration of one occupation pattern.

    The pattern is the canonical ``((n, kappa, q), ...)`` tuple sorted by
    ``(n, kappa)``; its spinors are laid out subshell by subshell with each
    subshell's ``2m`` values ascending, and each subshell's bits start at
    that subshell's offset.  Determinant masks are integers over that
    layout — the ascending-index convention
    :mod:`atomx.angular.mscheme` and ``benchmarks/mscheme_ci_reference.py``
    use, so creation/annihilation signs are the parity of the occupied
    lower indices in both.
    """

    __slots__ = ("pattern", "pattern_keys", "spinors", "offsets", "blocks",
                 "by_m2",)

    def __init__(self, pattern):
        self.pattern = tuple(sorted(tuple(p) for p in pattern))
        self.pattern_keys = tuple((n, k) for (n, k, _q) in self.pattern)
        spinors: list[tuple[int, int, int]] = []
        offsets: list[int] = []
        widths: list[int] = []
        twojs: list[int] = []
        off = 0
        for (n, kappa, q) in self.pattern:
            tj = twoj(kappa)
            ms = _m2_values(tj)
            if q < 0 or q > len(ms):
                raise ExactCISpaceError(
                    f"occupation {q} out of range for subshell "
                    f"(n={n}, kappa={kappa}) with 2j+1 = {len(ms)}")
            offsets.append(off)
            widths.append(len(ms))
            twojs.append(tj)
            spinors.extend((n, kappa, m) for m in ms)
            off += len(ms)
        self.spinors = tuple(spinors)
        self.offsets = tuple(offsets)
        self.blocks = tuple((offsets[i], widths[i], twojs[i])
                            for i in range(len(offsets)))

        self.by_m2 = _SectorMasks(self.pattern, self.offsets)

    @property
    def masks(self):
        """All sectors, materialized only for explicit diagnostic callers."""
        return tuple(sorted(m for sector, _ in pattern_m_counts(self.pattern)
                            for m in self.by_m2[sector]))

    @property
    def index(self):
        return {m: i for i, m in enumerate(self.masks)}


@lru_cache(maxsize=512)
def _subshell_m_counts(tj, q):
    """Count fixed-M determinants by integer dynamic programming."""
    if q < 0 or q > tj + 1:
        raise ExactCISpaceError(f"occupation {q} out of range for 2j={tj}")
    counts = {(0, 0): 1}
    for m in _m2_values(tj):
        updated = dict(counts)
        for (number, total), count in counts.items():
            if number < q:
                key = (number + 1, total + m)
                updated[key] = updated.get(key, 0) + count
        counts = updated
    return tuple(sorted((m, count) for (number, m), count in counts.items()
                        if number == q))


@lru_cache(maxsize=1024)
def pattern_m_counts(pattern):
    counts = {0: 1}
    for _, k, q in pattern:
        new = {}
        for m, count in counts.items():
            for shell_m, shell_count in _subshell_m_counts(twoj(k), q):
                key = m + shell_m
                new[key] = new.get(key, 0) + count * shell_count
        counts = new
    return tuple(sorted(counts.items()))


class _SectorMasks(dict):
    """Lazy fixed-M enumeration; never form the all-M Cartesian product."""
    def __init__(self, pattern, offsets):
        self.pattern = pattern
        self.offsets = offsets
        self.suffix = [dict(pattern_m_counts(pattern[i:])) for i in range(len(pattern) + 1)]

    def __missing__(self, M2):
        masks = []

        def visit(i, remaining, mask):
            if remaining not in self.suffix[i]:
                return
            if i == len(self.pattern):
                masks.append(mask)
                return
            _, k, q = self.pattern[i]
            for m, _ in _subshell_m_counts(twoj(k), q):
                if remaining - m not in self.suffix[i + 1]:
                    continue
                for local in determinants_by_m(twoj(k), q, m):
                    visit(i + 1, remaining - m, mask | (local << self.offsets[i]))

        visit(0, M2, 0)
        value = tuple(sorted(masks))
        self[M2] = value
        return value

    def get(self, key, default=None):
        return self[key] or default


@lru_cache(maxsize=256)
def pattern_layout(pattern) -> PatternLayout:
    """Cached :class:`PatternLayout` for an occupation pattern."""
    return PatternLayout(tuple(tuple(p) for p in pattern))


def _union_offsets(patterns: tuple) -> tuple:
    """``((n, kappa), offset)`` pairs of the shared spinor layout of a block.

    Every determinant of a block is expressed as a mask over this union
    layout, so pairs of determinants from *different* configurations are
    compared in one index convention, exactly as the oracle does when it
    enumerates determinants over ``sorted(orbitals_dict)``.
    """
    keys = sorted({(n, k) for p in patterns for (n, k, _q) in p})
    out = []
    off = 0
    for key in keys:
        out.append((key, off))
        off += len(_m2_values(twoj(key[1])))
    return tuple(out)


def _remap(mask: int, lay: PatternLayout, offsets: dict) -> int:
    """Move a pattern-local determinant mask into the block's union layout."""
    g = 0
    for (off, width, _tj), key in zip(lay.blocks, lay.pattern_keys,
                                      strict=True):
        g |= ((mask >> off) & ((1 << width) - 1)) << offsets[key]
    return g


# ---------------------------------------------------------------------------
# The J^2 operator in a pattern's determinant basis
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _jplus_block(pattern: tuple, M2: int) -> np.ndarray:
    """Total ``J_+`` matrix from the pattern's ``2M`` to its ``2M+2`` space."""
    lay = pattern_layout(pattern)
    ket = lay.by_m2.get(M2, ())
    bra = lay.by_m2.get(M2 + 2, ())
    bra_index = {m: i for i, m in enumerate(bra)}
    mat = np.zeros((len(bra), len(ket)))
    for col, mask in enumerate(ket):
        for (off, width, tj) in lay.blocks:
            block_mask = ((1 << width) - 1) << off
            local = (mask & block_mask) >> off
            for new_local, coeff in _jplus_apply(tj, local).items():
                new_mask = (mask & ~block_mask) | (new_local << off)
                row = bra_index.get(new_mask)
                if row is not None:
                    mat[row, col] += coeff
    return mat


def j2_matrix(pattern, M2: int) -> np.ndarray:
    """``J^2`` restricted to the pattern's ``2M`` determinant space.

    ``J^2 = J_z^2 + (J_+J_- + J_-J_+)/2`` with ``J_z = M`` diagonal and
    ``J_- = J_+^dagger``; the two cross terms are Gram matrices of the
    ``J_+`` blocks into and out of the ``2M`` space.
    """
    lay = pattern_layout(pattern)
    n = len(lay.by_m2.get(M2, ()))
    jz = 0.5 * M2
    mat = np.eye(n) * (jz * jz)
    g_plus = _jplus_block(tuple(lay.pattern), M2)            # M2 -> M2+2
    mat += 0.5 * (g_plus.T @ g_plus)
    g_minus = _jplus_block(tuple(lay.pattern), M2 - 2)       # M2-2 -> M2
    mat += 0.5 * (g_minus @ g_minus.T)
    return mat


# ---------------------------------------------------------------------------
# Coupling trees
# ---------------------------------------------------------------------------

def coupling_paths(order):
    """Every sequential coupling path of ``order``, in generator order.

    ``order`` is the sequence of ``(n, kappa, q)`` subshells *in the order
    they are to be coupled* (the CSF's own order).  A path is a tuple of
    ``(cumulative_total_2J_after_this_subshell, this_subshell_2J,
    degeneracy_index)`` entries.  The enumeration order and the
    multiplicities match
    :func:`atomx.config.csf._couple_j_count_maps`, which is what
    fixes the CSF count and the total-``2J`` multiset per configuration;
    both are independent of the coupling order.
    """
    order = tuple(tuple(p) for p in order)
    states: list[tuple[tuple, int]] = [((), 0)]
    for (_n, kappa, q) in order:
        shell_counts = subshell_j_counts_mscheme(twoj(kappa), int(q))
        new: list[tuple[tuple, int]] = []
        for chain, cum in states:
            for j2, mult in sorted(shell_counts.items()):
                for deg in range(mult):
                    for coupled in allowed_coupled_J2(cum, j2):
                        new.append((chain + ((coupled, j2, deg),), coupled))
        states = new
    return [chain for chain, _cum in states]


def canonical_coupling_tree(order, total_J2: int):
    """First coupling path of ``order`` whose final ``2J`` is ``total_J2``.

    Used for CSFs built by hand (``CSF.coupling is None``).  Paths that
    reach the same total ``2J`` are degenerate in ``J`` and span the same
    block; the first in generator order is the canonical representative.
    Raises :class:`ExactCISpaceError` when no path reaches ``total_J2``:
    silently inventing one would put a state of the wrong angular momentum
    into the block.
    """
    total_J2 = int(total_J2)
    chains = coupling_paths(order)
    for chain in chains:
        if chain[-1][0] == total_J2:
            return chain
    reachable = sorted({chain[-1][0] for chain in chains})
    raise ExactCISpaceError(
        f"no sequential jj coupling of {tuple(order)} reaches 2J = "
        f"{total_J2}; the reachable 2J values are {reachable}.  The CSF is "
        f"not consistent with its subshell occupations.")


class CSFSignature:
    """Hashable, orbital-free description of one CSF.

    ``subshells`` keeps the CSF's own order — the order its coupling tree
    is defined in; ``pattern`` is the canonical sorted form that fixes the
    determinant layout.
    """

    __slots__ = ("subshells", "pattern", "total_J2", "coupling", "key")

    def __init__(self, subshells, total_J2: int, coupling=None):
        self.subshells = tuple((int(n), int(k), int(q))
                               for (n, k, q) in subshells)
        self.pattern = tuple(sorted(self.subshells))
        self.total_J2 = int(total_J2)
        if coupling is None:
            coupling = canonical_coupling_tree(self.subshells, self.total_J2)
        coupling = tuple(tuple(int(x) for x in entry) for entry in coupling)
        if len(coupling) != len(self.subshells):
            raise ExactCISpaceError(
                f"coupling tree has {len(coupling)} entries for "
                f"{len(self.subshells)} subshells")
        if coupling[-1][0] != self.total_J2:
            raise ExactCISpaceError(
                f"coupling tree ends at 2J = {coupling[-1][0]} but the CSF "
                f"declares total_J2 = {self.total_J2}")
        for (n, kappa, q), (_cum, j2, deg) in zip(self.subshells, coupling,
                                                   strict=True):
            if q < 0 or q > twoj(kappa) + 1:
                raise ExactCISpaceError(
                    f"subshell (n={n}, kappa={kappa}) has occupation {q} "
                    f"outside 0..2j+1 = {twoj(kappa) + 1}")
            allowed = subshell_j_counts_mscheme(twoj(kappa), q)
            if j2 not in allowed:
                raise ExactCISpaceError(
                    f"subshell (n={n}, kappa={kappa}, 2j={twoj(kappa)}) with "
                    f"{q} electron(s) cannot couple to 2J = {j2}; the "
                    f"Pauli-allowed values are {sorted(allowed)}")
            if not 0 <= deg < allowed[j2]:
                raise ExactCISpaceError(
                    f"subshell (n={n}, kappa={kappa}) with {q} electron(s) "
                    f"has {allowed[j2]} state(s) with 2J = {j2}, so "
                    f"degeneracy index {deg} does not exist")
        self.coupling = coupling
        self.key = (self.subshells, self.total_J2, self.coupling)

    def __eq__(self, other):
        return isinstance(other, CSFSignature) and self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"CSFSignature({self.subshells!r}, {self.total_J2!r})"


def csf_signature(csf) -> CSFSignature:
    """Build the :class:`CSFSignature` of a
    :class:`~atomx.config.csf.CSF`.

    Subshell occupations are rounded to integers and validated: a
    fractional occupation is a configuration-average concept and has no
    meaning for an individual CSF, so a non-integer occupation raises
    instead of being silently truncated.
    """
    subshells = []
    for sh in csf.subshells:
        q = float(sh.occupation)
        qr = int(round(q))
        if abs(q - qr) > 1e-6:
            raise ExactCISpaceError(
                f"CSF {getattr(csf, 'label', '')!r} has non-integer "
                f"occupation {q} on (n={sh.n}, kappa={sh.kappa}); a CSF "
                f"must have definite subshell occupations.")
        subshells.append((int(sh.n), int(sh.kappa), qr))
    return CSFSignature(subshells, int(csf.total_J2),
                        getattr(csf, "coupling", None))


# ---------------------------------------------------------------------------
# CSF vectors in the determinant basis
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _subshell_series(tj: int, q: int, J2: int, deg: int):
    """``|j^q, J, M>`` for every ``M``, as ``(2M, ((mask, amp), ...))``.

    The highest-weight member comes from
    :func:`atomx.angular.mscheme.highest_weight_states`; the rest of the
    multiplet is generated by repeated ``J_-`` (``J_+^T`` between fixed-M
    blocks), which reproduces the exact normalisation
    ``J_-|J,M> = sqrt((J+M)(J-M+1))|J,M-1>``.
    """
    if q < 0 or q > tj + 1:
        raise ExactCISpaceError(f"occupation {q} out of range for 2j = {tj}")
    if q == 0 or q == tj + 1:
        if J2 != 0:
            raise ExactCISpaceError(
                f"an {'closed' if q else 'empty'} subshell (2j = {tj}, "
                f"q = {q}) has J = 0, not 2J = {J2}")
        full = (1 << (tj + 1)) - 1
        return ((0, (((0 if q == 0 else full), 1.0),)),)
    states = [s for s in highest_weight_states(tj, q) if s.J2 == J2]
    if deg >= len(states):
        raise ExactCISpaceError(
            f"subshell (2j = {tj})^{q} has {len(states)} state(s) with "
            f"2J = {J2}, so degeneracy index {deg} does not exist")
    state = states[deg]
    vector = np.asarray(state.coefficients, dtype=float)
    masks = state.determinants
    series = [(J2, tuple(zip(masks, (float(x) for x in vector), strict=True)))]
    M2 = J2
    while M2 > -J2:
        nxt = M2 - 2
        vector = jplus_matrix(tj, q, nxt).T @ vector   # J_- : M2 -> M2-2
        # J_-|J,M> = sqrt((J+M)(J-M+1))|J,M-1>: the norm is fixed by the
        # angular momentum, so it is checked rather than assumed, and the
        # state is normalised because the chain coupling below needs
        # orthonormal subshell states.
        m_hi = 0.5 * M2
        expected = math.sqrt((0.5 * J2 + m_hi) * (0.5 * J2 - m_hi + 1.0))
        norm = float(np.linalg.norm(vector))
        if abs(norm - expected) > 1e-8 * max(1.0, expected):
            raise ExactCISpaceError(
                f"lowering the (2j = {tj})^{q} state with 2J = {J2} from "
                f"2M = {M2} to 2M = {nxt} gave norm {norm!r}, expected "
                f"sqrt((J+M)(J-M+1)) = {expected!r}; the J_+ blocks of "
                f"atomx.angular.mscheme are inconsistent with the coupling "
                f"convention.")
        vector = vector / norm
        masks = determinants_by_m(tj, q, nxt)
        series.append((nxt, tuple(zip(masks, (float(x) for x in vector),
                                      strict=True))))
        M2 = nxt
    return tuple(series)


@lru_cache(maxsize=256)
def _csf_local_vector(subshells: tuple, total_J2: int, coupling: tuple):
    """CSF amplitudes over per-subshell local masks, in CSF order.

    Sequential Clebsch-Gordan coupling at total ``M = J``: at step ``i``
    the running state ``|J_prev M_prev>`` couples with ``|J_i M_i>`` to
    ``|J_cum M_prev + M_i>``, exactly the tree the CSF records.
    """
    specs = []
    for (_n, kappa, q), (_cum, j2, deg) in zip(subshells, coupling,
                                               strict=True):
        tj = twoj(kappa)
        specs.append((tj, _m2_values(tj),
                      dict(_subshell_series(tj, q, j2, deg))))
    partial: dict[tuple, float] = {(): 1.0}
    cum_prev = 0
    for i, (_tj, _m2_of_bit, series) in enumerate(specs):
        cum = coupling[i][0]
        new: dict[tuple, float] = {}
        for masks, amp in partial.items():
            m_cum = 0
            for k, mask in enumerate(masks):
                m_cum += _mask_m2(mask, specs[k][1])
            for m2_i, vec in series.items():
                cg = clebsch_gordan(cum_prev, m_cum, coupling[i][1], m2_i,
                                    cum, m_cum + m2_i)
                if cg == 0.0:
                    continue
                for mask_i, amp_i in vec:
                    if amp_i == 0.0:
                        continue
                    key = masks + (mask_i,)
                    new[key] = new.get(key, 0.0) + amp * amp_i * cg
        partial = new
        cum_prev = cum
    out = []
    for masks, amp in partial.items():
        if amp == 0.0:
            continue
        if sum(_mask_m2(m, specs[k][1]) for k, m in enumerate(masks)) \
                != total_J2:
            continue
        out.append((masks, amp))
    if not out:
        raise ExactCISpaceError(
            f"coupling tree {coupling} of {subshells} produced no "
            f"component with 2M = {total_J2}")
    return tuple(out)


def _permutation_sign(subshells, order_index) -> float:
    """Sign of reordering whole subshell blocks into the layout order.

    A determinant is a product of creation operators in ascending spinor
    index, while the CSF is coupled in its own subshell order; moving a
    block of ``n_p`` occupied spinors past a block of ``n_q`` costs
    ``(-1)^{n_p n_q}``.  The factor is the same for every component of one
    CSF, so it is a global sign: energies are unaffected, but keeping it
    makes the individual ``H_rs`` reproducible against the oracle.
    """
    order = [order_index[(n, k)] for (n, k, _q) in subshells]
    occ = [q for (_n, _k, q) in subshells]
    sign = 1
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if order[i] > order[j] and (occ[i] * occ[j]) % 2:
                sign = -sign
    return float(sign)


@lru_cache(maxsize=512)
def csf_magnetic_vector(signature, m2: int, orbital_keys: tuple):
    """Signed determinant expansion at any M, in one common spinor layout.

    The highest-weight CSF uses the Hamiltonian's coupling and fermionic
    ordering. Other M members follow normalized J_-; diagonalizing each M
    sector independently would destroy this shared phase convention.
    Returns immutable (mask, amplitude) pairs. Spinors are (n,kappa,2m),
    with sorted orbital keys and ascending 2m inside each orbital.
    """
    keys = tuple(sorted(orbital_keys))
    j2 = signature.total_J2
    if m2 not in range(-j2, j2 + 1, 2) or len(set(keys)) != len(keys):
        raise ValueError("valid M and unique orbital keys required")
    if not {(n, k) for n, k, _ in signature.subshells}.issubset(keys):
        raise ValueError("CSF orbitals are missing from the common layout")
    offsets, spinors = {}, []
    for n, k in keys:
        offsets[n, k] = len(spinors)
        spinors.extend((n, k, m) for m in _m2_values(twoj(k)))
    sign = _permutation_sign(
        signature.subshells, {key: i for i, key in enumerate(keys)}
    )
    vector = {}
    for masks, amplitude in _csf_local_vector(
        signature.subshells, j2, signature.coupling
    ):
        mask = 0
        for (n, k, _), local in zip(signature.subshells, masks, strict=True):
            mask |= local << offsets[n, k]
        vector[mask] = sign * amplitude
    for current in range(j2, m2, -2):
        lowered = {}
        for mask, amplitude in vector.items():
            for bit in _bits(mask):
                n, k, magnetic = spinors[bit]
                if magnetic == -twoj(k):
                    continue
                destination = bit - 1
                if mask & (1 << destination):
                    continue
                annihilated = _annihilate(mask, bit)
                if annihilated is None:
                    continue
                sign_rm, removed = annihilated
                created = _create(removed, destination)
                if created is None:
                    continue
                sign_cr, new = created
                coefficient = (
                    math.sqrt((twoj(k) + magnetic) * (twoj(k) - magnetic + 2)) / 2
                )
                lowered[new] = (
                    lowered.get(new, 0.0) + sign_rm * sign_cr * coefficient * amplitude
                )
        expected = math.sqrt((j2 + current) * (j2 - current + 2)) / 2
        norm = math.sqrt(sum(value**2 for value in lowered.values()))
        if abs(norm - expected) > 1e-10:
            raise ExactCISpaceError("magnetic CSF lowering violates the J_- norm")
        vector = {
            mask: amplitude / expected
            for mask, amplitude in lowered.items()
            if amplitude != 0.0
        }
    norm = sum(value**2 for value in vector.values())
    if abs(norm - 1) > 1e-10:
        raise ExactCISpaceError("magnetic CSF vector is not normalized")
    return tuple(sorted(vector.items()))


@lru_cache(maxsize=4096)
def one_body_tensor_terms(bra, ket, rank: int):
    """Reduced CSF tensor as ``(destination, source, coefficient)`` terms.

    Multiply each coefficient by the corresponding one-electron reduced
    matrix element. The determinant phases and coupling trees are exactly
    those of the CI Hamiltonian. Evaluating the highest-weight component
    and dividing by its Wigner 3j gives the many-electron reduced element;
    no occupation-average or closed-shell normalization is involved.
    """
    jf, ji = bra.total_J2, ket.total_J2
    q2 = jf - ji
    if abs(q2) > 2 * rank or jf + ji < 2 * rank or q2 % 2:
        return ()
    if sum(p[2] for p in bra.pattern) != sum(p[2] for p in ket.pattern):
        return ()
    occupations = {(n, k): q for n, k, q in bra.pattern}
    for n, k, q in ket.pattern:
        occupations[n, k] = occupations.get((n, k), 0) - q
    if sum(abs(q) for q in occupations.values()) > 2:
        return ()
    denominator = wigner_3j(jf, 2 * rank, ji, -jf, q2, ji)
    if denominator == 0.0:
        return ()
    offsets = dict(_union_offsets((bra.pattern, ket.pattern)))
    order = {key: i for i, key in enumerate(offsets)}

    def vector(sig):
        sign = _permutation_sign(sig.subshells, order)
        result = {}
        for masks, amplitude in _csf_local_vector(
                sig.subshells, sig.total_J2, sig.coupling):
            determinant = 0
            for (n, k, _), mask in zip(sig.subshells, masks, strict=True):
                determinant |= mask << offsets[n, k]
            result[determinant] = sign * amplitude
        return result

    vf, vi = vector(bra), vector(ket)
    spinors = [(n, k, m) for n, k in offsets
               for m in _m2_values(twoj(k))]
    destinations = {}
    for ia, (n, k, m) in enumerate(spinors):
        destinations.setdefault(m, []).append((ia, (n, k), twoj(k)))
    terms = {}
    for determinant, amplitude in vi.items():
        for ib in _bits(determinant):
            nb, kb, mb = spinors[ib]
            annihilated = _annihilate(determinant, ib)
            if annihilated is None:
                continue
            sign_b, removed = annihilated
            for ia, key_a, ja in destinations.get(mb + q2, ()):
                created = _create(removed, ia)
                if created is None:
                    continue
                sign_a, final = created
                amplitude_f = vf.get(final, 0.0)
                if amplitude_f == 0.0:
                    continue
                phase = -1 if ((ja - mb - q2) // 2) % 2 else 1
                angular = phase * wigner_3j(
                    ja, 2 * rank, twoj(kb), -mb - q2, q2, mb)
                key = (key_a, (nb, kb))
                terms[key] = terms.get(key, 0.0) + (
                    amplitude_f * amplitude * sign_a * sign_b * angular / denominator)
    return tuple((a, b, coefficient) for (a, b), coefficient in sorted(terms.items())
                 if coefficient != 0.0)


def nonorthogonal_tensor_terms(bra, ket, rank, overlap):
    """One-body tensor coefficients between independently optimized bases.

    The generalized Slater–Condon expression uses overlap cofactors,
    including singular overlap matrices (where an inverse is invalid).
    ``overlap(a, b)`` is the radial bra/ket overlap for two same-kappa
    orbitals. This path is needed only for distinct radial orbital sets.
    """
    jf, ji = bra.total_J2, ket.total_J2
    q2 = jf - ji
    denominator = wigner_3j(jf, 2 * rank, ji, -jf, q2, ji)
    if denominator == 0.0 or sum(p[2] for p in bra.pattern) != sum(p[2] for p in ket.pattern):
        return ()
    offsets = dict(_union_offsets((bra.pattern, ket.pattern)))
    order = {key: i for i, key in enumerate(offsets)}
    spinors = [(n, k, m) for n, k in offsets for m in _m2_values(twoj(k))]

    def determinants(sig):
        sign = _permutation_sign(sig.subshells, order)
        out = []
        for masks, amplitude in _csf_local_vector(sig.subshells, sig.total_J2, sig.coupling):
            mask = 0
            for (n, k, _), local in zip(sig.subshells, masks, strict=True):
                mask |= local << offsets[n, k]
            out.append((tuple(_bits(mask)), sign * amplitude))
        return out

    terms = {}
    radial_overlaps = {}
    for occupied_f, amplitude_f in determinants(bra):
        for occupied_i, amplitude_i in determinants(ket):
            number = len(occupied_f)
            metric = np.zeros((number, number))
            for row, ia in enumerate(occupied_f):
                na, ka, ma = spinors[ia]
                for col, ib in enumerate(occupied_i):
                    nb, kb, mb = spinors[ib]
                    if ka == kb and ma == mb:
                        pair = ((na, ka), (nb, kb))
                        if pair not in radial_overlaps:
                            radial_overlaps[pair] = overlap(*pair)
                        metric[row, col] = radial_overlaps[pair]
            for row, ia in enumerate(occupied_f):
                na, ka, ma = spinors[ia]
                for col, ib in enumerate(occupied_i):
                    nb, kb, mb = spinors[ib]
                    if ma - mb != q2:
                        continue
                    angular = wigner_3j(twoj(ka), 2 * rank, twoj(kb), -ma, q2, mb)
                    if angular == 0.0:
                        continue
                    minor = np.delete(np.delete(metric, row, axis=0), col, axis=1)
                    cofactor = np.linalg.det(minor) if number > 1 else 1.0
                    exponent = row + col + (twoj(ka) - ma) // 2
                    weight = (-1.0 if exponent % 2 else 1.0) * cofactor
                    pair = ((na, ka), (nb, kb))
                    terms[pair] = terms.get(pair, 0.0) + (
                        amplitude_f * amplitude_i * weight * angular / denominator)
    return tuple((a, b, value) for (a, b), value in sorted(terms.items()) if value != 0.0)


def csf_matrix(signatures, pattern, offsets=None) -> np.ndarray:
    """Determinant-space coefficient matrix ``C`` of a CSFSignature list.

    Returns an ``(n_det, n_csf)`` array whose column ``r`` is CSF ``r``'s
    vector over the pattern's ``2M = total_J2`` determinants, in
    :attr:`PatternLayout.by_m2` order.  ``offsets`` maps each ``(n, kappa)``
    of the pattern to its bit offset: the pattern's own layout by default,
    or the shared union layout of a whole (2J, parity) block when the
    block spans several configurations.
    """
    lay = pattern_layout(pattern)
    order_index = {key: i for i, key in enumerate(lay.pattern_keys)}
    if offsets is None:
        offsets = {key: lay.offsets[i] for i, key in enumerate(lay.pattern_keys)}
    total_J2 = int(signatures[0].total_J2) if signatures else 0
    masks_at_j2 = [_remap(m, lay, offsets)
                   for m in lay.by_m2.get(total_J2, ())]
    index = {m: i for i, m in enumerate(masks_at_j2)}
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for col, sig in enumerate(signatures):
        if sig.pattern != lay.pattern:
            raise ExactCISpaceError(
                f"CSF {sig!r} does not have the pattern {lay.pattern}")
        sign = _permutation_sign(sig.subshells, order_index)
        for masks, amp in _csf_local_vector(sig.subshells, sig.total_J2,
                                            sig.coupling):
            g = 0
            for k, mask in enumerate(masks):
                key = (sig.subshells[k][0], sig.subshells[k][1])
                g |= mask << offsets[key]
            rows.append(index[g])
            cols.append(col)
            vals.append(sign * amp)
    mat = np.zeros((len(masks_at_j2), len(signatures)))
    if vals:
        mat[np.asarray(rows), np.asarray(cols)] = np.asarray(vals)
    return mat


# ---------------------------------------------------------------------------
# Angular kernel: <ab|1/r12|cd>, the oracle's convention
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32768)
def two_body_angular(ka: int, m2a: int, kb: int, m2b: int,
                     kc: int, m2c: int, kd: int, m2d: int):
    """``(k, angular)`` terms of ``<ab|1/r12|cd>`` for spinor labels.

    .. math::

        \\langle ab|r_{12}^{-1}|cd\\rangle = \\sum_k R^k(ac, bd)
          (-1)^{j_a-m_a+j_b-m_b} \\sum_q (-1)^q
          \\begin{pmatrix} j_a & k & j_c \\\\ -m_a & q & m_c \\end{pmatrix}
          \\begin{pmatrix} j_b & k & j_d \\\\ -m_b & -q & m_d \\end{pmatrix}
          \\langle\\kappa_a\\|C^k\\|\\kappa_c\\rangle
          \\langle\\kappa_b\\|C^k\\|\\kappa_d\\rangle

    ``R^k(ac, bd)`` is the Slater integral the caller's radial evaluator
    provides — keeping it outside is what makes this module independent of
    any radial workspace.  The 3j symbols come from
    :func:`atomx.angular.wigner.wigner_3j` and the reduced ``C^k`` from
    :func:`atomx.angular.coefficients.reduced_C_jj` (Grant's convention); the
    formula is the one ``benchmarks/mscheme_ci_reference.py::_angular_terms``
    implements with sympy.
    """
    if m2a + m2b != m2c + m2d:
        return ()
    q2 = m2a - m2c
    if q2 % 2:
        return ()
    q = q2 // 2
    tja, tjb, tjc, tjd = twoj(ka), twoj(kb), twoj(kc), twoj(kd)
    kmax = min((tja + tjc) // 2, (tjb + tjd) // 2)
    out = []
    for k in range(kmax + 1):
        ca = reduced_C_jj(ka, kc, k)
        if ca == 0.0:
            continue
        cb = reduced_C_jj(kb, kd, k)
        if cb == 0.0:
            continue
        t1 = wigner_3j(tja, 2 * k, tjc, -m2a, q2, m2c)
        if t1 == 0.0:
            continue
        t2 = wigner_3j(tjb, 2 * k, tjd, -m2b, -q2, m2d)
        if t2 == 0.0:
            continue
        exponent = (tja - m2a) // 2 + (tjb - m2b) // 2 + q
        phase = -1.0 if exponent % 2 else 1.0
        out.append((k, phase * t1 * t2 * ca * cb))
    return tuple(out)


# ---------------------------------------------------------------------------
# Slater-Condon terms of one determinant pair
# ---------------------------------------------------------------------------

def _ops_sign(mask: int, ops) -> int:
    """Sign of applying ``ops`` right-to-left to ``mask``; 0 if it vanishes.

    The same convention as the oracle's ``_apply_ops``: each step carries
    the canonical fermion sign of :func:`atomx.angular.mscheme._annihilate`
    / ``_create`` — the parity of the occupied lower indices.
    """
    sign = 1
    occ = mask
    for idx, create in reversed(ops):
        applied = _create(occ, idx) if create else _annihilate(occ, idx)
        if applied is None:
            return 0
        phase, occ = applied
        sign *= phase
    return sign


def _radial_terms(ia, ib, ic, id_, coeff, spinor_labels):
    """Radial keys of ``coeff * <ia ib|1/r12|ic id>``.

    The key is ``("r", k, key_a, key_c, key_b, key_d)`` — the role order
    ``R^k(ac, bd)`` that :func:`atomx.physics.csf.matrix._rk` takes as
    ``_rk(Pa, Qa, Pc, Qc, Pb, Qb, Pd, Qd, bsp_data, k)``.
    """
    na, ka, ma = spinor_labels[ia]
    nb, kb, mb = spinor_labels[ib]
    nc, kc, mc = spinor_labels[ic]
    nd, kd, md = spinor_labels[id_]
    terms = two_body_angular(ka, ma, kb, mb, kc, mc, kd, md)
    if not terms:
        return ()
    key_a, key_c = (na, ka), (nc, kc)
    key_b, key_d = (nb, kb), (nd, kd)
    return tuple((("r", k, key_a, key_c, key_b, key_d), coeff * ang)
                 for k, ang in terms)


def _pair_terms(mask_a: int, mask_b: int, spinor_labels, term_fn=None):
    """``(radial_key, coefficient)`` terms of ``<a|H|b>``.

    Explicit Slater-Condon classification (0, 1 or 2 spinor differences) of

        H = sum_ab h_ab a+_a a_b
            + sum_{a<b} sum_{c<d} [v(ab,cd) - v(ab,dc)] a+_a a+_b a_d a_c

    with ``v(ab,cd) = <ab|1/r12|cd>``.  Every operator combination that can
    connect the two determinants is enumerated with its exact sign from
    :func:`_ops_sign`; no term is approximated and none is dropped except
    by the Pauli principle.

    ``term_fn`` evaluates one two-body matrix element ``<ia ib|v|ic id>``
    as ``(key, coeff)`` pairs; it defaults to the Coulomb kernel
    :func:`_radial_terms`.  The one-body ``("h", ...)`` terms are emitted
    only by the Coulomb kernel — a two-body operator such as the Breit
    interaction has none, so ``term_fn`` returning ``()` for a
    zero-difference pair loop is exactly right.
    """
    if term_fn is None:
        term_fn = _radial_terms
    diff = mask_a ^ mask_b
    n_diff = diff.bit_count()
    if n_diff not in (0, 2, 4):
        return ()
    terms: list[tuple] = []

    if n_diff == 0:
        # One-body <a|h|a> for every occupied spinor.
        if term_fn is _radial_terms:
            for i in _bits(mask_b):
                n, kappa, _m2 = spinor_labels[i]
                terms.append((("h", n, kappa, n), 1.0))
        # Two-body direct minus exchange for every occupied pair.
        for c, d in itertools.combinations(list(_bits(mask_b)), 2):
            terms.extend(term_fn(c, d, c, d, 1.0, spinor_labels))
            terms.extend(term_fn(c, d, d, c, -1.0, spinor_labels))
        return tuple(terms)

    if n_diff == 2:
        p = next(_bits(mask_a & diff))          # created spinor
        q = next(_bits(mask_b & diff))          # annihilated spinor
        np_, kp, mp = spinor_labels[p]
        nq, kq, mq = spinor_labels[q]
        if term_fn is _radial_terms and kp == kq and mp == mq:
            sign = _ops_sign(mask_b, [(p, True), (q, False)])
            if sign:
                terms.append((("h", np_, kp, nq), float(sign)))
        common = [i for i in _bits(mask_a & mask_b)]
        for i in common:
            pa, pb = sorted((p, i))
            qa, qb = sorted((q, i))
            sign = _ops_sign(mask_b, [(pa, True), (pb, True),
                                      (qb, False), (qa, False)])
            if not sign:
                continue
            terms.extend(term_fn(pa, pb, qa, qb, float(sign),
                                 spinor_labels))
            terms.extend(term_fn(pa, pb, qb, qa, -float(sign),
                                 spinor_labels))
        return tuple(terms)

    # Two replacements have exactly two disjoint creation/removal bits.
    # Resolve their ordered indices directly instead of allocating generators,
    # sorted lists and a four-operation list for every connected determinant.
    created, removed = mask_a & diff, mask_b & diff
    if created.bit_count() != 2 or removed.bit_count() != 2:
        raise ValueError('double excitation requires two created and removed spinors')
    p_bit, r_bit = created & -created, removed & -removed
    q_bit, s_bit = created ^ p_bit, removed ^ r_bit
    p, q = p_bit.bit_length() - 1, q_bit.bit_length() - 1
    r, s = r_bit.bit_length() - 1, s_bit.bit_length() - 1
    remaining = mask_b ^ removed
    # Apply a_r, a_s, a+_q, a+_p. Removing r decreases the population
    # below s by one; q > p means creating q leaves the p count unchanged.
    parity = ((mask_b & (r_bit - 1)).bit_count()
              + (mask_b & (s_bit - 1)).bit_count() - 1
              + (remaining & (q_bit - 1)).bit_count()
              + (remaining & (p_bit - 1)).bit_count())
    sign = -1.0 if parity & 1 else 1.0
    terms.extend(term_fn(p, q, r, s, sign, spinor_labels))
    terms.extend(term_fn(p, q, s, r, -sign, spinor_labels))
    return tuple(terms)


def continuum_hamiltonian_terms(final_vector, initial_vector, spinor_labels,
                                kappa: int, m2: int):
    """Signed radial terms of <(N-1)-electron CI, epsilon|h+V|N-electron CI>.

    The continuum spinor (-1, kappa, m2) is appended after every bound spinor.
    Reuse the exact bound CI Slater-Condon kernel, including one-body terms
    and all spectators. No occupation or J-average factors are added.
    Vectors are magnetic determinant expansions in the same bound layout.
    The caller owns caching: CI mixing changes must invalidate these terms.
    """
    if kappa == 0 or m2 not in _m2_values(twoj(kappa)):
        raise ValueError("invalid continuum spinor quantum numbers")
    labels = tuple(spinor_labels) + ((-1, int(kappa), int(m2)),)
    continuum_bit = 1 << len(spinor_labels)
    output = {}
    for mask_i, amplitude_i in initial_vector.items():
        for mask_f, amplitude_f in final_vector.items():
            if mask_f.bit_count() + 1 != mask_i.bit_count():
                raise ValueError("continuum Hamiltonian requires N -> N-1")
            bra = mask_f | continuum_bit
            if (bra ^ mask_i).bit_count() > 4:
                continue
            factor = amplitude_f * amplitude_i
            for key, angular in _pair_terms(bra, mask_i, labels):
                output[key] = output.get(key, 0.0) + factor * angular
    return tuple((key, value) for key, value in output.items() if value != 0)


# ---------------------------------------------------------------------------
# Block structure: determinants + the sparse Slater-Condon table
# ---------------------------------------------------------------------------

def _connected_pairs(dets, *, triangular=False):
    """Slater–Condon connections from common N-2 spectator determinants.

    Yield unique pairs in row/column order. Removing up to two occupied
    spinors indexes every possible one-/two-body connection without scanning
    all determinant pairs. Only one row's neighbours are materialized;
    duplicate single-excitation connections collapse before angular work.
    """
    spectators = {}
    frozen = dets[0] if dets else 0
    for mask in dets:
        frozen &= mask

    def spectators_of(mask):
        # Spinors occupied in every determinant never participate in a
        # replacement. Removing frozen-core electrons only generates
        # duplicate singleton buckets and can dominate heavy-atom work.
        occupied = tuple(_bits(mask ^ frozen))
        remove = min(2, len(occupied))
        for bits in itertools.combinations(occupied, remove):
            spectator = mask
            for bit in bits:
                spectator ^= 1 << bit
            yield spectator

    for index, mask in enumerate(dets):
        for spectator in spectators_of(mask):
            previous = spectators.get(spectator)
            if previous is None:
                spectators[spectator] = index
            elif isinstance(previous, int):
                spectators[spectator] = [previous, index]
            else:
                previous.append(index)
    # A singleton bucket contributes only its determinant's diagonal.
    # First occurrences share the row integer instead of allocating a list;
    # discard their keys before streaming and insert each diagonal explicitly.
    spectators = {key: indices for key, indices in spectators.items()
                  if isinstance(indices, list)}
    # A pair belongs to the old global union iff its two indices share a
    # spectator bucket. Taking that same union one row at a time preserves
    # membership and sorted first-encounter order without retaining all pairs.
    for row, mask in enumerate(dets):
        columns = {row}
        for spectator in spectators_of(mask):
            columns.update(spectators.get(spectator, ()))
        for col in sorted(columns):
            if not triangular or col >= row:
                yield row, col


def _coefficient_table(dets, labels, term_fn=None, *, radial_symmetry=False):
    pair_terms = _block_pair_terms(dets, labels, term_fn, radial_symmetry)
    keys, key_index = [], {}
    indptr, cols, pairs = array('q', [0]), array('q'), array('q')
    data = array('d')
    complex_data = False
    complex_types = (complex, np.complexfloating)
    for row, col in _connected_pairs(dets, triangular=radial_symmetry):
        terms = pair_terms(dets[row], dets[col])
        if not terms:
            continue
        pairs.extend((row, col))
        for key, coefficient in terms:
            if key not in key_index:
                key_index[key] = len(keys)
                keys.append(key)
            cols.append(key_index[key])
            # Native angular coefficients are real. Preserve the generic
            # callback's complex values by promoting once, without retaining
            # Python objects for every coefficient in the production path.
            if not complex_data and isinstance(coefficient, complex_types):
                data = array('d', (part for value in data for part in (value, 0.)))
                complex_data = True
            if complex_data:
                data.extend((coefficient.real, coefficient.imag))
            else:
                data.append(coefficient)
        indptr.append(len(cols))
    # Each retained connection is already one CSR row. Typed buffers avoid
    # COO row indices, Python triplets and a second full coordinate array.
    table = csr_matrix(
        (np.frombuffer(data, dtype=complex if complex_data else float),
         np.frombuffer(cols, dtype=np.int64),
         np.frombuffer(indptr, dtype=np.int64)),
        shape=(len(pairs)//2, max(len(keys), 1)))
    table.sum_duplicates()
    table.sort_indices()
    coordinates = np.frombuffer(pairs, dtype=np.int64).reshape(-1, 2)
    return tuple(keys), table, coordinates


def _block_pair_terms(dets, labels, term_fn=None, radial_symmetry=False):
    """Shared signed pair algebra for cached and streamed numeric assembly."""
    # Normal order against spinors occupied in every determinant. Their
    # core-core energy and core-active one-body field are identical across
    # the block; recomputing all N(N-1)/2 pairs for each diagonal determinant
    # is particularly expensive for heavy atoms. This is an exact contraction,
    # not frozen radial orbitals: every radial integral remains in the table
    # and is differentiated when the orbitals change.
    frozen = dets[0] if dets else 0
    for mask in dets:
        frozen &= mask
    pair_terms = _core_contracted_pair_terms(frozen, labels, term_fn)
    if not frozen and radial_symmetry:
        # The static canonical-key collection below already combines terms.
        # With no common core there is no normal-order contraction to perform;
        # avoid collecting the same determinant terms twice.
        def pair_terms(a, b):
            return _pair_terms(a, b, labels, term_fn=term_fn)
    if not radial_symmetry:
        return pair_terms
    from atomx.physics.csf.radial_symmetry import canonical_static_radial_key

    def canonical_terms(a, b):
        terms = pair_terms(a, b)
        combined = {}
        for key, coefficient in terms:
            key, sign = canonical_static_radial_key(key)
            if sign:
                combined[key] = combined.get(key, 0.) + sign*coefficient
        return tuple((key, value) for key, value in combined.items() if value != 0.)

    return canonical_terms


def _streamed_determinant_matrix(dets, labels, evaluate_integral, *,
                                 breit_evaluator=None, integral_dtype=float,
                                 radial_symmetry=False):
    """Contract the shared pair algebra directly into numeric CSR rows.

    Only radial values are memoized, for this immutable build. No angular
    coefficient table or connected-coordinate array is formed. Every pair
    with nonempty angular terms is retained, including exact numeric zeros.
    """
    dtype = np.dtype(integral_dtype)
    if dtype not in (np.dtype(float), np.dtype(complex)):
        raise ValueError("integral_dtype must be float64 or complex128")
    operators = [(_block_pair_terms(dets, labels, radial_symmetry=radial_symmetry),
                  evaluate_integral, {})]
    if breit_evaluator is not None:
        from atomx.physics.csf.breit_ci import breit_terms
        operators.append((_block_pair_terms(dets, labels, breit_terms, radial_symmetry),
                          breit_evaluator, {}))
    indptr, columns, data = array('q', [0]), array('q'), array('d')
    current_row = 0
    for row, col in _connected_pairs(dets, triangular=radial_symmetry):
        while current_row < row:
            indptr.append(len(columns))
            current_row += 1
        total = 0j if dtype.kind == 'c' else 0.
        nonempty = False
        for pair_terms, evaluate, cache in operators:
            terms = pair_terms(dets[row], dets[col])
            nonempty = nonempty or bool(terms)
            contribution = 0j if dtype.kind == 'c' else 0.
            for key, coefficient in terms:
                value = cache.get(key)
                if value is None:
                    value = evaluate(key)
                    if dtype.kind != 'c' and np.iscomplexobj(value):
                        raise ValueError("complex radial integral requires integral_dtype=complex")
                    if not np.isscalar(value) or not np.isfinite(value):
                        raise ValueError(f"nonfinite or nonscalar radial integral {key}")
                    value = dtype.type(value).item()
                    cache[key] = value
                contribution += coefficient*value
            total += contribution
        if nonempty:
            columns.append(col)
            if dtype.kind == 'c':
                data.extend((total.real, total.imag))
            else:
                data.append(total)
    while current_row < len(dets):
        indptr.append(len(columns))
        current_row += 1
    result = csr_matrix((np.frombuffer(data, dtype=dtype),
                         np.frombuffer(columns, dtype=np.int64),
                         np.frombuffer(indptr, dtype=np.int64)),
                        shape=(len(dets), len(dets)))
    if radial_symmetry:
        from scipy.sparse import diags
        result = result + result.T - diags(result.diagonal(), format='csr')
    return result


def _core_contracted_pair_terms(frozen, labels, term_fn=None):
    """Exact Slater–Condon contractions with a common occupied determinant."""
    operator = _radial_terms if term_fn is None else term_fn

    def collect(terms):
        combined = {}
        for key, value in terms:
            combined[key] = combined.get(key, 0.) + value
        return tuple((key, value) for key, value in combined.items() if value != 0.)

    core = collect(_pair_terms(frozen, frozen, labels, term_fn=operator))
    fields = {}

    def field(p, q):
        if (p, q) not in fields:
            terms = []
            for i in _bits(frozen):
                pa, pb = sorted((p, i))
                qa, qb = sorted((q, i))
                order_sign = -1. if (p > i) != (q > i) else 1.
                terms.extend(operator(pa, pb, qa, qb, order_sign, labels))
                terms.extend(operator(pa, pb, qb, qa, -order_sign, labels))
            fields[p, q] = collect(terms)
        return fields[p, q]

    def pair(mask_a, mask_b):
        active_a, active_b = mask_a ^ frozen, mask_b ^ frozen
        difference = mask_a ^ mask_b
        count = difference.bit_count()
        if count == 0:
            terms = list(core)
            terms.extend(_pair_terms(active_a, active_b, labels, term_fn=operator))
            for p in _bits(active_a):
                terms.extend(field(p, p))
            return collect(terms)
        if count == 2:
            p, q = next(_bits(mask_a & difference)), next(_bits(mask_b & difference))
            sign = _ops_sign(mask_b, [(p, True), (q, False)])
            active_sign = _ops_sign(active_b, [(p, True), (q, False)])
            # Removing common core bits may change determinant ordering.
            terms = [(key, sign * active_sign * value) for key, value in
                     _pair_terms(active_a, active_b, labels, term_fn=operator)]
            terms.extend((key, sign * value) for key, value in field(p, q))
            return collect(terms)
        return collect(_pair_terms(mask_a, mask_b, labels, term_fn=operator))

    return pair


class BlockStructure:
    """Orbital-independent structure of one (2J, parity) block."""

    __slots__ = ("patterns", "j2", "dets", "slices", "keys", "table",
                 "spinor_labels", "pairs", "radial_symmetry")

    def __init__(self, patterns, j2, dets, slices, keys, table,
                 spinor_labels, pairs, radial_symmetry=False):
        self.patterns = patterns
        self.j2 = j2
        self.dets = dets
        self.slices = slices
        self.keys = keys
        self.table = table
        self.spinor_labels = spinor_labels
        self.pairs = pairs
        self.radial_symmetry = radial_symmetry


def _block_basis(patterns: tuple, j2: int, max_determinants: int):
    """Canonical determinant basis without allocating angular coefficients."""
    patterns = tuple(sorted(tuple(sorted(p)) for p in patterns))
    dimensions = [dict(pattern_m_counts(p)).get(j2, 0) for p in patterns]
    dimension = sum(dimensions)
    if dimension > max_determinants:
        sizes = ", ".join(f"{p}: {dim}" for p, dim in zip(patterns, dimensions, strict=True))
        raise ExactCISpaceError(
            f"exact CI block (2J = {j2}) needs a raw determinant space of "
            f"{dimension} determinants over {len(patterns)} configuration(s), "
            f"above the cap of {max_determinants}. Per-configuration sizes: {sizes}. "
            "Reduce the configuration list or raise max_determinants explicitly.")
    layouts = [pattern_layout(p) for p in patterns]
    offsets = dict(_union_offsets(patterns))
    dets: list[int] = []
    slices: list[tuple[int, int]] = []
    for lay in layouts:
        start = len(dets)
        dets.extend(_remap(m, lay, offsets) for m in lay.by_m2.get(j2, ()))
        slices.append((start, len(dets)))
    union_keys = [key for key, _off in _union_offsets(patterns)]
    spinor_labels: list[tuple[int, int, int]] = []
    for key in union_keys:
        spinor_labels.extend((key[0], key[1], m)
                             for m in _m2_values(twoj(key[1])))
    return patterns, tuple(dets), tuple(slices), tuple(spinor_labels)


@lru_cache(maxsize=32)
def _block_structure(patterns: tuple, j2: int,
                     max_determinants: int, radial_symmetry=False) -> BlockStructure:
    patterns, dets, slices, spinor_labels = _block_basis(patterns, j2, max_determinants)
    keys, table, pairs = _coefficient_table(dets, spinor_labels,
                                           radial_symmetry=radial_symmetry)
    return BlockStructure(patterns, j2, dets, slices,
                          keys, table, spinor_labels, pairs, radial_symmetry)


def block_terms(patterns, j2, max_determinants=MAX_BLOCK_DETERMINANTS, *,
                radial_symmetry=False):
    """Cached :class:`BlockStructure` of a block's configuration list."""
    if not isinstance(radial_symmetry, bool):
        raise ValueError('radial_symmetry must be bool')
    return _block_structure(tuple(patterns), int(j2), int(max_determinants), radial_symmetry)


class BreitBlockStructure:
    """Sparse table of one block's *Breit* (Gaunt plus retardation) determinant elements.

    Same determinant ordering as the block's :class:`BlockStructure`, so
    the two matrices add directly in the determinant basis.
    """

    __slots__ = ("dets", "keys", "table", "pairs", "radial_symmetry")

    def __init__(self, dets, keys, table, pairs, radial_symmetry=False):
        self.dets = dets
        self.keys = keys
        self.table = table
        self.pairs = pairs
        self.radial_symmetry = radial_symmetry


@lru_cache(maxsize=32)
def _breit_block_table(patterns: tuple, j2: int, max_determinants: int,
                       radial_symmetry=False):
    """Cached full static Breit table, keyed like the Coulomb structure."""
    from atomx.physics.csf.breit_ci import breit_terms

    structure = block_terms(patterns, j2, max_determinants, radial_symmetry=radial_symmetry)
    dets = structure.dets
    spinor_labels = structure.spinor_labels
    keys, table, pairs = _coefficient_table(dets, spinor_labels, breit_terms,
                                           radial_symmetry=radial_symmetry)
    return BreitBlockStructure(dets, keys, table, pairs, radial_symmetry)


def breit_block_terms(patterns, j2, max_determinants=MAX_BLOCK_DETERMINANTS, *,
                      radial_symmetry=False):
    """Cached static Breit determinant table of a block."""
    if not isinstance(radial_symmetry, bool):
        raise ValueError('radial_symmetry must be bool')
    return _breit_block_table(tuple(patterns), int(j2),
                              int(max_determinants), radial_symmetry)


def evaluate_block(structure: BlockStructure, evaluate_integral, *, sparse=False,
                   integral_dtype=float):
    """Numeric determinant Hamiltonian from the connected-entry table.

    Complex radial integrals require explicit ``integral_dtype=complex``;
    the cached angular algebra is shared without discarding imaginary parts.
    This contraction does not construct a complex-scaled no-pair projector.
    """
    dtype = np.dtype(integral_dtype)
    if dtype not in (np.dtype(float), np.dtype(complex)):
        raise ValueError("integral_dtype must be float64 or complex128")

    def values_iter():
        for key in structure.keys:
            value = evaluate_integral(key)
            if dtype.kind != 'c' and np.iscomplexobj(value):
                raise ValueError("complex radial integral requires integral_dtype=complex")
            if not np.isscalar(value) or not np.isfinite(value):
                raise ValueError(f"nonfinite or nonscalar radial integral {key}")
            yield value

    n = len(structure.dets)
    if structure.keys:
        rvec = np.fromiter(values_iter(), dtype, len(structure.keys))
        values = structure.table @ rvec
    else:
        values = np.zeros(len(structure.pairs), dtype=dtype)
    matrix = csr_matrix((values, (structure.pairs[:, 0], structure.pairs[:, 1])),
                        shape=(n, n))
    if structure.radial_symmetry:
        from scipy.sparse import diags
        # Transpose, not adjoint: radial c-products can be genuinely complex.
        matrix = matrix + matrix.T - diags(matrix.diagonal(), format='csr')
    return matrix if sparse else matrix.toarray()


# ---------------------------------------------------------------------------
# Block CSF matrix and its span checks
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _block_csf_sparse_matrix(patterns: tuple, j2: int, sig_groups: tuple):
    """Keep the exact configuration blocks without materialising their zeros."""
    pairs = sorted(((tuple(sorted(tuple(p) for p in pat)), tuple(g))
                    for pat, g in zip(patterns, sig_groups, strict=True)),
                   key=lambda pg: pg[0])
    patterns = tuple(pat for pat, _g in pairs)
    sig_groups = tuple(g for _pat, g in pairs)
    offsets = dict(_union_offsets(patterns))
    layouts = [pattern_layout(p) for p in patterns]
    blocks = []
    for lay, sigs in zip(layouts, sig_groups, strict=True):
        dim = len(lay.by_m2.get(j2, ()))
        if sigs:
            blocks.append(csr_matrix(csf_matrix(sigs, lay.pattern, offsets)))
        else:
            blocks.append(csr_matrix((dim, 0)))
    return block_diag(blocks, format='csr') if blocks else csr_matrix((0, 0))


@lru_cache(maxsize=16)
def _block_csf_matrix(patterns: tuple, j2: int, sig_groups: tuple):
    return _block_csf_sparse_matrix(patterns, j2, sig_groups).toarray()


def csf_block_matrix(patterns, j2, sig_groups, *, sparse=False):
    """Cached block CSF matrix, dense by default; canonical group column order.

    The sparse representation retains all coefficients without thresholding.
    Exact Hamiltonian projection uses it to avoid inter-configuration zeros.
    """
    if not isinstance(sparse, bool):
        raise ValueError('sparse must be bool')
    build = _block_csf_sparse_matrix if sparse else _block_csf_matrix
    return build(
        tuple(patterns), int(j2),
        tuple(tuple(g) for g in sig_groups))


@lru_cache(maxsize=256)
def _verify_cached(patterns: tuple, j2: int, sig_groups: tuple) -> bool:
    """Cache :func:`verify_csf_span` keyed like the matrix it checks.

    The checks are a pure function of the CSF signatures (the matrix they
    inspect is cached and immutable), and they include one ``eigh`` per
    configuration, so re-running them inside every MCDHF outer iteration
    would be pure overhead.
    """
    verify_csf_span(patterns, j2,
                    _block_csf_sparse_matrix(patterns, j2, sig_groups))
    return True


def verify_csf_span(patterns, j2, csf_mat) -> None:
    """Check that the CSFs are orthonormal and span the ``J``-block.

    Two independent invariants:

    * ``C^T C = I`` — the CSFs are orthonormal (a wrong coupling tree, a
      missing Clebsch-Gordan step or a bad permutation sign breaks it);
    * inside the configuration's ``2M = 2J`` determinant space, the ``J^2``
      eigenspace with eigenvalue ``J(J+1)`` has exactly as many dimensions
      as there are CSF columns, and contains every CSF column.  Its
      dimension equals the number of CSFs of that configuration with total
      ``2J``, so a mismatch means the CSF generator and the m-scheme
      multiplicity disagree.
    """
    csf_mat = csr_matrix(csf_mat)
    n_csf = csf_mat.shape[1]
    patterns = tuple(sorted(tuple(sorted(tuple(p) for p in pat))
                            for pat in patterns))
    if n_csf:
        gram_error = csf_mat.T @ csf_mat - eye(n_csf, format='csr')
        err = float(np.max(np.abs(gram_error.data), initial=0.0))
        if err > 1e-8:
            raise ExactCISpaceError(
                f"CSF vectors of block 2J = {j2} over "
                f"{tuple(patterns)} are not orthonormal "
                f"(max |C^T C - I| = {err:.3e}); the coupling-tree "
                f"expansion is internally inconsistent.")
    row0 = 0
    for pattern in patterns:
        lay = pattern_layout(pattern)
        dim = len(lay.by_m2.get(j2, ()))
        block = csf_mat[row0:row0 + dim]
        row0 += dim
        n_here = int((np.asarray(abs(block).sum(axis=0)).ravel() > 1e-12).sum())
        if n_here == 0:
            continue
        counts = dict(pattern_m_counts(lay.pattern))
        dim_target = counts.get(j2, 0) - counts.get(j2 + 2, 0)
        if dim_target != n_here:
            raise ExactCISpaceError(
                f"configuration {lay.pattern} has {n_here} CSF(s) with "
                f"2J = {j2}, but its m-scheme determinant space at "
                f"2M = {j2} contains {dim_target} state(s) with "
                f"J = {j2 / 2:g}.  The CSF list is not a complete basis of "
                f"the intended J-block.")
        raised = csr_matrix(_jplus_block(lay.pattern, j2)) @ block
        leak = float(np.max(np.abs(raised.data), initial=0.0))
        if leak > 1e-8:
            raise ExactCISpaceError(
                f"a CSF of configuration {lay.pattern} with 2J = {j2} is not "
                f"an eigenstate of J^2 (leakage {leak:.3e}); the coupling "
                f"tree does not describe a state of definite J.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def exact_block_matrix(signatures, evaluate_integral, *,
                       max_determinants: int = MAX_BLOCK_DETERMINANTS,
                       verify: bool = True,
                       breit_evaluator=None, as_operator=False, integral_dtype=float,
                       radial_symmetry=False, angular_storage='cached'):
    """Exact ``H`` of one ``(2J, parity)`` block in the CSF basis.

    Parameters
    ----------
    signatures : list of CSFSignature
        Every CSF of the block, in the order the caller wants the matrix
        rows and columns.
    evaluate_integral : callable
        Maps a radial key to its numeric value; see
        :func:`atomx.physics.csf.matrix._radial_evaluator`.
    max_determinants : int
        Cap on the raw determinant dimension of the block.
    verify : bool
        Run the orthonormality and ``J^2`` span checks (default on; the
        cost is one ``eigh`` per configuration, cached across calls).
    breit_evaluator : callable or None
        Maps a static Breit radial key to its numeric value (see
        :func:`atomx.physics.csf.breit_ci.gaunt_terms` for the key
        shape).  When given, the Breit operator's determinant matrix is
        added to ``H_det`` before projection, so it enters the CI matrix
        block-by-block and shifts levels differentially.  ``None`` (the
        default) leaves the block pure Dirac-Coulomb, so callers that do
        not ask for the Breit interaction pay nothing.
    integral_dtype : float or complex
        Explicit numeric radial representation. Complex128 reuses the
        real signed CSF angular projection with the c-product C.T H C;
        the matrix is complex symmetric, not Hermitian. An iterative
        operator's rmatvec nevertheless means the conjugate adjoint as
        required by SciPy. No complex eigenstate/no-pair provider is inferred.
    radial_symmetry : bool
        Explicitly promise real-or-c-product static reciprocal h/r/g/t
        integrals. Compile exact radial permutation/sign identities and
        evaluate only one determinant triangle. Arbitrary evaluators and
        frequency-dependent kernels must retain the default False.
    angular_storage : {'cached', 'streamed'}
        Cache the orbital-independent coefficient table (default), or contract
        the same signed terms directly into numeric determinant CSR. Streamed
        assembly retains all connections and memoizes radial values only for
        this build; it does not provide orbital-density coefficient tables.

    Returns
    -------
    ndarray
        Symmetric ``(n_csf, n_csf)`` matrix with
        ``H[r, s] = <CSF_r|H|CSF_s>``.
    """
    if angular_storage not in ('cached', 'streamed'):
        raise ValueError("angular_storage must be 'cached' or 'streamed'")
    if np.dtype(integral_dtype) not in (np.dtype(float), np.dtype(complex)):
        raise ValueError("integral_dtype must be float64 or complex128")
    if not isinstance(radial_symmetry, bool):
        raise ValueError('radial_symmetry must be bool')
    if not signatures:
        return np.zeros((0, 0), dtype=integral_dtype)
    j2 = int(signatures[0].total_J2)
    for sig in signatures:
        if int(sig.total_J2) != j2:
            raise ExactCISpaceError(
                "exact_block_matrix needs a single (2J, parity) block; got "
                f"2J = {j2} and 2J = {sig.total_J2}")
    patterns: list[tuple] = []
    groups: dict[tuple, list[CSFSignature]] = {}
    for sig in signatures:
        if sig.pattern not in groups:
            patterns.append(sig.pattern)
            groups[sig.pattern] = []
        groups[sig.pattern].append(sig)
    if angular_storage == 'cached':
        structure = block_terms(patterns, j2, max_determinants, radial_symmetry=radial_symmetry)
        canonical_patterns, slices = structure.patterns, structure.slices
    else:
        canonical_patterns, dets, slices, labels = _block_basis(patterns, j2, max_determinants)
    # ``block_terms`` and ``csf_block_matrix`` canonicalise the pattern order
    # (sorted), so the projected matrix comes back with the columns permuted
    # relative to ``signatures``.  Undo that permutation: callers index H by
    # the CSF list they passed in.
    start: dict[tuple, int] = {}
    col = 0
    for pattern in canonical_patterns:
        start[pattern] = col
        col += len(groups[pattern])
    seen: dict[tuple, int] = {}
    perm = []
    for sig in signatures:
        stop = seen.get(sig.pattern, 0)
        seen[sig.pattern] = stop + 1
        perm.append(start[sig.pattern] + stop)
    sig_groups = tuple(tuple(groups[p]) for p in canonical_patterns)
    csf_mat = csf_block_matrix(canonical_patterns, j2, sig_groups, sparse=True)
    if verify:
        _verify_cached(canonical_patterns, j2, sig_groups)
    if angular_storage == 'streamed':
        h_det = _streamed_determinant_matrix(
            dets, labels, evaluate_integral, breit_evaluator=breit_evaluator,
            integral_dtype=integral_dtype, radial_symmetry=radial_symmetry)
    else:
        h_det = evaluate_block(structure, evaluate_integral, sparse=True, integral_dtype=integral_dtype)
        if breit_evaluator is not None:
            b_struct = breit_block_terms(canonical_patterns, j2,
                                         max_determinants, radial_symmetry=radial_symmetry)
            h_det = h_det + evaluate_block(b_struct, breit_evaluator, sparse=True, integral_dtype=integral_dtype)
    if as_operator:
        projection = csr_matrix(csf_mat[:, perm])
        # Symmetry removes accumulation roundoff only; no dense determinant
        # or CSF Hamiltonian is formed on this execution path.
        h_det = 0.5 * (h_det + h_det.T)
        # A CSF column has support in one occupation pattern. Cross-pattern
        # H entries therefore cannot contribute to its diagonal; keep their
        # full action while avoiding a large global H_det @ projection here.
        diagonal = np.zeros(len(perm), dtype=h_det.dtype)
        for lo, hi in slices:
            local = projection[lo:hi]
            diagonal += np.asarray(local.multiply(
                h_det[lo:hi, lo:hi] @ local).sum(axis=0)).ravel()

        def apply(v):
            return projection.T @ (h_det @ (projection @ v))

        def adjoint(v):
            return projection.T @ (h_det.conjugate().T @ (projection @ v))

        operator = LinearOperator((len(perm), len(perm)), matvec=apply,
                                  matmat=apply, rmatvec=adjoint, dtype=h_det.dtype)
        operator.diagonal = lambda: diagonal.copy()
        return operator
    h = (csf_mat.T @ (h_det @ csf_mat)).toarray()
    h = h[np.ix_(perm, perm)]
    return 0.5 * (h + h.T)
