"""Kernel-level gates for the exact jj CI (``atomx.methods.mcdhf.exact_ci``).

The end-to-end gates in ``test_ci_matrix_vs_mscheme_reference.py`` compare
whole spectra against the independent oracle.  These tests pin the pieces
that comparison cannot localise on its own:

* the production angular kernel equals the oracle's sympy 3j algebra
  spinor by spinor;
* the Slater-Condon term generation equals a brute-force second-quantised
  evaluation over the same determinants (synthetic radial data, so no SCF
  is needed);
* the CSF expansion produces orthonormal vectors, and CSFs that differ
  only in an *intermediate* coupling are exactly orthogonal — the property
  the old ``#1``/``#2`` placeholder CSFs could not express;
* the CSF generator emits one CSF per sequential coupling path, with the
  same total count and the same multiset of total ``2J`` as before;
* the guards raise actionable errors instead of approximating.
"""

from __future__ import annotations

import hashlib
import itertools
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

from mscheme_ci_reference import _angular_terms, mscheme_levels  # noqa: E402

import atomx.methods.mcdhf.matrix as MAT  # noqa: E402
from atomx.methods.dirac_fock import Subshell  # noqa: E402
from atomx.methods.mcdhf import exact_ci  # noqa: E402
from atomx.methods.mcdhf.config import (  # noqa: E402
    CSF,
    count_j_resolved_csfs,
    enumerate_jj_csfs,
)


def _fake(*args) -> float:
    """Deterministic pseudo-random stand-in for a radial integral."""
    digest = hashlib.sha256(repr(args).encode()).digest()
    return int.from_bytes(digest[:8], "little") / 2 ** 63 - 0.5


def _evaluate(key) -> float:
    """Feed :func:`exact_ci.evaluate_block` from the same fake values."""
    if key[0] == "h":
        _, n_a, kappa, n_c = key
        return _fake("h", n_a, kappa, n_c)
    _, k, key_a, key_c, key_b, key_d = key
    return _fake("rk", key_a, key_c, key_b, key_d, k)


# ---------------------------------------------------------------------------
# Angular kernel
# ---------------------------------------------------------------------------

KAPPAS = (1, -1, 2, -2, 3, -3, 4)


def test_angular_kernel_matches_the_sympy_oracle():
    """``two_body_angular`` must equal ``_angular_terms`` term for term.

    Both evaluate the same Wigner-Eckart expression; the oracle takes its
    3j symbols from sympy, the production kernel from
    :func:`atomx.angular.wigner.wigner_3j`.  The spinor quadruples are
    enumerated exhaustively and then sampled with a fixed stride (sympy is
    ~2 ms per symbol, so the full cross product would dominate the suite).
    """
    combos = []
    for ka in KAPPAS:
        for kb in KAPPAS:
            for kc in KAPPAS:
                for kd in KAPPAS:
                    for m2a in exact_ci._m2_values(exact_ci.twoj(ka)):
                        for m2b in exact_ci._m2_values(exact_ci.twoj(kb)):
                            for m2c in exact_ci._m2_values(
                                    exact_ci.twoj(kc)):
                                m2d = m2a + m2b - m2c
                                if abs(m2d) > exact_ci.twoj(kd):
                                    continue
                                combos.append((ka, m2a, kb, m2b,
                                               kc, m2c, kd, m2d))
    assert len(combos) > 5000
    stride = max(1, len(combos) // 900)
    checked = 0
    nonempty = 0
    for combo in combos[::stride]:
        mine = dict(exact_ci.two_body_angular(*combo))
        theirs = dict(_angular_terms(*combo))
        nonempty += bool(theirs)
        # sympy returns an exact 0.0 for a vanishing 3j where the numeric
        # kernel returns ~1e-17, so compare over the union of k values with
        # an absolute tolerance.
        for k in set(mine) | set(theirs):
            assert abs(mine.get(k, 0.0) - theirs.get(k, 0.0)) < 1e-12, (
                f"k={k} for {combo}")
        checked += 1
    assert checked >= 900
    assert nonempty > 100, "the sample must include non-vanishing elements"


# ---------------------------------------------------------------------------
# Slater-Condon term generation vs brute-force second quantisation
# ---------------------------------------------------------------------------

def _brute_force_block(pattern, M2, monkeypatch):
    """Oracle-assembled ``H`` in one pattern's ``2M`` determinant space.

    Uses the oracle's own ``mscheme_levels`` with the radial helpers
    monkeypatched to the synthetic values :func:`_evaluate` returns, so
    only the *assembly* is under test.
    """
    # ``orbitals_dict`` values are the (n, kappa) pairs themselves, so the
    # oracle's orbital arguments carry the labels the fake values key on.
    monkeypatch.setattr(MAT, "_h_one_body",
                        lambda na, ka, nc, kc, kappa, Z, bsp, ns:
                        _fake("h", na, kappa, nc))
    monkeypatch.setattr(MAT, "_norm_sq", lambda *a, **k: 1.0)
    monkeypatch.setattr(MAT, "_rk",
                        lambda na, ka, nc, kc, nb, kb, nd, kd, bsp, k:
                        _fake("rk", (na, ka), (nc, kc), (nb, kb), (nd, kd), k))
    orbitals = {(n, kappa): (n, kappa) for (n, kappa, _q) in pattern}
    evals, n_det = mscheme_levels(orbitals, None, 1, 1, [pattern], M2=M2)
    return np.sort(evals), n_det


PROBE_PATTERNS = [
    ((1, -1, 2), (2, -1, 2)),
    ((1, -1, 2), (2, -1, 1), (2, -2, 1)),
    ((1, -1, 2), (2, -1, 1), (2, 1, 1), (3, -2, 1)),
    ((2, -2, 3),),
    ((1, -1, 1), (2, -1, 1), (2, -2, 1), (3, -2, 1)),
]


@pytest.mark.parametrize("pattern", PROBE_PATTERNS,
                         ids=lambda p: "-".join(f"{n}{k}^{q}" for n, k, q in p))
def test_slater_condon_terms_match_brute_force(pattern, monkeypatch):
    """The term table must reproduce a brute-force m-scheme Hamiltonian.

    ``evaluate_block`` builds ``<det_a|H|det_b>`` from a cached
    ``(radial key, coefficient)`` table (0/1/2 spinor differences, explicit
    signs).  The oracle builds the same matrix by applying every
    second-quantised operator to every determinant.  On identical synthetic
    radial data the two spectra must be identical.
    """
    for M2 in range(-6, 7, 2):
        if not exact_ci.pattern_layout(pattern).by_m2.get(M2):
            continue
        structure = exact_ci.block_terms([pattern], M2)
        mine = np.sort(np.linalg.eigvalsh(
            exact_ci.evaluate_block(structure, _evaluate)))
        theirs, n_det = _brute_force_block(pattern, M2, monkeypatch)
        assert len(mine) == n_det == len(structure.dets)
        assert np.abs(mine - theirs).max() < 1e-12, (
            f"{pattern} at 2M = {M2}: term table differs from brute force")


# ---------------------------------------------------------------------------
# CSF vectors
# ---------------------------------------------------------------------------

def _csf_specs(config_str):
    out = []
    for csf in enumerate_jj_csfs(config_str):
        subshells = tuple((sh.n, sh.kappa, int(round(sh.occupation)))
                          for sh in csf.subshells)
        out.append(exact_ci.CSFSignature(subshells, csf.total_J2, csf.coupling))
    return out


def test_csf_vectors_are_orthonormal_per_block():
    for config in ("1s2 2s1 2p1", "1s2 2s1 2p1 3d1", "1s2 2p3"):
        specs = _csf_specs(config)
        by_block: dict[tuple, list] = {}
        for spec in specs:
            by_block.setdefault((spec.total_J2, spec.pattern), []).append(spec)
        for (j2, pattern), group in by_block.items():
            mat = exact_ci.csf_block_matrix([pattern], j2, [group])
            gram = mat.T @ mat
            assert np.abs(gram - np.eye(len(group))).max() < 1e-10, (
                f"{config}: CSFs of {pattern} at 2J = {j2} are not "
                f"orthonormal")
        # the block as a whole (all patterns at once) must be orthonormal too
        by_j: dict[int, list] = {}
        for spec in specs:
            by_j.setdefault(spec.total_J2, []).append(spec)
        for j2, group in by_j.items():
            patterns = []
            groups = []
            for spec in group:
                if spec.pattern not in patterns:
                    patterns.append(spec.pattern)
                    groups.append([])
                groups[patterns.index(spec.pattern)].append(spec)
            mat = exact_ci.csf_block_matrix(patterns, j2, groups)
            assert np.abs(mat.T @ mat - np.eye(len(group))).max() < 1e-10
        # and the J^2 span check (dimension + containment) must pass
        for j2, group in by_j.items():
            patterns = []
            groups = []
            for spec in group:
                if spec.pattern not in patterns:
                    patterns.append(spec.pattern)
                    groups.append([])
                groups[patterns.index(spec.pattern)].append(spec)
            mat = exact_ci.csf_block_matrix(patterns, j2, groups)
            exact_ci.verify_csf_span(patterns, j2, mat)


def test_csfs_differing_in_intermediate_coupling_are_orthogonal():
    """Two paths to the same total J are distinct, orthogonal basis vectors.

    The old generator emitted ``#1``/``#2`` placeholders with no tree, so
    these states collapsed onto the same vector — which is why the
    orthonormality check below is the load-bearing one.
    """
    specs = _csf_specs("1s1 2s1 2p1 3d1")
    pairs = 0
    by_key: dict[tuple, list] = {}
    for spec in specs:
        by_key.setdefault((spec.pattern, spec.total_J2), []).append(spec)
    for (_pattern, _j2), group in by_key.items():
        trees = [spec.coupling for spec in group]
        assert len(set(trees)) == len(trees), (
            "two CSFs of the same configuration, total J and coupling tree "
            "were emitted — the CSF list would not be a basis")
        for a, b in itertools.combinations(range(len(group)), 2):
            if group[a].coupling == group[b].coupling:
                continue
            if all(x[0] == y[0] for x, y in
                   zip(group[a].coupling, group[b].coupling,
                                 strict=True)):
                continue        # same cumulative chain: same state
            mat = exact_ci.csf_block_matrix([group[a].pattern], _j2,
                                            [[group[a], group[b]]])
            assert mat.shape[1] == 2
            assert abs(float(mat[:, 0] @ mat[:, 1])) < 1e-10
            pairs += 1
    assert pairs > 0, "no repeated-parentage pair found in the probe config"


@pytest.mark.snapshot
def test_csf_count_and_j_multiset_unchanged():
    # Generator count (audit F10): the per-2J multiset cross-check is the
    # physics-free part; the total == 29 is a frozen count.
    """One CSF per coupling path must not change the space's dimension.

    ``count_j_resolved_csfs`` is an independent counter (it convolves
    subshell ``2J`` multiplicities without enumerating paths), so agreeing
    with it pins both the total count and the per-``2J`` multiplicities.
    """
    for config in ("1s2 2s2", "1s2 2s1 2p1", "1s2 2p2", "1s2 2s1 2p1 3d1",
                   "1s1 2s2 2p1"):
        got = Counter(csf.total_J2 for csf in enumerate_jj_csfs(config))
        assert dict(sorted(got.items())) == count_j_resolved_csfs(config)
    sd_panel = [
        "1s2 2s2", "1s1 2s2 2p1", "1s1 2s2 3s1", "1s1 2s2 3p1",
        "1s1 2s2 3d1", "1s2 2s1 2p1", "1s2 2s1 3s1", "1s2 2s1 3p1",
        "1s2 2s1 3d1",
    ]
    assert sum(len(enumerate_jj_csfs(c)) for c in sd_panel) == 29


def test_labels_keep_the_bracketed_j_tag():
    """Consumers parse ``[J=<2J>/2]`` out of the label (see
    ``matrix._has_explicit_J`` and the transition calculator); the tree
    suffix must be additive."""
    for csf in enumerate_jj_csfs("1s2 2s1 2p1"):
        assert f"[J={csf.total_J2}/2]" in csf.label
        assert "tree=" in csf.label
        assert csf.coupling[-1][0] == csf.total_J2


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_determinant_cap_raises_and_reports_sizes():
    specs = [s for s in _csf_specs("1s2 2s1 2p1 3d1")]
    j2 = specs[0].total_J2
    group = [s for s in specs if s.total_J2 == j2]
    with pytest.raises(exact_ci.ExactCISpaceError) as excinfo:
        exact_ci.exact_block_matrix(group, lambda key: 1.0, max_determinants=1)
    message = str(excinfo.value)
    assert "determinant space" in message
    assert "cap of 1" in message
    assert "Per-configuration sizes" in message


def test_unreachable_total_j_raises():
    """A CSF whose declared ``2J`` no coupling can reach must not be
    silently reinterpreted (that would put a state of the wrong angular
    momentum into the block)."""
    csf = CSF(subshells=[Subshell(1, -1, 2), Subshell(2, -1, 1)],
              total_J2=0, parity=1)
    with pytest.raises(exact_ci.ExactCISpaceError, match="reaches 2J"):
        exact_ci.csf_signature(csf)
    # ... and the physically correct declaration is accepted.
    ok = CSF(subshells=[Subshell(1, -1, 2), Subshell(2, -1, 1)],
             total_J2=1, parity=1)
    assert exact_ci.csf_signature(ok).coupling == ((0, 0, 0), (1, 1, 0))


def test_fractional_occupation_raises():
    csf = CSF(subshells=[Subshell(1, -1, 1.5)], total_J2=0, parity=1)
    with pytest.raises(exact_ci.ExactCISpaceError, match="non-integer"):
        exact_ci.csf_signature(csf)


def test_cached_structures_are_reused_and_deterministic():
    """Repeat builds must reuse the cached tables and be bit-identical."""
    pattern = ((1, -1, 2), (2, -1, 1), (2, -2, 1))
    exact_ci._block_structure.cache_clear()
    first = exact_ci.block_terms([pattern], 2)
    h1 = exact_ci.evaluate_block(first, _evaluate)
    info = exact_ci._block_structure.cache_info()
    second = exact_ci.block_terms([pattern], 2)
    assert second is first
    h2 = exact_ci.evaluate_block(second, _evaluate)
    assert np.array_equal(h1, h2)
    assert exact_ci._block_structure.cache_info().misses == info.misses


def test_degenerate_subshell_j_basis_is_deterministic():
    """Repeated-J subshell terms (here ``(5/2)^3``) may span a degenerate
    subspace; the basis is an SVD choice, so this pins determinism and
    orthonormality rather than any seniority labelling."""
    import numpy as np

    from atomx.angular.mscheme import highest_weight_states

    for q in (3, 4):
        first = highest_weight_states(5, q)
        second = highest_weight_states(5, q)
        assert first == second
        for state in first:
            vec = np.asarray(state.coefficients)
            assert abs(float(vec @ vec) - 1.0) < 1e-10
