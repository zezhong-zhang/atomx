"""The RKB banks are a function of the B-spline grid, so a process builds
each of them once — not once per DFSolver.

The MCDHF driver constructs a fresh ``DFSolver`` (and therefore a fresh
``ConvolCache``) on every outer iteration — 5 per Fe solve — so the old
per-instance cache rebuilt all 20 banks 5 times with bit-identical
content: 100 bank builds per solve (instrumented), measured at 12.6 % of
the wall time at ns=60 and 13.8 % at ns=113.  Keying the registry on grid
*content* (never on ``id()``: the five ``bsp_data`` dicts are distinct
objects with bit-identical content) drops that to 20.

See ``docs/PHYSICS_NUMERICS_REVIEW_2026-09-19.md`` (performance item C7 /
TASK 1 of the remediation).
"""

from __future__ import annotations

import collections

import numpy as np
import pytest

from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.numerics.bspline import convol
from atomx.numerics.bspline.convol import ConvolCache, grid_bank_key

_BUILDERS = ("_build_rkb_sym", "_build_rkb_pqpq", "_build_rkb_qpqp")


@pytest.fixture
def bank_builds(monkeypatch):
    """Count every rkb bank build and start from an empty registry."""
    counts: collections.Counter = collections.Counter()
    for name in _BUILDERS:
        original = getattr(convol, name)

        def wrapper(*args, _name=name, _orig=original, **kwargs):
            counts[_name] += 1
            return _orig(*args, **kwargs)

        monkeypatch.setattr(convol, name, wrapper)
    convol._RKB_BANKS.clear()
    yield counts
    convol._RKB_BANKS.clear()


def _workspace(ns=30, ks=6, Z=3):
    """A DFSolver (hence a fresh bsp_data dict) without running the SCF."""
    subs = [Subshell(n=1, kappa=-1, occupation=2.0),
            Subshell(n=2, kappa=-1, occupation=1.0)]
    return DFSolver(Z=Z, subshells=subs, ns=ns, ks=ks)


def test_rebuilt_but_identical_workspaces_share_banks(bank_builds):
    a, b = _workspace(), _workspace()
    assert a.bsp_data is not b.bsp_data                       # distinct objects
    assert grid_bank_key(a.bsp_data) == grid_bank_key(b.bsp_data)  # same content

    cache_a, cache_b = ConvolCache(a.bsp_data), ConvolCache(b.bsp_data)
    assert cache_a._cache is cache_b._cache

    bank_a, _, _ = cache_a.get_rkb(2, "pppp")
    assert bank_builds["_build_rkb_sym"] == 1
    bank_b, _, _ = cache_b.get_rkb(2, "pppp")
    assert bank_builds["_build_rkb_sym"] == 1
    assert bank_a is bank_b        # bit-identical by construction


def test_transposed_banks_are_shared_too(bank_builds):
    a, b = _workspace(), _workspace()
    cache_a, cache_b = ConvolCache(a.bsp_data), ConvolCache(b.bsp_data)
    bank_t, _, _ = cache_a.get_rkb_t(1, "pppp")
    other_t, _, _ = cache_b.get_rkb_t(1, "pppp")
    assert bank_t is other_t
    assert bank_builds["_build_rkb_sym"] == 1


def test_large_basis_keeps_one_bank_copy_and_preserves_raw_integrals(
        bank_builds, monkeypatch):
    # Force the large-basis policy on a small workspace so this checks the
    # memory layout without allocating a Rydberg-sized integral bank.
    monkeypatch.setattr(convol, "_TRANSPOSE_ONLY_NS", 30)
    cache = ConvolCache(_workspace(ns=30).bsp_data)
    raw, ns, ks = cache.get_rkb(0, "pppp")
    reference = raw.copy()
    transposed, _, _ = cache.get_rkb_t(0, "pppp")
    recovered, _, _ = cache.get_rkb(0, "pppp")

    assert (0, "pppp") not in cache._cache
    assert np.shares_memory(recovered, transposed)
    np.testing.assert_array_equal(recovered, reference)
    assert bank_builds["_build_rkb_sym"] == 1


def test_five_rebuilt_solvers_build_each_bank_once(bank_builds):
    """The measured 100 → 20 bank builds of an MCDHF solve, in miniature."""
    keys = [(k, itype) for k in range(5)
            for itype in ("pppp", "qqqq", "pqpq", "qpqp")]
    for _ in range(5):                      # five outer iterations
        cache = ConvolCache(_workspace().bsp_data)
        for k, itype in keys:
            cache.get_rkb(k, itype)
    assert sum(bank_builds.values()) == 20          # was 5 × 20 = 100


def test_a_different_grid_gets_its_own_banks(bank_builds):
    a, b = _workspace(ns=30), _workspace(ns=34)
    assert grid_bank_key(a.bsp_data) != grid_bank_key(b.bsp_data)
    ConvolCache(a.bsp_data).get_rkb(0, "pppp")
    ConvolCache(b.bsp_data).get_rkb(0, "pppp")
    assert bank_builds["_build_rkb_sym"] == 2


def test_registry_is_bounded(bank_builds):
    for ns in (24, 26, 28, 30, 32, 34, 36):
        ConvolCache(_workspace(ns=ns).bsp_data)
    assert len(convol._RKB_BANKS) <= convol._BANK_CACHE_LIMIT


def test_ci_build_keeps_the_stashed_cache(bank_builds):
    """``build_ci_matrix`` must not drop the cache it was handed.

    The CI build calls ``_reset_memo()`` on entry, which used to discard
    the stashed ``_convol_cache``; it is now carried across the reset, and
    a rebuilt-but-identical workspace still matches by content, so a
    second build over the same grid rebuilds no bank.
    """
    from atomx.methods.mcdhf.config import (
        get_shared_orbital_basis,
        make_csf_list,
    )
    from atomx.methods.mcdhf.matrix import build_ci_matrix

    dhf = _workspace()
    cache = ConvolCache(dhf.bsp_data)
    csfs = make_csf_list(Z=3, config_strings=["1s2 2s1"])
    # ``make_csf_list`` leaves total_J2 at its default 0, which no coupling
    # of 1s2 2s1 can reach (a 2s electron carries J = 1/2).  The exact CI
    # needs a CSF of definite J, so declare it: 1s2 2s1 is 2S_1/2.
    csfs[0].total_J2 = 1
    from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices, solve_dirac
    h, s = build_dirac_matrices(dhf.ns, -1, 3, dhf.bsp_data)
    roots = solve_dirac(h, s, dhf.ns, -1,
                        nsp=dhf.bsp_data.get("nsp"), nsq=dhf.bsp_data.get("nsq"))
    orbitals = {(n, -1): roots.get_orbital(n - 1)[:2] for n in (1, 2)}
    basis = get_shared_orbital_basis(csfs)

    build_ci_matrix(csfs, orbitals, basis, dhf.bsp_data, Z=3,
                    ns=dhf.ns, convol_cache=cache)
    builds_after_first = sum(bank_builds.values())
    assert builds_after_first > 0        # the first build had work to do

    # Second build, same grid handed over as a *rebuilt* (equal-content)
    # dict: every bank it needs is already in the shared registry.
    build_ci_matrix(csfs, orbitals, basis, dict(dhf.bsp_data), Z=3,
                    ns=dhf.ns, convol_cache=cache)
    assert sum(bank_builds.values()) == builds_after_first
