"""NaN tripwires on the SCF hot paths (HYP-1590).

Under extreme CPU oversubscription (box load > 150 with many concurrent
runs), single numerical calls on the DHF/B-spline path were observed to
return NaN from finite inputs while an immediate identical re-evaluation
recovered.  This produced load-sensitive flaky failures in the
``he_dhf`` and ``pair_ladder`` test families (NaN total energy with
``converged=True``, ``nan == nan`` in the 1x1 closed form) that no
single-scope rerun reproduced.

The sentinels added for HYP-1590:

- :func:`compute_slater_rk_cell` retries a non-finite R^k evaluation
  (bounded) and raises ``FloatingPointError`` if corruption persists.
- :func:`solve_dirac` retries a non-finite ``eigh`` spectrum and raises
  ``LinAlgError`` if it persists.
- :meth:`DFSolver._final_energy_checked` retries a non-finite final
  total energy and raises ``FloatingPointError`` if it persists.

These tests simulate transient and persistent NaN at each layer with
monkeypatching (deterministic — no load required), plus one end-to-end
finiteness sentinel on the real he_dhf solve.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from scipy import linalg

from atomx import Atom
from atomx.methods.dirac_fock import DFSolver
from atomx.numerics.bspline import convol, dirac_assembly
from atomx.numerics.bspline.convol import ConvolCache, compute_slater_rk_cell
from atomx.numerics.bspline.dirac_assembly import (
    build_dirac_matrices,
    solve_dirac,
)

# ---------------------------------------------------------------------------
# Shared, tiny real inputs (he_dhf-scale basis, built once per module)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def he_result():
    atom = Atom(Z=2, numerics="bspline")
    atom.solve()
    return atom._result


# ---------------------------------------------------------------------------
# compute_slater_rk_cell tripwire
# ---------------------------------------------------------------------------


def test_slater_rk_recovers_from_transient_nan(he_result, monkeypatch):
    res = he_result
    cache = ConvolCache(res._bsp_data)
    P, Q, _ = res.orbitals_raw[-1][0]
    args = (P, Q, P, Q, P, Q, P, Q, res._bsp_data, 0)

    real_once = convol._compute_slater_rk_cell_once
    calls = {"n": 0}

    def flaky_once(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return float("nan")   # transient corruption on first call
        return real_once(*a, **kw)

    monkeypatch.setattr(convol, "_compute_slater_rk_cell_once", flaky_once)
    value = compute_slater_rk_cell(*args, cache=cache)
    assert np.isfinite(value)
    assert value == pytest.approx(real_once(*args, cache=cache), rel=1e-14)
    assert calls["n"] == 2   # exactly one retry


def test_slater_rk_raises_on_persistent_nan(he_result, monkeypatch):
    res = he_result
    cache = ConvolCache(res._bsp_data)
    P, Q, _ = res.orbitals_raw[-1][0]
    args = (P, Q, P, Q, P, Q, P, Q, res._bsp_data, 0)

    monkeypatch.setattr(
        convol, "_compute_slater_rk_cell_once",
        lambda *a, **kw: float("nan"))
    with pytest.raises(FloatingPointError, match="HYP-1590"):
        compute_slater_rk_cell(*args, cache=cache)


# ---------------------------------------------------------------------------
# solve_dirac tripwire
# ---------------------------------------------------------------------------


def test_solve_dirac_recovers_from_transient_nan(he_result, monkeypatch):
    bsp = he_result._bsp_data
    ns = bsp["ns"]
    H, S = build_dirac_matrices(ns, -1, 2, bsp)
    real_eigh = dirac_assembly.linalg.eigh
    calls = {"n": 0}

    def flaky_eigh(H_red, S_red, **kw):
        calls["n"] += 1
        w, v = real_eigh(H_red, S_red, **kw)
        if calls["n"] == 1:
            w = w.copy()
            w[0] = np.nan   # transient corruption of the spectrum
        return w, v

    monkeypatch.setattr(dirac_assembly.linalg, "eigh", flaky_eigh)
    result = solve_dirac(H, S, ns, -1)
    assert np.all(np.isfinite(result.eigenvalues))
    assert result.n_bound >= 1
    assert calls["n"] == 2


def test_solve_dirac_raises_on_persistent_nan(he_result, monkeypatch):
    bsp = he_result._bsp_data
    ns = bsp["ns"]
    H, S = build_dirac_matrices(ns, -1, 2, bsp)
    real_eigh = dirac_assembly.linalg.eigh

    def nan_eigh(H_red, S_red, **kw):
        w, v = real_eigh(H_red, S_red, **kw)
        w = w.copy()
        w[0] = np.nan
        return w, v

    monkeypatch.setattr(dirac_assembly.linalg, "eigh", nan_eigh)
    with pytest.raises(linalg.LinAlgError, match="HYP-1590"):
        solve_dirac(H, S, ns, -1)


# ---------------------------------------------------------------------------
# DFSolver final-energy tripwire (end-to-end through solve())
# ---------------------------------------------------------------------------


def test_dhf_solve_recovers_from_transient_nan(monkeypatch):
    real_cte = DFSolver._compute_total_energy
    calls = {"n": 0}

    def flaky_cte(self, orbitals, orbital_energies=None):
        calls["n"] += 1
        E, comp = real_cte(self, orbitals, orbital_energies)
        if calls["n"] == 1:
            return float("nan"), comp   # transient corruption
        return E, comp

    monkeypatch.setattr(DFSolver, "_compute_total_energy", flaky_cte)
    atom = Atom(Z=2, numerics="bspline")
    atom.solve()
    assert np.isfinite(atom._result.total_energy)
    assert atom._result.converged


def test_dhf_solve_raises_on_persistent_nan(monkeypatch):
    monkeypatch.setattr(
        DFSolver, "_compute_total_energy",
        lambda self, orbitals, orbital_energies=None: (
            float("nan"), {"E_1e": float("nan")}))
    atom = Atom(Z=2, numerics="bspline")
    with pytest.raises(FloatingPointError, match="HYP-1590"):
        atom.solve()


# ---------------------------------------------------------------------------
# Deterministic end-to-end finiteness sentinel on the real he_dhf solve
# (fails loudly in CI if NaN ever reaches a delivered SCFResult again)
# ---------------------------------------------------------------------------


def test_he_dhf_result_is_finite(he_result):
    res = he_result
    assert res.converged
    assert np.isfinite(res.total_energy)
    assert res.energy_components is not None
    for name, value in res.energy_components.items():
        if value is None or isinstance(value, str):
            continue
        assert np.isfinite(value), f"energy_components[{name!r}] = {value}"
    for label, energy in res.orbital_energies.items():
        assert np.isfinite(energy), f"orbital_energies[{label!r}] = {energy}"
    for kappa, orb_list in res.orbitals_raw.items():
        for idx, (P, Q, _occ) in enumerate(orb_list):
            assert np.all(np.isfinite(P)), f"kappa={kappa}[{idx}]: NaN in P"
            assert np.all(np.isfinite(Q)), f"kappa={kappa}[{idx}]: NaN in Q"


# ---------------------------------------------------------------------------
# HYP-1590 root cause: _convol_case2_mixed out-of-bounds density read
#
# Under kinetic balance nsp = ns - 1, but the kernel looped the left-pair
# contraction over ns rows of a density with only nsp rows.  Numba
# nopython mode has no bounds checking, so the kernel multiplied zero
# R^k rows by whatever garbage float64s followed the array on the heap —
# benign until the garbage was NaN (0 * NaN = NaN), which is why the
# he_dhf / pair_ladder families flaked only under heavy box load.
# ---------------------------------------------------------------------------


def _case2_mixed_reference(rkb, d_left, ns, ks_left, ks_right):
    """Pure-python reference: the documented exact summation domain."""
    n_left = d_left.shape[0]
    a = np.zeros((ns, ks_right))
    for j in range(ns):
        for jp in range(ks_right):
            if j + jp >= ns:
                continue
            s = 0.0
            for ip in range(ks_left):
                for i in range(n_left):
                    if i + ip >= n_left:
                        continue
                    s += rkb[i, j, ip, jp] * d_left[i, ip]
            a[j, jp] = s
    return a


def test_convol_case2_mixed_trims_to_density_rows():
    """Result must be exact when the density has fewer rows than ns —
    the R^k rows beyond the density basis are exactly zero, so the
    contraction must trim to ``d_left.shape[0]`` (never reading past it).
    """
    rng = np.random.default_rng(20260827)
    ns, n_left, ks_left, ks_right = 7, 5, 3, 4
    rkb = rng.standard_normal((ns, ns, ks_left, ks_right))
    # Built exactly zero beyond the left density basis, as the real
    # _build_rkb_* banks are (pairs i, i+ip with i+ip >= n_left do not
    # exist) — this is what makes trimming exact.
    for i in range(ns):
        for ip in range(ks_left):
            if i + ip >= n_left:
                rkb[i, :, ip, :] = 0.0
    d_left = rng.standard_normal((n_left, ks_left))
    # zero the band corner the same way _density_sym does
    for i in range(n_left):
        for ip in range(ks_left):
            if i + ip >= n_left:
                d_left[i, ip] = 0.0

    got = convol._convol_case2_mixed(rkb, d_left, ns, ks_left, ks_right)
    want = _case2_mixed_reference(rkb, d_left, ns, ks_left, ks_right)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-14)


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="guard-page test requires linux mmap/mprotect")
def test_convol_case2_mixed_never_reads_past_density_rows():
    """Place ``d_left`` so its last byte abuts a PROT_NONE guard page.

    Any read past the end of the density array faults deterministically
    (pre-HYP-1590 the kernel looped over ns > d_left rows and would
    segfault here); post-fix the call completes and matches the
    reference.
    """
    import ctypes
    import mmap

    ns, n_left, ks_left, ks_right = 7, 5, 3, 4
    rng = np.random.default_rng(7)
    rkb = rng.standard_normal((ns, ns, ks_left, ks_right))
    for i in range(ns):
        for ip in range(ks_left):
            if i + ip >= n_left:
                rkb[i, :, ip, :] = 0.0
    d_ref = rng.standard_normal((n_left, ks_left))

    d_bytes = n_left * ks_left * 8
    assert d_bytes <= mmap.PAGESIZE
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                          ctypes.c_int, ctypes.c_int, ctypes.c_int,
                          ctypes.c_long]
    libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                              ctypes.c_int]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    PROT_NONE, PROT_READ, PROT_WRITE = 0, 1, 2
    MAP_PRIVATE, MAP_ANONYMOUS = 0x02, 0x20
    base = libc.mmap(None, 2 * mmap.PAGESIZE, PROT_NONE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
    assert base not in (None, -1)
    try:
        assert libc.mprotect(ctypes.c_void_p(base), mmap.PAGESIZE,
                             PROT_READ | PROT_WRITE) == 0
        buf = (ctypes.c_char * mmap.PAGESIZE).from_address(base)
        offset = mmap.PAGESIZE - d_bytes
        d_left = np.frombuffer(
            buf, dtype=np.float64, count=n_left * ks_left,
            offset=offset).reshape(n_left, ks_left)
        d_left[:] = d_ref  # array now ends exactly at the guard page

        got = convol._convol_case2_mixed(rkb, d_left, ns, ks_left, ks_right)
    finally:
        libc.munmap(ctypes.c_void_p(base), 2 * mmap.PAGESIZE)

    want = _case2_mixed_reference(rkb, d_ref, ns, ks_left, ks_right)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-14)


# ---------------------------------------------------------------------------
# HYP-1608 review: seeding guard must fail loud (positional indexing makes
# skip-and-rebuild unsound, and the tfda/gsz path has no SCF sweep);
# tfda/gsz final energy must pass through the same tripwire.
# ---------------------------------------------------------------------------


def test_dhf_raises_on_poisoned_hydrogenic_seed(monkeypatch):
    """A NaN eigenvector from the initial eigensolve raises immediately —
    a silent skip would misalign positional orbital indexing and (in the
    tfda/gsz path) deliver a wrong energy with converged=True."""
    from atomx.numerics.bspline.dirac_assembly import DiracResult

    real_get = DiracResult.get_orbital
    poisoned = {"done": False}

    def poison_get_orbital(self, idx):
        P, Q, E = real_get(self, idx)
        if not poisoned["done"]:
            poisoned["done"] = True
            P = P.copy()
            P[0] = np.nan   # NaN eigenvector with finite eigenvalue
        return P, Q, E

    monkeypatch.setattr(DiracResult, "get_orbital", poison_get_orbital)
    atom = Atom(Z=2, numerics="bspline")
    with pytest.raises(FloatingPointError, match="HYP-1590"):
        atom.solve()


def test_tfda_final_energy_uses_tripwire(monkeypatch):
    """The parametric (tfda) early return routes through
    _final_energy_checked: transient NaN recovers, persistent raises."""
    real_cte = DFSolver._compute_total_energy
    calls = {"n": 0}

    def flaky_cte(self, orbitals, orbital_energies=None):
        calls["n"] += 1
        E, comp = real_cte(self, orbitals, orbital_energies)
        if calls["n"] == 1:
            return float("nan"), comp
        return E, comp

    monkeypatch.setattr(DFSolver, "_compute_total_energy", flaky_cte)
    atom = Atom(Z=2, numerics="bspline", exchange="tfda")
    atom.solve()
    assert np.isfinite(atom._result.total_energy)
    assert calls["n"] >= 2   # the first NaN evaluation was retried


def test_tfda_raises_on_persistent_nan(monkeypatch):
    monkeypatch.setattr(
        DFSolver, "_compute_total_energy",
        lambda self, orbitals, orbital_energies=None: (
            float("nan"), {"E_1e": float("nan")}))
    atom = Atom(Z=2, numerics="bspline", exchange="tfda")
    with pytest.raises(FloatingPointError, match="HYP-1590"):
        atom.solve()
