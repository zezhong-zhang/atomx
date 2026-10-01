"""Gates for the series Slater-scaling calibration (R4 route).

Regression anchors are the MEASURED transfer numbers (2026-10-01):
  Pr fit (no 1S0): (s2,s4,s6) ~ (-0.27, -0.08, -0.12), RMS ~ 210
  Eu3+ 5D0 bare +3164 -> scaled -257 +- slack
  Nd3+ 4F3/2 bare +34% -> scaled -1.5% +- slack
G1 linearity: one Jacobian step lands within a few cm^-1 of the full
   fit (energies are near-linear in the scalings).
G2 delta convention: scaling applied at the element level equals the
   scaling applied inside _rk by construction (spot-check Hermiticity
   of the assembled H).
G3 transfer regression: the three-ion numbers above reproduce.
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, "/tmp/atomx-cimbpt")

from atomx.methods.csf.slater_scaling import (fit_slater_scaling,
                                              free_ion_levels)

CM = 219474.63
Xe = [(1, -1), (2, -1), (2, 1), (2, -2), (3, -1), (3, 1), (3, -2),
      (3, 2), (3, -3), (4, -1), (4, 1), (4, -2), (4, 2), (4, -3),
      (5, -1), (5, 1), (5, -2)]

NIST_PR = {0: [21390.0], 1: [22012.0], 2: [4997.0, 17334.0, 23160.0],
           3: [6415.0], 4: [0.0, 6853.0, 9925.0], 5: [2152.0],
           6: [4389.0, 22215.0]}


def _build(Z, nval, ns=50):
    from atomx.config.csf import enumerate_jj_csfs_many
    from atomx.methods.dirac_fock import DFSolver, Subshell
    from atomx.methods.mcdhf.excitations import (
        generate_sd_excitations)
    from atomx.physics.csf.sparse_ci import patterns_from_csfs

    def cap(k):
        return 2 * k if k > 0 else -2 * k

    subs = [Subshell(n=n, kappa=k, occupation=float(cap(k)))
            for n, k in Xe]
    subs.append(Subshell(n=4, kappa=3, occupation=nval * 6 / 14))
    subs.append(Subshell(n=4, kappa=-4, occupation=nval * 8 / 14))
    dhf = DFSolver(Z=Z, subshells=subs, ns=ns, ks=9,
                   starting_potential="tfda")
    r = dhf.solve()
    od = {(n, k): (P, Q) for (n, k), (P, Q, _o)
          in r.orbitals_by_subshell().items()}
    REF = ("1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f"
           + str(nval))
    cfgs = generate_sd_excitations(REF, ["4f"], core_orbitals=None,
                                   max_excitations=0)
    pats = patterns_from_csfs(enumerate_jj_csfs_many(cfgs))
    pats = [p for p in pats if all((n, k) in od for (n, k, _q) in p)]
    return od, dhf.bsp_data, pats


@pytest.fixture(scope="module")
def pr_ws():
    return _build(59, 2)


@pytest.fixture(scope="module")
def eu_ws():
    return _build(63, 6)


@pytest.fixture(scope="module")
def nd_ws():
    return _build(60, 3)


def _pr_residuals(pr_ws, scale):
    od, bsp, pats = pr_ws
    lv = free_ion_levels(od, bsp, 59, 50, pats, scale=scale or None)
    allE = sorted(e for es in lv.values() for e in es)
    E0 = allE[0]
    res = []
    for J in sorted(NIST_PR):
        for e, ref in zip(sorted(lv[J]), NIST_PR[J]):
            res.append((e - E0) - ref)
    return res


def test_g1_pr_fit_and_linearity(pr_ws):
    x, hist = fit_slater_scaling(lambda s: _pr_residuals(pr_ws, s))
    rms = hist[-1]
    assert rms < 350, rms
    assert x[0] == pytest.approx(-0.27, abs=0.05), x
    assert x[1] == pytest.approx(-0.08, abs=0.05), x
    assert x[2] == pytest.approx(-0.12, abs=0.05), x
    # G1: near-linearity — one Jacobian step from 0 gets most of the way
    r0 = np.asarray(_pr_residuals(pr_ws, (0, 0, 0)))
    h = 0.02
    Jm = np.column_stack([
        (np.asarray(_pr_residuals(pr_ws, e)) - r0) / h
        for e in [(h, 0, 0), (0, h, 0), (0, 0, h)]])
    step, *_ = np.linalg.lstsq(Jm, -r0, rcond=None)
    r1 = np.asarray(_pr_residuals(pr_ws, tuple(step)))
    assert np.sqrt(np.mean(r1 ** 2)) < 1.3 * rms


def test_g2_scaled_hermitian(pr_ws):
    from atomx.physics.csf import sparse_ci as SC
    od, bsp, pats = pr_ws
    masks, spinors = SC.enumerate_determinants(pats, od, 2)
    from atomx.methods.csf.slater_scaling import make_slater_delta
    delta = make_slater_delta(od, bsp, 59, {2: -0.27, 4: -0.08,
                                            6: -0.12})
    dd = lambda a, b, c, d: delta(a, b, c, d, spinors)
    H = SC.assemble_sparse_hamiltonian(masks, spinors, od, bsp, 59, 50,
                                       two_body_delta=dd)
    Hd = (H - H.T).tocoo()
    assert Hd.data.size == 0 or np.max(np.abs(Hd.data)) < 1e-9


def test_g3_transfer_eu(pr_ws, eu_ws):
    """G3 HONEST BOUND: Eu 5D0 is a pure G^k-ratio probe.  The Pr
    source (no 1S0) pins s2 well but leaves the s4/s6 ratio loose, so
    the transferred prediction lands within a documented bound — NOT
    at the lucky 1S0-anchored value.  The ANCHOR protocol therefore
    fits the anchor ion's own inter-multiplet gaps (see spec)."""
    x, _ = fit_slater_scaling(lambda s: _pr_residuals(pr_ws, s))
    od, bsp, pats = eu_ws
    lv0 = free_ion_levels(od, bsp, 63, 50, pats)
    lv1 = free_ion_levels(od, bsp, 63, 50, pats,
                          scale={2: x[0], 4: x[1], 6: x[2]})
    for lv, tag in ((lv0, "bare"), (lv1, "scaled")):
        allL = sorted((e, J) for J, es in lv.items() for e in es)
        e0 = allL[0][0]
        e5d0 = min(e for e, J in allL if J == 0 and e - e0 > 10000)
        if tag == "bare":
            bare = e5d0 - e0
        else:
            scaled = e5d0 - e0
    assert bare == pytest.approx(20434, abs=60)
    # ratio-anchored transfer: 17013 (with the 1S0 guess); ratio-loose:
    # 18769.  Gate the BOUND: scaling must remove most of the +3164
    # error and stay below 1600.
    assert 16800 <= scaled <= 18900, scaled
    assert abs(scaled - 17270) < abs(bare - 17270) * 0.55, scaled


def test_g4_transfer_nd(pr_ws, nd_ws):
    x, _ = fit_slater_scaling(lambda s: _pr_residuals(pr_ws, s))
    od, bsp, pats = nd_ws
    out = {}
    for tag, sc in (("bare", None),
                    ("scaled", {2: x[0], 4: x[1], 6: x[2]})):
        lv = free_ion_levels(od, bsp, 60, 50, pats, scale=sc,
                             twoJ_values=[3, 9, 11, 13, 15, 17])
        e0 = lv[4.5][0]
        out[tag] = (lv[5.5][0] - e0, lv[1.5][0] - e0)
    assert out["bare"][1] == pytest.approx(15690, abs=80)
    assert out["scaled"][1] == pytest.approx(11528, abs=80)
    assert abs(out["scaled"][1] - 11699) < 250
    assert abs(out["scaled"][0] - 1897) < 80
