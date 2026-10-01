"""Independent operator/state-space regressions from the October 2026 review."""

from functools import lru_cache
from itertools import combinations, product
from unittest.mock import patch

import numpy as np
import pytest
from test_mbpt_sigma2 import _VALENCE, _VIRT
from test_mbpt_sigma2 import workspace as base_workspace

import atomx.physics.csf.matrix as MAT
from atomx.angular.clebsch import clebsch_gordan
from atomx.angular.csf_hamiltonian import two_body_angular, twoj
from atomx.config.csf import enumerate_jj_csfs_many
from atomx.methods.csf.crystal_field import build_cf_operator
from atomx.methods.csf.slater_scaling import fit_slater_scaling, make_slater_delta
from atomx.methods.mbpt_sigma2 import build_sigma2_holeline, build_sigma2_mscheme
from atomx.methods.rpa import solve_rpa
from atomx.physics.csf import sparse_ci as SC


@pytest.fixture(scope="module")
def workspace():
    return base_workspace.__wrapped__()


def _coulomb(od, data):
    @lru_cache(maxsize=None)
    def g(a, b, c, d):
        Pa, Qa = od[a[:2]]
        Pb, Qb = od[b[:2]]
        Pc, Qc = od[c[:2]]
        Pd, Qd = od[d[:2]]
        return sum(
            co * MAT._rk(Pa, Qa, Pc, Qc, Pb, Qb, Pd, Qd, data, k)
            for k, co in two_body_angular(
                a[1], a[2], b[1], b[2], c[1], c[2], d[1], d[2]
            )
        )

    return g


def test_sector_cache_tracks_in_place_orbital_changes(workspace):
    dhf, od, _ = workspace
    od = {key: (P.copy(), Q.copy()) for key, (P, Q) in od.items()}
    pats = SC.patterns_from_csfs(enumerate_jj_csfs_many(["1s2 2s2", "1s2 2p2"]))
    SC._SECTOR_CACHE.clear()
    initial, _ = SC.sparse_ci_spectrum(od, dhf.bsp_data, 4, dhf.ns, pats, n_states=100)
    for component in range(2):
        od[(2, -1)][component][:] = (
            np.cos(0.2) * od[(2, -1)][component] + np.sin(0.2) * od[(3, -1)][component]
        )
    cached, _ = SC.sparse_ci_spectrum(od, dhf.bsp_data, 4, dhf.ns, pats, n_states=100)
    SC._SECTOR_CACHE.clear()
    rebuilt, _ = SC.sparse_ci_spectrum(od, dhf.bsp_data, 4, dhf.ns, pats, n_states=100)
    assert abs(initial[0] - rebuilt[0]) > 0.1
    np.testing.assert_allclose(cached, rebuilt, atol=1e-13, rtol=0)
    # The grid fingerprint includes in-place quadrature changes as well.
    data = dict(dhf.bsp_data)
    data["grw"] = data["grw"].copy()
    before = SC._sector_fingerprint(od, data)
    data["grw"][0, 0] *= 1.01
    assert before != SC._sector_fingerprint(od, data)


@pytest.mark.parametrize("method", ["python", "numba"])
@pytest.mark.parametrize(
    "bkq,pats",
    [
        ({(2, 2): 3000.0, (2, -2): 3000.0}, (((2, -2, 1),),)),
        ({(2, 0): 5000.0}, (((2, 1, 1),), ((2, -2, 1),))),
    ],
)
def test_cf_general_one_body_matrix_equals_one_electron_operator(
    workspace, method, bkq, pats
):
    dhf, od, _ = workspace
    masks, spinors = SC.enumerate_determinants(pats, od, None)
    cf = build_cf_operator(od, dhf.bsp_data, 4, dhf.ns, bkq)

    def delta(a, c):
        return float(cf(cf.index[spinors[a]], cf.index[spinors[c]]))

    kwargs = dict(method=method)
    H0 = SC.assemble_sparse_hamiltonian(
        masks, spinors, od, dhf.bsp_data, 4, dhf.ns, **kwargs
    )
    H1 = SC.assemble_sparse_hamiltonian(
        masks, spinors, od, dhf.bsp_data, 4, dhf.ns, one_body_m_delta=delta, **kwargs
    )
    # A one-electron determinant has exactly the operator's spinor matrix.
    indices = [mask.bit_length() - 1 for mask in masks]
    expected = np.array([[delta(a, c) for c in indices] for a in indices])
    assert np.max(abs(expected - np.diag(np.diag(expected)))) > 0.01
    np.testing.assert_allclose((H1 - H0).toarray(), expected, atol=1e-14, rtol=0)


def test_slater_hook_matches_scaling_radial_kernel(workspace):
    dhf, od, _ = workspace
    pats = SC.patterns_from_csfs(enumerate_jj_csfs_many(["1s2 2s2", "1s2 2p2"]))
    masks, spinors = SC.enumerate_determinants(pats, od, 0)
    scale = {1: 0.2, 2: 0.3}
    delta = make_slater_delta(od, dhf.bsp_data, scale)
    H = SC.assemble_sparse_hamiltonian(
        masks,
        spinors,
        od,
        dhf.bsp_data,
        4,
        dhf.ns,
        method="python",
        two_body_delta=lambda a, b, c, d: delta(a, b, c, d, spinors),
    )
    original = MAT._rk

    def scaled(*args, **kwargs):
        return (1 + scale.get(args[-1], 0)) * original(*args, **kwargs)

    with patch.object(MAT, "_rk", side_effect=scaled):
        reference = SC.assemble_sparse_hamiltonian(
            masks, spinors, od, dhf.bsp_data, 4, dhf.ns, method="python"
        )
    np.testing.assert_allclose(H.toarray(), reference.toarray(), atol=2e-14, rtol=0)


@pytest.mark.parametrize("builder", [build_sigma2_mscheme, build_sigma2_holeline])
def test_sigma2_all_valence_permutation_symmetries(workspace, builder):
    dhf, od, eps = workspace
    delta = builder(
        od,
        dhf.bsp_data,
        4,
        dhf.ns,
        core_keys=[(1, -1)],
        virt_keys=_VIRT,
        valence_keys=_VALENCE,
        eps=eps,
    )
    indices = [i for i, s in enumerate(delta.spinors) if s[:2] in _VALENCE]
    nonzero = 0
    for (a, b), (c, d) in product(combinations(indices, 2), repeat=2):
        value = delta(a, b, c, d)
        nonzero += abs(value) > 1e-12
        assert value == pytest.approx(delta(c, d, a, b), abs=1e-16)
        assert value == pytest.approx(-delta(b, a, c, d), abs=1e-16)
        assert value == pytest.approx(-delta(a, b, d, c), abs=1e-16)
    assert nonzero > 10


def test_rpa_both_matrices_match_explicit_spin_orbital_tensors(workspace):
    dhf, od, eps = workspace
    occupied = [(1, -1)]
    virtual = [(2, 1), (2, -2), (3, 1)]
    result = solve_rpa(od, dhf.bsp_data, eps, occupied, virtual, 2, -1)
    g = _coulomb(od, dhf.bsp_data)
    tensors = []
    for p, h in result.ph_pairs:
        jp, jh = twoj(p[1]), twoj(h[1])
        tensors.append(
            [
                (
                    (*p, m),
                    (*h, m),
                    (-1.0) ** ((jh - m) // 2) * clebsch_gordan(jp, m, jh, -m, 2, 0),
                )
                for m in range(-min(jp, jh), min(jp, jh) + 1, 2)
            ]
        )
    A = np.zeros_like(result.A)
    B = np.zeros_like(result.B)
    for i, t1 in enumerate(tensors):
        for j, t2 in enumerate(tensors):
            for p, h, c1 in t1:
                for pp, hh, c2 in t2:
                    A[i, j] += c1 * c2 * (g(p, hh, h, pp) - g(p, hh, pp, h))
                    B[i, j] += c1 * c2 * (g(p, pp, h, hh) - g(p, pp, hh, h))
        p, h = result.ph_pairs[i]
        A[i, i] += eps[p] - eps[h]
    np.testing.assert_allclose(result.A, A, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result.B, B, atol=1e-12, rtol=0)


def test_rpa_rejects_spatially_overlapping_virtual_with_distinct_label(workspace):
    dhf, od, eps = workspace
    od = dict(od)
    eps = dict(eps)
    od[(99, -1)] = od[(1, -1)]
    eps[(99, -1)] = 0.5
    with pytest.raises(ValueError, match="orthonormal"):
        solve_rpa(od, dhf.bsp_data, eps, [(1, -1)], [(99, -1)], 0, 1)


def test_two_parameter_fit_has_no_unused_coordinate():
    coefficients = np.array([[1.0, 2.0], [3.0, -1.0], [2.0, 1.0]])
    target = np.array([-0.08, -0.119])
    fitted, _, diagnostics = fit_slater_scaling(
        lambda x: coefficients @ (x - target), x0=(0.0, 0.0), tol=1e-12
    )
    np.testing.assert_allclose(fitted, target, atol=1e-12)
    assert np.isfinite(diagnostics["cond"])
    assert len(diagnostics["sigma"]) == 2


def test_ci_mbpt_public_exports_exist():
    import atomx.methods.ci_mbpt as module

    assert all(hasattr(module, name) for name in module.__all__)
