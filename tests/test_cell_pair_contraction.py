"""Batched density action retains signed native four-orbital integrals."""

import itertools

import numpy as np
import pytest

from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.convol import ConvolCache, compute_slater_rk_cell
from atomx.numerics.bspline.coulomb_features import CellCoulombFeatures
from atomx.numerics.bspline.knots import make_knots
from atomx.numerics.bspline.pair_contraction import CellPairContraction


@pytest.mark.parametrize("kinetic_balance", [False, True])
@pytest.mark.parametrize("rank", [0, 3, 7])
def test_full_signed_density_against_public_native_integrals(kinetic_balance, rank):
    ns, ks = 14, 5
    bd = prepare_bspline_data(
        make_knots(ns, ks, 0.02, 4.0), ns, ks, kinetic_balance=kinetic_balance
    )
    feature, cache = CellCoulombFeatures(bd), ConvolCache(bd)
    rng = np.random.default_rng(381)
    waves = [
        [(rng.normal(size=ns), rng.normal(size=ns)) for _ in range(n)]
        for n in (3, 2, 2, 4)
    ]
    samples = [np.column_stack([feature.sample(*a) for a in group]) for group in waves]
    rho = rng.normal(size=(2, 2))
    expected = np.zeros((3, 4))
    for m, r, s, n in itertools.product(range(3), range(2), range(2), range(4)):
        quartet = (waves[0][m], waves[1][r], waves[2][s], waves[3][n])
        expected[m, n] += rho[r, s] * compute_slater_rk_cell(
            *(v for a in quartet for v in a), bd, rank, cache
        )
    a = CellPairContraction(feature, *samples, rank, edge_batch_size=17)
    b = CellPairContraction(feature, *samples, rank, edge_batch_size=4096)
    np.testing.assert_allclose(a(rho), expected, atol=2e-11, rtol=3e-13)
    np.testing.assert_allclose(b(rho), expected, atol=2e-11, rtol=3e-13)
    # No cutoff: keep signed small source coordinates, complex C-products,
    # and caller mutation independence in the same complete source space.
    tiny = np.array([[0.0, 1e-14], [-2e-14, 0.0]])
    np.testing.assert_allclose(
        a(rho + 1j * tiny), a(rho) + 1j * a(tiny), atol=2e-11, rtol=3e-13
    )
    old = a(rho)
    samples[0][:] *= -2
    np.testing.assert_array_equal(a(rho), old)
    for array in (*a._arrays, a._left_rise, a._right_suffix):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_complex_operands_one_cell_and_whole_payload_guards():
    ns = ks = 4
    bd = prepare_bspline_data(np.r_[np.zeros(ks), np.full(ks, 0.7)], ns, ks, n_gauss=5)
    f = CellCoulombFeatures(bd)
    z = f.sample(np.linspace(0, 0.7, ns), np.zeros(ns))[:, None]
    a = CellPairContraction(f, z * (1 + 1j), z * 2, z * 3, z * (1 - 2j), 1)
    expected = 0.7**5 / 5 * (2 / 4) * (1 + 1j) * 2 * 3 * (1 - 2j)
    assert a(np.ones((1, 1)))[0, 0] == pytest.approx(expected, rel=3e-13)
    with pytest.raises(MemoryError, match="complete cell-pair"):
        CellPairContraction(f, z, z, z, z, 0, max_work_bytes=1)
    with pytest.raises(ValueError, match="integer"):
        CellPairContraction(f, z, z, z, z, 0, edge_batch_size=True)
    with pytest.raises(ValueError, match="density matrix"):
        a(np.zeros((1, 2)))
    with pytest.raises(ValueError, match="column"):
        CellPairContraction(f, z[:-1], z, z, z, 0)
