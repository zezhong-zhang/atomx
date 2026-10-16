"""Independent native convolution and analytic anchors for auxiliary nodes."""

import itertools

import numpy as np
import pytest

from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.convol import ConvolCache, compute_slater_rk_cell
from atomx.numerics.bspline.coulomb_features import (
    CellCoulombFeatures,
    PolynomialCellCoulombFeatures,
)
from atomx.numerics.bspline.knots import make_knots


@pytest.mark.parametrize("kinetic_balance", [False, True])
@pytest.mark.parametrize("rank", [0, 1, 2, 3, 7])
@pytest.mark.parametrize(
    "features_type", [CellCoulombFeatures, PolynomialCellCoulombFeatures]
)
def test_signed_quartets_match_full_native_cell_integral(
    kinetic_balance, rank, features_type
):
    ns, ks = 16, 5
    bd = prepare_bspline_data(
        make_knots(ns, ks, 0.02, 4.0), ns, ks, kinetic_balance=kinetic_balance
    )
    features = features_type(bd)
    cache = ConvolCache(bd)
    rng = np.random.default_rng(781)
    orbitals = [(rng.normal(size=ns), rng.normal(size=ns)) for _ in range(4)]
    expected = compute_slater_rk_cell(
        *(v for pair in orbitals for v in pair), bd, rank, cache
    )
    actual = features.integral(orbitals, rank)
    assert actual == pytest.approx(expected, abs=1e-11, rel=1e-13)
    # Explicitly isolate every P/P, P/Q, Q/P, Q/Q density contribution.
    parts = []
    for left, right in itertools.product(range(2), repeat=2):
        selected = [
            (
                p if (left if i < 2 else right) == 0 else np.zeros(ns),
                q if (left if i < 2 else right) == 1 else np.zeros(ns),
            )
            for i, (p, q) in enumerate(orbitals)
        ]
        ref = compute_slater_rk_cell(
            *(v for pair in selected for v in pair), bd, rank, cache
        )
        got = features.integral(selected, rank)
        assert got == pytest.approx(ref, abs=1e-11, rel=1e-13)
        parts.append(got)
    assert sum(parts) == pytest.approx(actual, abs=1e-11, rel=1e-13)
    a, b, c, d = [features.sample(*pair) for pair in orbitals]
    assert np.dot(a * b, features.apply_density(c * d, rank)) == pytest.approx(
        np.dot(c * d, features.apply_density(a * b, rank)), abs=1e-11
    )


@pytest.mark.parametrize(
    "features_type", [CellCoulombFeatures, PolynomialCellCoulombFeatures]
)
def test_complex_cproduct_linearity_and_immutable_geometry(features_type):
    ns, ks = 12, 4
    bd = prepare_bspline_data(
        make_knots(ns, ks, 0.02, 3.0), ns, ks, kinetic_balance=True
    )
    f = features_type(bd, rank_cache_size=1)
    rng = np.random.default_rng(213)
    a, b = rng.normal(size=len(f.radii)), rng.normal(size=len(f.radii))
    complex_action = f.apply_density(a + 1j * b, 2)
    np.testing.assert_allclose(
        complex_action, f.apply_density(a, 2) + 1j * f.apply_density(b, 2), atol=1e-12
    )
    assert np.dot(a + 1j * b, complex_action) == pytest.approx(
        np.dot(a, f.apply_density(a, 2))
        - np.dot(b, f.apply_density(b, 2))
        + 2j * np.dot(a, f.apply_density(b, 2)),
        abs=1e-12,
    )
    before = f.apply_density(a, 2)
    bd["gr"] *= 2
    np.testing.assert_array_equal(before, f.apply_density(a, 2))
    for array in (f.radii, f.outer_indices, f.edge_left, *f.rank_arrays(0)):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    assert len(f._rank_cache) == 1


@pytest.mark.parametrize(
    "features_type", [CellCoulombFeatures, PolynomialCellCoulombFeatures]
)
def test_origin_cell_monomial_double_integral(features_type):
    # Constant spline coefficients represent constant P exactly, including
    # the origin cell. R^k(r^p,r^q) = R^(p+q+1)/(p+q+1)
    # times [1/(p+k+1)+1/(q+k+1)]. Use p=q=2 for integrability.
    ns = ks = 4
    radius = 0.7
    t = np.r_[np.zeros(ks), np.full(ks, radius)]
    bd = prepare_bspline_data(t, ns, ks, n_gauss=5)
    features = features_type(bd)
    # Linear B-spline function P(r)=r in a single Bernstein cell.
    p = np.linspace(0, radius, ns)
    q = np.zeros(ns)
    for rank in (0, 1, 3):
        expected = radius**5 / 5 * (2 / (3 + rank))
        assert features.integral([(p, q)] * 4, rank) == pytest.approx(
            expected, rel=2e-13
        )


def test_whole_payload_and_geometry_guards():
    ns, ks = 12, 4
    bd = prepare_bspline_data(make_knots(ns, ks, 0.02, 3.0), ns, ks)
    with pytest.raises(MemoryError, match="complete auxiliary"):
        CellCoulombFeatures(bd, memory_bytes=1)
    with pytest.raises(ValueError, match="integer"):
        CellCoulombFeatures(bd, rank_cache_size=True)
    f = CellCoulombFeatures(bd)
    with pytest.raises(ValueError, match="density"):
        f.apply_density(np.ones(2), 0)
    with pytest.raises(ValueError, match="multipole"):
        f.apply_density(np.ones(len(f.radii)), -1)
    with pytest.raises(MemoryError, match="rank-aware"):
        f.rank_arrays(100_000_000)
    with pytest.raises(ValueError, match="coefficient"):
        f.sample(np.ones(ns - 1), np.ones(ns))
    bd["bsp_p"][0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="basis tables"):
        CellCoulombFeatures(bd)


def test_polynomial_payload_includes_construction_and_owned_maps():
    ns, ks = 12, 4
    bd = prepare_bspline_data(
        make_knots(ns, ks, 0.02, 3.0), ns, ks, kinetic_balance=True
    )
    native = CellCoulombFeatures(bd)
    transformed = PolynomialCellCoulombFeatures(bd)
    assert transformed.identity["auxiliary_nodes"] < native.identity["auxiliary_nodes"]
    required = transformed.identity["payload_estimate_bytes"]
    assert required > native.identity["payload_estimate_bytes"]
    with pytest.raises(MemoryError, match="complete polynomial"):
        PolynomialCellCoulombFeatures(bd, memory_bytes=required - 1)
    assert (
        PolynomialCellCoulombFeatures(bd, memory_bytes=required).identity
        == transformed.identity
    )
    with pytest.raises(ValueError):
        transformed._interpolation.setflags(write=True)
